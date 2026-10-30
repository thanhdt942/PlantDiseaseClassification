"""Training and evaluation routines for PyTorch models with AMP, Gradient Accumulation, and Checkpointing."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    grad_accum_steps: int = 1,
    use_fp16: bool = False,
) -> Dict[str, float]:
    """Train model for one epoch with optional Mixed Precision and Gradient Accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    optimizer.zero_grad()
    use_amp = use_fp16 and (device.type == "cuda")

    for step, (images, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "train_time": epoch_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    use_fp16: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on validation or test dataloader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    start_time = time.time()
    use_amp = use_fp16 and (device.type == "cuda")

    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    eval_loss = running_loss / total if total > 0 else 0.0
    eval_time = time.time() - start_time

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    metrics["loss"] = eval_loss
    metrics["eval_time"] = eval_time
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred

    return metrics


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "last_checkpoint.pth",
    is_best: bool = False,
) -> None:
    """Save training checkpoint to disk with atomic write and sync."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / filename
    
    # Save checkpoint
    torch.save(state, save_path)
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pth"
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint weights and training state (including optimizer and scheduler)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def append_log_sync(log_path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Append a dictionary record to a CSV log file and force immediate disk synchronization (os.fsync).
    Essential for Google Drive FUSE filesystem to prevent loss during sudden Colab disconnections.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists() and log_path.stat().st_size > 0

    import csv
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        f.flush()
        os.fsync(f.fileno())
