#!/usr/bin/env python3
"""Unified training script for Plant Disease Classification.
Supports configuration loading from YAML files (configs/default_config.yaml and configs/models/*.yaml),
ResNet, Vision Transformer (ViT), EfficientNet, AMP FP16, Fault-Tolerant Checkpointing, and Auto-Skip.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PlantVillageDataset, get_default_transforms
from src.models import get_model, count_parameters
from src.engine import train_one_epoch, evaluate, save_checkpoint, load_checkpoint, append_log_sync
from src.utils import set_seed, get_device


def load_yaml(file_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    if not file_path:
        return {}
    p = Path(file_path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def find_model_config(model_name: str, config_dir: Union[str, Path] = "configs/models") -> Optional[Path]:
    """Auto-detect corresponding YAML file for a model name in configs/models."""
    p_dir = Path(config_dir)
    if not p_dir.is_dir():
        p_dir = PROJECT_ROOT / config_dir
    if not p_dir.is_dir():
        return None

    # 1. Exact filename match: e.g. resnet50 -> resnet50.yaml
    candidate = p_dir / f"{model_name}.yaml"
    if candidate.is_file():
        return candidate

    # 2. Check model.name inside all YAML files
    for yml in sorted(p_dir.glob("*.yaml")):
        cfg = load_yaml(yml)
        if cfg.get("model", {}).get("name") == model_name:
            return yml

    # 3. Normalized stem substring match (e.g. vit_base -> vit_base_patch16_224 or vice versa)
    norm = model_name.lower().replace("-", "_")
    for yml in sorted(p_dir.glob("*.yaml")):
        stem = yml.stem.lower().replace("-", "_")
        if stem in norm or norm in stem:
            return yml

    return None


def parse_args():
    """Two-stage argument parser:
    1. Loads global default config from configs/default_config.yaml
    2. Loads model-specific config from configs/models/*.yaml (specified by --config or auto-detected by --model_name)
    3. Allows CLI flags to override configuration values dynamically.
    """
    # First pass: parse config file paths and model name
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="Path to specific model config YAML (e.g. configs/models/resnet50.yaml)")
    pre_parser.add_argument("--default_config", type=str, default=None,
                            help="Path to default project config YAML (e.g. configs/default_config.yaml)")
    pre_parser.add_argument("--model_name", type=str, default=None,
                            help="Model architecture name to auto-detect config file")
    pre_args, _ = pre_parser.parse_known_args()

    # 1. Resolve default_config path
    default_config_path = None
    if pre_args.default_config:
        p = Path(pre_args.default_config)
        default_config_path = p if p.is_file() else (PROJECT_ROOT / pre_args.default_config)
    else:
        for cand in [Path("configs/default_config.yaml"), PROJECT_ROOT / "configs/default_config.yaml"]:
            if cand.is_file():
                default_config_path = cand
                break

    default_cfg = load_yaml(default_config_path)

    # Defaults from default_config.yaml
    cfg_seed = default_cfg.get("project", {}).get("seed", 42)
    cfg_batch_size = default_cfg.get("data", {}).get("batch_size", 32)
    cfg_num_workers = default_cfg.get("data", {}).get("num_workers", 4)
    cfg_image_size = default_cfg.get("data", {}).get("image_size", 224)
    cfg_split_dir = default_cfg.get("paths", {}).get("data_splits", "data/split")
    cfg_output_dir = default_cfg.get("paths", {}).get("outputs_checkpoints", "outputs/checkpoints")
    cfg_log_dir = default_cfg.get("paths", {}).get("outputs_logs", "outputs/logs")
    cfg_table_dir = default_cfg.get("paths", {}).get("outputs_tables", "outputs/tables")

    # 2. Resolve model config path
    model_config_path = None
    if pre_args.config:
        p = Path(pre_args.config)
        model_config_path = p if p.is_file() else (PROJECT_ROOT / pre_args.config)
    elif pre_args.model_name:
        model_config_path = find_model_config(pre_args.model_name)
    else:
        # Default to resnet50
        model_config_path = find_model_config("resnet50")

    model_cfg = load_yaml(model_config_path)

    # If model_cfg contains project/data/paths sections (monolithic config), merge them
    if "project" in model_cfg:
        cfg_seed = model_cfg["project"].get("seed", cfg_seed)
    if "data" in model_cfg:
        cfg_batch_size = model_cfg["data"].get("batch_size", cfg_batch_size)
        cfg_num_workers = model_cfg["data"].get("num_workers", cfg_num_workers)
        cfg_image_size = model_cfg["data"].get("image_size", cfg_image_size)
    if "paths" in model_cfg:
        cfg_split_dir = model_cfg["paths"].get("data_splits", cfg_split_dir)
        cfg_output_dir = model_cfg["paths"].get("outputs_checkpoints", cfg_output_dir)
        cfg_log_dir = model_cfg["paths"].get("outputs_logs", cfg_log_dir)
        cfg_table_dir = model_cfg["paths"].get("outputs_tables", cfg_table_dir)

    model_sec = model_cfg.get("model", {})
    train_sec = model_cfg.get("training", {})

    cfg_model_name = pre_args.model_name or model_sec.get("name", "resnet50")
    cfg_pretrained = model_sec.get("pretrained", True)
    cfg_epochs = train_sec.get("epochs", 10)
    cfg_lr = float(train_sec.get("lr", 1e-4))
    cfg_weight_decay = float(train_sec.get("weight_decay", 1e-4))

    # 3. Main ArgumentParser with dynamic defaults derived from YAML configs
    parser = argparse.ArgumentParser(
        description="Unified training script for Plant Disease Classification (Config-driven)."
    )
    parser.add_argument("--config", type=str,
                        default=str(model_config_path) if model_config_path and model_config_path.is_file() else None,
                        help="Path to model config YAML file (e.g. configs/models/resnet50.yaml)")
    parser.add_argument("--default_config", type=str,
                        default=str(default_config_path) if default_config_path and default_config_path.is_file() else None,
                        help="Path to default project config YAML file (e.g. configs/default_config.yaml)")
    parser.add_argument("--model_name", type=str, default=cfg_model_name,
                        help=f"Model architecture (default from config: '{cfg_model_name}')")
    parser.add_argument("--data_root", type=str, default="",
                        help="Path to the 'color' directory. If empty, uses cached path.")
    parser.add_argument("--split_dir", type=str, default=cfg_split_dir,
                        help=f"Directory containing train.csv, val.csv, test.csv (default: '{cfg_split_dir}')")
    parser.add_argument("--output_dir", type=str, default=cfg_output_dir,
                        help=f"Directory to save model checkpoints (default: '{cfg_output_dir}')")
    parser.add_argument("--log_dir", type=str, default=cfg_log_dir,
                        help=f"Directory to save training logs (default: '{cfg_log_dir}')")
    parser.add_argument("--table_dir", type=str, default=cfg_table_dir,
                        help=f"Directory to save evaluation tables (default: '{cfg_table_dir}')")
    parser.add_argument("--batch_size", type=int, default=cfg_batch_size,
                        help=f"Input batch size (default: {cfg_batch_size})")
    parser.add_argument("--image_size", type=int, default=cfg_image_size,
                        help=f"Input image resolution (default: {cfg_image_size})")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=cfg_epochs,
                        help=f"Number of training epochs (default: {cfg_epochs})")
    parser.add_argument("--lr", type=float, default=cfg_lr,
                        help=f"Learning rate (default: {cfg_lr})")
    parser.add_argument("--weight_decay", type=float, default=cfg_weight_decay,
                        help=f"Weight decay (default: {cfg_weight_decay})")
    parser.add_argument("--pretrained", type=bool, default=cfg_pretrained,
                        help=f"Use pretrained weights (default: {cfg_pretrained})")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable Mixed Precision training (FP16)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze feature extractor backbone (E4)")
    parser.add_argument("--no_aug", action="store_true",
                        help="Disable training data augmentation (E5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_checkpoint.pth if exists or auto-skip if completed")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (runs 1 epoch on 64 train / 32 val samples)")
    parser.add_argument("--seed", type=int, default=cfg_seed,
                        help=f"Random seed for reproducibility (default: {cfg_seed})")
    parser.add_argument("--num_workers", type=int, default=cfg_num_workers,
                        help=f"Number of dataloader workers (default: {cfg_num_workers})")

    parsed_args = parser.parse_args()
    return parsed_args, default_config_path, model_config_path


def main():
    args, default_config_path, model_config_path = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Determine Experiment Name
    exp_name = args.model_name
    if args.freeze_backbone:
        exp_name += "_frozen"
    if args.no_aug:
        exp_name += "_noaug"
    if args.debug:
        exp_name += "_debug"

    ckpt_dir = Path(args.output_dir) / exp_name
    log_dir = Path(args.log_dir)
    table_dir = Path(args.table_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    target_epochs = 1 if args.debug else args.epochs
    last_ckpt_path = ckpt_dir / "last_checkpoint.pth"
    test_metrics_path = table_dir / f"{exp_name}_test_metrics.json"

    # --- AUTO-SKIP CHECK (QUICK EXIT IF EXPERIMENT ALREADY COMPLETED) ---
    if args.resume:
        is_completed = False
        if test_metrics_path.exists():
            is_completed = True
        elif last_ckpt_path.exists():
            try:
                probe = torch.load(last_ckpt_path, map_location="cpu")
                if probe.get("epoch", 0) >= target_epochs:
                    is_completed = True
            except Exception:
                pass

        if is_completed:
            print(f"\n==================================================================")
            print(f"⏩ [AUTO-SKIP] Thí nghiệm '{exp_name}' đã hoàn tất 100% ({target_epochs}/{target_epochs} epochs)!")
            print(f"Bỏ qua huấn luyện để chuyển sang thí nghiệm tiếp theo.")
            print(f"Checkpoint: {last_ckpt_path}")
            print(f"==================================================================\n")
            sys.exit(0)

    print(f"=== TRAINING CONFIGURATION ===")
    print(f"Config (Default): {default_config_path if default_config_path else 'None'}")
    print(f"Config (Model):   {model_config_path if model_config_path else 'None'}")
    print(f"Experiment:       {exp_name}")
    print(f"Model:            {args.model_name}")
    print(f"Device:           {device}")
    print(f"Image Size:       {args.image_size}")
    print(f"Pretrained:       {args.pretrained}")
    print(f"Mixed Precision:  {args.fp16}")
    print(f"Freeze Backbone:  {args.freeze_backbone}")
    print(f"No Augmentation:  {args.no_aug}")
    print(f"Batch size:       {args.batch_size} (Grad Accum: {args.grad_accum_steps})")
    print(f"Target Epochs:    {target_epochs} | LR: {args.lr} | Weight Decay: {args.weight_decay}")

    # Resolve data_root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(os.path.expanduser(
            "~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"
        ))
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    print(f"DATA_ROOT: {data_root}")

    # Load splits
    split_dir = Path(args.split_dir)
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")

    num_classes = 38
    class_names = [f"Class_{i}" for i in range(num_classes)]
    if "label" in train_df.columns and "class_id" in train_df.columns:
        mapping = train_df[["class_id", "label"]].drop_duplicates().sort_values("class_id")
        class_names = mapping["label"].tolist()

    if args.debug:
        print("[DEBUG MODE] Subsetting 64 random train and 32 random val samples across classes.")
        train_df = train_df.sample(n=min(64, len(train_df)), random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(32, len(val_df)), random_state=args.seed).reset_index(drop=True)
        args.num_workers = 0

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Target Classes: {num_classes}")

    # Datasets and Loaders
    train_transform = get_default_transforms(img_size=args.image_size, is_train=True, no_aug=args.no_aug)
    val_transform = get_default_transforms(img_size=args.image_size, is_train=False)

    train_ds = PlantVillageDataset(train_df, data_root=data_root, transform=train_transform)
    val_ds = PlantVillageDataset(val_df, data_root=data_root, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if (args.fp16 and device.type == "cuda") else None

    start_epoch = 1
    best_val_macro_f1 = 0.0
    history = []

    # Resume capability with full state restore
    if args.resume and last_ckpt_path.exists():
        print(f"\n🔄 [RESUME] Phát hiện checkpoint dở dang tại: {last_ckpt_path}")
        ckpt_state = load_checkpoint(
            checkpoint_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = ckpt_state.get("epoch", 0) + 1
        best_val_macro_f1 = ckpt_state.get("best_val_macro_f1", 0.0)
        history = ckpt_state.get("history", [])
        print(f"Khôi phục thành công! Chạy tiếp từ epoch {start_epoch}/{target_epochs} (Best Val Macro-F1 hiện tại: {best_val_macro_f1:.4f})\n")

    # Training Loop
    total_train_start = time.time()
    csv_log_path = log_dir / f"{exp_name}_log.csv"

    for epoch in range(start_epoch, target_epochs + 1):
        print(f"\n--- Epoch {epoch}/{target_epochs} ---")
        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            grad_accum_steps=args.grad_accum_steps,
            use_fp16=args.fp16,
        )

        val_res = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            use_fp16=args.fp16,
        )

        is_best = val_res["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_res["macro_f1"]

        print(f"Train Loss: {train_res['train_loss']:.4f} | Train Acc: {train_res['train_acc']*100:.2f}% | Time: {train_res['train_time']:.1f}s")
        print(f"Val Loss:   {val_res['loss']:.4f} | Val Acc:   {val_res['accuracy']*100:.2f}% | Val Macro-F1: {val_res['macro_f1']:.4f}{' (BEST)' if is_best else ''}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "train_acc": train_res["train_acc"],
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_weighted_f1": val_res["weighted_f1"],
            "epoch_time": train_res["train_time"] + val_res["eval_time"],
        }
        history.append(epoch_record)

        # 1. Full State Checkpoint saving (overwriting last_checkpoint.pth)
        ckpt_state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_macro_f1": best_val_macro_f1,
            "history": history,
            "args": vars(args),
            "configs": {
                "default_config": str(default_config_path) if default_config_path else None,
                "model_config": str(model_config_path) if model_config_path else None,
            },
        }
        save_checkpoint(ckpt_state, ckpt_dir, filename="last_checkpoint.pth", is_best=is_best)

        # 2. Append Log with Immediate Sync (os.fsync)
        append_log_sync(csv_log_path, epoch_record)

    total_training_time = time.time() - total_train_start
    print(f"\nTraining completed in {total_training_time:.2f}s! Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Saved checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()
