#!/usr/bin/env python3
"""Plot and diagnose learning curves from existing training logs (no retraining needed).

For every `<exp_name>_log.csv` found in the log directory, this script:
  1. Plots train_loss vs val_loss per epoch (marks the best validation-loss epoch).
  2. Plots train F1 vs val_macro_f1 per epoch (shades the generalization gap).
  3. Prints a diagnosis per experiment:
       - SUSPECTED OVERFITTING : train loss keeps decreasing while val loss rises after its minimum.
       - LARGE F1 GAP          : train and validation F1 differ by more than --gap_tol.
       - GOOD AND STABLE       : both curves are good and flat (positive signal, NOT proof of
                                 generalization; still evaluate on the frozen test set).
       - INCONCLUSIVE          : none of the above clearly applies.

About train F1:
  Older logs do not contain a per-epoch `train_macro_f1` column. In that case the script falls
  back to `train_acc` and labels it clearly as a proxy. Per-epoch train macro-F1 cannot be
  recovered afterwards because only best/last checkpoints are saved. Use --eval_train_f1 to
  compute the exact train macro-F1 of the best checkpoint (one point, evaluated without
  augmentation), which is plotted as a star marker.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --experiments resnet50 vit_base_patch16_224
    python scripts/plot_training_curves.py --eval_train_f1 --data_root "<path to color dir>"
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Headless backend; works on Colab, local terminals and servers
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def diagnose(df: pd.DataFrame, loss_rise_tol: float, loss_rise_abs: float, gap_tol: float,
             good_f1: float, stable_std: float, tail: int) -> Dict[str, Any]:
    """Compute overfitting / gap / stability indicators from a per-epoch log."""
    df = df.sort_values("epoch").reset_index(drop=True)
    has_train_f1 = "train_macro_f1" in df.columns
    train_metric_col = "train_macro_f1" if has_train_f1 else "train_acc"

    best_idx = int(df["val_loss"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    last_epoch = int(df["epoch"].iloc[-1])
    min_val_loss = float(df.loc[best_idx, "val_loss"])
    final_val_loss = float(df["val_loss"].iloc[-1])
    val_loss_rise = (final_val_loss - min_val_loss) / max(min_val_loss, 1e-12)
    train_loss_drop_after_best = float(df.loc[best_idx, "train_loss"] - df["train_loss"].iloc[-1])

    final_train_metric = float(df[train_metric_col].iloc[-1])
    final_val_f1 = float(df["val_macro_f1"].iloc[-1])
    f1_gap = final_train_metric - final_val_f1
    tail_df = df.tail(min(tail, len(df)))
    val_f1_tail_std = float(tail_df["val_macro_f1"].std(ddof=0))
    val_loss_tail_std = float(tail_df["val_loss"].std(ddof=0))

    abs_rise = final_val_loss - min_val_loss
    # Like-for-like accuracy gap (useful when train F1 is only available as an accuracy proxy)
    acc_gap = (float(df["train_acc"].iloc[-1] - df["val_acc"].iloc[-1])
               if {"train_acc", "val_acc"}.issubset(df.columns) else None)

    pattern = best_epoch < last_epoch and train_loss_drop_after_best > 0 and val_loss_rise > loss_rise_tol
    # Require both a relative and an absolute rise: with very small losses a tiny change
    # (e.g. 0.017 -> 0.019) is a large percentage but not a meaningful divergence.
    overfitting = pattern and abs_rise > loss_rise_abs
    mild_uptick = pattern and not overfitting
    large_gap = abs(f1_gap) > gap_tol
    good_and_stable = (not overfitting and not large_gap and final_val_f1 >= good_f1
                       and val_f1_tail_std < stable_std)

    findings: List[str] = []
    if overfitting:
        findings.append(
            f"SUSPECTED OVERFITTING: val_loss rose {abs_rise:.4f} ({val_loss_rise * 100:.1f}%) above its "
            f"minimum (epoch {best_epoch}) while train_loss kept dropping by {train_loss_drop_after_best:.4f}. "
            f"Consider early stopping at epoch {best_epoch}, stronger regularization or augmentation."
        )
    if mild_uptick:
        findings.append(
            f"NOTE: mild val_loss uptick after epoch {best_epoch} (+{abs_rise:.4f}, "
            f"{val_loss_rise * 100:.1f}%) while train_loss still decreases. Early sign worth watching; "
            f"the best checkpoint (epoch {best_epoch}) is the one to keep."
        )
    if large_gap:
        proxy_note = ""
        if not has_train_f1:
            proxy_note = (" Note: this compares train ACCURACY with val MACRO-F1, which overstates the gap"
                          + (f"; like-for-like accuracy gap is {acc_gap:+.4f}." if acc_gap is not None else "."))
        findings.append(
            f"LARGE F1 GAP: final train {'macro-F1' if has_train_f1 else 'accuracy (proxy)'} "
            f"{final_train_metric:.4f} vs val macro-F1 {final_val_f1:.4f} (gap {f1_gap:+.4f}). "
            f"Check for overfitting or a distribution difference between train and val "
            f"(inspect per-class F1 and the split procedure).{proxy_note}"
        )
    if good_and_stable:
        findings.append(
            f"GOOD AND STABLE: val macro-F1 {final_val_f1:.4f}, std over last {len(tail_df)} epochs "
            f"{val_f1_tail_std:.4f}. Positive signal, but NOT proof of generalization: confirm on "
            f"the frozen test set and, ideally, on out-of-distribution (in-the-wild) images."
        )
    if not findings:
        findings.append("INCONCLUSIVE: no rule triggered clearly. Inspect the curves manually.")

    if overfitting:
        verdict = "SUSPECTED OVERFITTING"
    elif large_gap:
        verdict = "LARGE F1 GAP"
    elif good_and_stable:
        verdict = "GOOD AND STABLE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "findings": findings,
        "train_metric_col": train_metric_col,
        "train_metric_is_proxy": not has_train_f1,
        "best_val_loss_epoch": best_epoch,
        "last_epoch": last_epoch,
        "min_val_loss": min_val_loss,
        "final_val_loss": final_val_loss,
        "val_loss_rise_pct": val_loss_rise * 100,
        "val_loss_rise_abs": abs_rise,
        "acc_gap": acc_gap,
        "train_loss_drop_after_best": train_loss_drop_after_best,
        "final_train_metric": final_train_metric,
        "final_val_macro_f1": final_val_f1,
        "f1_gap": f1_gap,
        "val_f1_tail_std": val_f1_tail_std,
        "val_loss_tail_std": val_loss_tail_std,
    }


# ---------------------------------------------------------------------------
# Optional: exact train macro-F1 of the best checkpoint
# ---------------------------------------------------------------------------
def eval_train_macro_f1(ckpt_path: Path, data_root: Path, split_dir: Path, batch_size: int,
                        num_workers: int, max_samples: Optional[int]) -> Dict[str, float]:
    """Evaluate a checkpoint on the training split (no augmentation, eval mode)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.dataset import PlantVillageDataset, get_default_transforms
    from src.engine import evaluate
    from src.models import get_model
    from src.utils import get_device

    device = get_device()
    # weights_only=False: checkpoints also store history/args (trusted, self-generated files)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    train_df = pd.read_csv(split_dir / "train.csv")
    num_classes = int(train_df["class_id"].nunique())
    if max_samples:
        train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=42)

    img_size = int(ckpt_args.get("image_size", 224))
    dataset = PlantVillageDataset(train_df, data_root=data_root,
                                  transform=get_default_transforms(img_size=img_size, is_train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = get_model(model_name=ckpt["model_name"], num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    res = evaluate(model, loader, nn.CrossEntropyLoss(), device,
                   class_names=None, use_fp16=(device.type == "cuda"))
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "train_macro_f1_eval": float(res["macro_f1"]),
        "train_loss_eval": float(res["loss"]),
        "n_samples": len(train_df),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_experiment(exp_name: str, df: pd.DataFrame, diag: Dict[str, Any],
                    exact_point: Optional[Dict[str, float]], out_path: Path) -> None:
    df = df.sort_values("epoch")
    epochs = df["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss panel
    ax1.plot(epochs, df["train_loss"], "o-", label="Train loss")
    ax1.plot(epochs, df["val_loss"], "s-", label="Val loss")
    ax1.axvline(diag["best_val_loss_epoch"], color="gray", linestyle="--",
                label=f"Best val loss (epoch {diag['best_val_loss_epoch']})")
    if exact_point:
        ax1.plot(exact_point["epoch"], exact_point["train_loss_eval"], "*", markersize=16,
                 color="red", label="Train loss, eval mode (best ckpt)")
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # F1 panel
    train_col = diag["train_metric_col"]
    train_label = "Train macro-F1" if not diag["train_metric_is_proxy"] else "Train accuracy (proxy for F1)"
    ax2.plot(epochs, df[train_col], "o-", label=train_label)
    ax2.plot(epochs, df["val_macro_f1"], "s-", label="Val macro-F1")
    ax2.fill_between(epochs, df[train_col], df["val_macro_f1"], alpha=0.15, label="Gap")
    if exact_point:
        ax2.plot(exact_point["epoch"], exact_point["train_macro_f1_eval"], "*", markersize=16,
                 color="red", label="Exact train macro-F1 (best ckpt)")
    ax2.set_title("Train vs validation F1 per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle(f"{exp_name}  |  Diagnosis: {diag['verdict']}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot and diagnose learning curves from training logs.")
    p.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "logs"))
    p.add_argument("--fig_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "figures"))
    p.add_argument("--table_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "tables"))
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Experiment names to include (default: all *_log.csv files)")
    p.add_argument("--loss_rise_tol", type=float, default=0.05,
                   help="Relative rise of final val_loss over its minimum to flag overfitting (default 5%%)")
    p.add_argument("--loss_rise_abs", type=float, default=0.005,
                   help="Minimum absolute val_loss rise also required to flag overfitting (default 0.005)")
    p.add_argument("--gap_tol", type=float, default=0.03,
                   help="Absolute train-vs-val F1 gap to flag as large (default 0.03)")
    p.add_argument("--good_f1", type=float, default=0.95, help="Minimum val macro-F1 to call 'good'")
    p.add_argument("--stable_std", type=float, default=0.005,
                   help="Max std of val macro-F1 over the last epochs to call 'stable'")
    p.add_argument("--tail", type=int, default=3, help="Number of final epochs used for stability")
    # Optional exact train macro-F1 evaluation
    p.add_argument("--eval_train_f1", action="store_true",
                   help="Evaluate best checkpoint on the train split to get exact train macro-F1")
    p.add_argument("--ckpt_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "checkpoints"))
    p.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"))
    p.add_argument("--split_dir", type=str, default=str(PROJECT_ROOT / "data" / "split"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Subsample the train split for a quicker (approximate) evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    log_dir, fig_dir, table_dir = Path(args.log_dir), Path(args.fig_dir), Path(args.table_dir)

    log_files = sorted(log_dir.glob("*_log.csv"))
    if args.experiments:
        wanted = set(args.experiments)
        log_files = [f for f in log_files if f.stem[: -len("_log")] in wanted]
    if not log_files:
        print(f"No *_log.csv files found in {log_dir}. Train a model first or pass --log_dir.")
        sys.exit(1)

    rows = []
    for log_path in log_files:
        exp_name = log_path.stem[: -len("_log")]
        df = pd.read_csv(log_path)
        required = {"epoch", "train_loss", "val_loss", "val_macro_f1"}
        if not required.issubset(df.columns):
            print(f"Skipping {exp_name}: missing columns {required - set(df.columns)}")
            continue
        # Resumed runs may append duplicated epochs; keep the most recent record per epoch
        df = df.drop_duplicates(subset="epoch", keep="last")

        diag = diagnose(df, args.loss_rise_tol, args.loss_rise_abs, args.gap_tol,
                        args.good_f1, args.stable_std, args.tail)

        exact_point = None
        if args.eval_train_f1:
            ckpt_path = Path(args.ckpt_dir) / exp_name / "best_checkpoint.pth"
            if ckpt_path.is_file():
                print(f"[{exp_name}] Evaluating best checkpoint on train split...")
                exact_point = eval_train_macro_f1(ckpt_path, Path(args.data_root), Path(args.split_dir),
                                                  args.batch_size, args.num_workers, args.max_samples)
                best_row = df[df["epoch"] == exact_point["epoch"]]
                if not best_row.empty:
                    exact_point["val_macro_f1_same_epoch"] = float(best_row["val_macro_f1"].iloc[0])
            else:
                print(f"[{exp_name}] No checkpoint at {ckpt_path}; skipping exact train F1.")

        fig_path = fig_dir / f"{exp_name}_learning_curves.png"
        plot_experiment(exp_name, df, diag, exact_point, fig_path)

        print("\n" + "=" * 78)
        print(f"Experiment: {exp_name}   ->   {diag['verdict']}")
        print("=" * 78)
        print(f"Best val loss: {diag['min_val_loss']:.4f} at epoch {diag['best_val_loss_epoch']} | "
              f"final val loss: {diag['final_val_loss']:.4f} ({diag['val_loss_rise_pct']:+.1f}%)")
        metric_name = "train acc (proxy)" if diag["train_metric_is_proxy"] else "train macro-F1"
        print(f"Final {metric_name}: {diag['final_train_metric']:.4f} | final val macro-F1: "
              f"{diag['final_val_macro_f1']:.4f} | gap: {diag['f1_gap']:+.4f}")
        if diag["acc_gap"] is not None:
            print(f"Like-for-like accuracy gap (train_acc - val_acc): {diag['acc_gap']:+.4f}")
        if exact_point:
            same = exact_point.get("val_macro_f1_same_epoch")
            gap_txt = f" | exact gap: {exact_point['train_macro_f1_eval'] - same:+.4f}" if same is not None else ""
            print(f"Exact train macro-F1 (best ckpt, epoch {exact_point['epoch']}, "
                  f"n={exact_point['n_samples']}): {exact_point['train_macro_f1_eval']:.4f}{gap_txt}")
        for finding in diag["findings"]:
            print(f"  - {finding}")
        print(f"Figure saved: {fig_path}")

        row = {"experiment": exp_name, **{k: v for k, v in diag.items() if k != "findings"}}
        if exact_point:
            row.update({f"exact_{k}": v for k, v in exact_point.items()})
        rows.append(row)

    if rows:
        table_dir.mkdir(parents=True, exist_ok=True)
        summary_path = table_dir / "learning_curve_diagnostics.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nDiagnostics summary saved: {summary_path}")


if __name__ == "__main__":
    main()
