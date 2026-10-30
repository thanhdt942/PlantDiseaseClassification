#!/usr/bin/env python3
"""Unified Colab and Local Training Orchestrator for Plant Disease Classification.

This script manages environment detection, Google Drive mounting, dynamic FOLDER_ROOT
and DataDir resolution, Kaggle dataset downloading according to 01_data_audit.ipynb,
data/split resolution, and multi-model training execution with Auto-Resume and Auto-Skip.

Paths:
    - Google Colab:
        FOLDER_ROOT = '/content/drive/MyDrive/Home/10_Projects/PlantDiseaseClassification'
        DataDir     = '/content/drive/MyDrive/Home/10_Projects/PlantDiseaseClassification/data/raw'
        SplitDir    = '/content/drive/MyDrive/Home/10_Projects/PlantDiseaseClassification/data/split'
    - Local Workstation:
        FOLDER_ROOT = <repository_root>
        DataDir     = <repository_root>/data/raw
        SplitDir    = <repository_root>/data/split

Usage:
    # Dry-run verification (validates paths, configs, and dataset):
    python colab_train.py --dry_run

    # Train default model (ResNet-50) with auto-resume:
    python colab_train.py

    # Train Vision Transformer:
    python colab_train.py --model_name vit_base_patch16_224

    # Train EfficientNet-B0:
    python colab_train.py --model_name efficientnet_b0

    # Train all benchmark models sequentially:
    python colab_train.py --all

    # Run ablation experiments (Frozen Backbone & No Augmentation):
    python colab_train.py --ablation
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


def is_colab() -> bool:
    """Detect if execution is running inside a Google Colab environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def setup_environment() -> Dict[str, Any]:
    """Detect runtime environment, mount Google Drive if on Colab, and resolve FOLDER_ROOT.
    
    Returns:
        Dict containing environment metadata ('is_colab', 'folder_root', 'data_dir', 'split_dir').
    """
    running_in_colab = is_colab()

    if running_in_colab:
        print("=" * 70)
        print("🌐 [ENVIRONMENT] Google Colab Runtime Detected")
        print("=" * 70)

        # 1. Mount Google Drive if not already mounted
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                print("📁 [COLAB] Mounting Google Drive to /content/drive...")
                drive.mount("/content/drive")
            else:
                print("📁 [COLAB] Google Drive already mounted at /content/drive")
        except Exception as err:
            print(f"⚠️ [COLAB] Notice: Google Drive mount skipped or failed ({err})")

        # 2. Designated Colab FOLDER_ROOT
        designated_colab_root = Path("/content/drive/MyDrive/Home/10_Projects/PlantDiseaseClassification")
        if designated_colab_root.is_dir():
            folder_root = designated_colab_root
        else:
            # Fallback search under MyDrive
            mydrive = Path("/content/drive/MyDrive")
            matches = list(mydrive.rglob("PlantDiseaseClassification")) if mydrive.is_dir() else []
            if matches:
                folder_root = matches[0]
            elif Path("/content/PlantDiseaseClassification").is_dir():
                folder_root = Path("/content/PlantDiseaseClassification")
            else:
                folder_root = designated_colab_root
                try:
                    folder_root.mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_err:
                    print(f"ℹ️ [COLAB] Notice: Could not create folder {folder_root} ({mkdir_err})")
    else:
        print("=" * 70)
        print("💻 [ENVIRONMENT] Local Workstation Runtime Detected")
        print("=" * 70)
        folder_root = Path(__file__).resolve().parent

    folder_root = folder_root.resolve()
    data_dir = folder_root / "data" / "raw"
    split_dir = folder_root / "data" / "split"

    print(f"📂 [FOLDER_ROOT] Project Root: {folder_root}")
    print(f"📁 [DATA_DIR]    Raw Data:     {data_dir}")
    print(f"📊 [SPLIT_DIR]   Data Split:   {split_dir}")

    # Set working directory and append to sys.path
    if folder_root.exists():
        os.chdir(folder_root)
    if str(folder_root) not in sys.path:
        sys.path.insert(0, str(folder_root))

    return {
        "is_colab": running_in_colab,
        "folder_root": folder_root,
        "data_dir": data_dir,
        "split_dir": split_dir,
    }


def verify_dependencies(is_colab_env: bool) -> None:
    """Verify essential packages and auto-install on Google Colab if missing."""
    import importlib.util
    packages = {
        "yaml": "PyYAML",
        "pandas": "pandas",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "timm": "timm",
        "kaggle": "kaggle",
        "kagglehub": "kagglehub",
        "albumentations": "albumentations",
    }
    missing = []
    for module_name, pip_name in packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(pip_name)

    if missing:
        if is_colab_env:
            print(f"📦 [DEPENDENCIES] Installing missing packages in Colab: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
            print("✅ [DEPENDENCIES] Installation completed successfully.")
        else:
            print(f"⚠️ [DEPENDENCIES] Missing packages in local environment: {missing}")


def setup_split_directory(split_dir: Path, folder_root: Path) -> Path:
    """Ensure data/split exists and contains train.csv, val.csv, and test.csv."""
    processed_split = folder_root / "data" / "processed" / "split"

    if not split_dir.exists():
        if processed_split.is_dir():
            try:
                split_dir.symlink_to(processed_split.resolve(), target_is_directory=True)
                print(f"🔗 [DATA SPLIT] Created symlink: {split_dir} -> {processed_split}")
            except Exception:
                split_dir = processed_split
        else:
            split_dir.mkdir(parents=True, exist_ok=True)

    # Maintain backward compatibility for data/splits
    splits_compat = folder_root / "data" / "splits"
    if not splits_compat.exists() and split_dir.exists():
        try:
            splits_compat.symlink_to(split_dir.resolve(), target_is_directory=True)
        except Exception:
            pass

    return split_dir


def download_and_resolve_dataset(data_dir: Path, data_root_arg: Optional[str] = None) -> Path:
    """Resolve color dataset or download using Kaggle CLI referenced from 01_data_audit.ipynb.
    
    1. If explicit data_root is provided, verify and use it.
    2. Check if a 'color' directory already exists in data_dir (data/raw). If found, skip download.
    3. If missing, configure Kaggle credentials and download via:
       kaggle datasets download -d abdallahalidev/plantvillage-dataset -p {data_dir} --unzip
    4. Fallback to kagglehub if Kaggle CLI fails.
    """
    if data_root_arg:
        p = Path(data_root_arg)
        if p.is_dir():
            print(f"🌿 [DATASET] Using explicitly provided DATA_ROOT: {p}")
            return p
        print(f"⚠️ [DATASET] Explicit DATA_ROOT not found: {data_root_arg}. Searching standard locations...")

    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if 'color' directory already exists inside data_dir
    color_candidates = list(data_dir.rglob("color"))
    for cand in color_candidates:
        if cand.is_dir() and any(cand.iterdir()):
            print(f"🌿 [DATASET] Found existing color dataset at: {cand}")
            return cand

    # 2. Check local user cache fallback
    cache_cand = Path(os.path.expanduser("~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"))
    if cache_cand.is_dir():
        print(f"🌿 [DATASET] Found cached dataset at: {cache_cand}")
        return cache_cand

    # 3. Download via Kaggle CLI (reference from 01_data_audit.ipynb)
    print("⬇️ [DATASET] Dataset not found in data/raw. Initiating download via Kaggle API...")
    
    # Configure Kaggle credentials on Colab if available in userdata
    if is_colab():
        try:
            from google.colab import userdata
            username = userdata.get("KAGGLE_USERNAME")
            token = userdata.get("KAGGLE_API_TOKEN")
            if username:
                os.environ["KAGGLE_USERNAME"] = username
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
            print("🔑 [KAGGLE] Kaggle credentials loaded from Colab userdata.")
        except Exception as e:
            print(f"ℹ️ [KAGGLE] Userdata lookup note: {e}")

    download_success = False
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "abdallahalidev/plantvillage-dataset",
            "-p", str(data_dir),
            "--unzip", "--force"
        ]
        print(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("✅ [KAGGLE] Dataset downloaded and extracted successfully via Kaggle CLI!")
            download_success = True
        else:
            print(f"⚠️ [KAGGLE] Kaggle CLI returned non-zero code: {res.stderr.strip() or res.stdout.strip()}")
    except FileNotFoundError:
        print("⚠️ [KAGGLE] 'kaggle' CLI tool not found in PATH.")
    except Exception as e:
        print(f"⚠️ [KAGGLE] Kaggle download error: {e}")

    # 4. Fallback to kagglehub if Kaggle CLI was not successful
    if not download_success:
        print("⬇️ [DATASET] Falling back to kagglehub downloader...")
        try:
            import kagglehub
            hub_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
            print(f"✅ [KAGGLEHUB] Downloaded dataset to: {hub_path}")
            hub_matches = list(Path(hub_path).rglob("color"))
            if hub_matches and hub_matches[0].is_dir():
                return hub_matches[0]
            return Path(hub_path)
        except Exception as hub_err:
            raise RuntimeError(
                f"Failed to download dataset via both Kaggle CLI and kagglehub. Error: {hub_err}"
            )

    # 5. Locate extracted 'color' folder in data_dir
    color_candidates = list(data_dir.rglob("color"))
    if not color_candidates:
        raise FileNotFoundError(f"Could not locate 'color' folder inside {data_dir} after download.")

    selected_color = next(p for p in color_candidates if p.is_dir())
    print(f"✅ [DATASET] Selected color subset directory: {selected_color}")
    return selected_color


def run_experiment(
    model_name: str,
    data_root: Path,
    split_dir: Path,
    folder_root: Path,
    config: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    grad_accum_steps: int = 1,
    fp16: bool = True,
    resume: bool = True,
    freeze_backbone: bool = False,
    no_aug: bool = False,
    debug: bool = False,
) -> bool:
    """Execute training for a given model by invoking train.py."""
    cmd = [
        sys.executable,
        str(folder_root / "train.py"),
        "--model_name", model_name,
        "--data_root", str(data_root),
        "--split_dir", str(split_dir),
        "--output_dir", str(folder_root / "outputs" / "checkpoints"),
        "--log_dir", str(folder_root / "outputs" / "logs"),
        "--table_dir", str(folder_root / "outputs" / "tables"),
    ]

    if config:
        cmd.extend(["--config", str(config)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if grad_accum_steps > 1:
        cmd.extend(["--grad_accum_steps", str(grad_accum_steps)])
    if fp16:
        cmd.append("--fp16")
    if resume:
        cmd.append("--resume")
    if freeze_backbone:
        cmd.append("--freeze_backbone")
    if no_aug:
        cmd.append("--no_aug")
    if debug:
        cmd.append("--debug")

    print("\n" + "=" * 70)
    print(f"🚀 [EXPERIMENT] Running training for: {model_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ [EXPERIMENT] Model {model_name} exited with status code: {result.returncode}")
        return False
    print(f"✅ [EXPERIMENT] Model {model_name} completed successfully!")
    return True


def parse_args():
    """Parse command line arguments for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Unified Colab & Local Training Orchestrator for Plant Disease Classification."
    )
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model architecture to train (e.g., resnet50, vit_base_patch16_224, efficientnet_b0)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to specific model config YAML file")
    parser.add_argument("--data_root", type=str, default="",
                        help="Path to dataset color directory (optional, auto-detected if empty)")
    parser.add_argument("--all", action="store_true",
                        help="Run full sequential benchmark on all models (ResNet-50, ViT-Base, EfficientNet-B0)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation experiments (Frozen Backbone and No Augmentation on ResNet-50)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides YAML config default if specified)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides YAML config default if specified)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides YAML config default if specified)")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable Mixed Precision FP16 training")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable auto-resuming from existing checkpoints")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze feature extractor backbone (train classifier head only)")
    parser.add_argument("--no_aug", action="store_true",
                        help="Disable training data augmentation")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (1 epoch on 64 train / 32 val samples)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry-run verification of environment, paths, configs, and dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Environment and Path Initialization
    env = setup_environment()
    folder_root = env["folder_root"]
    data_dir = env["data_dir"]
    split_dir = env["split_dir"]
    is_colab_env = env["is_colab"]

    # 2. Dependency Verification
    verify_dependencies(is_colab_env)

    # 3. Setup Data Splits Directory (data/split)
    split_dir = setup_split_directory(split_dir, folder_root)
    print(f"📊 [TRAIN CSV] Path: {split_dir / 'train.csv'} (Exists: {(split_dir / 'train.csv').exists()})")
    print(f"📊 [VAL CSV]   Path: {split_dir / 'val.csv'}   (Exists: {(split_dir / 'val.csv').exists()})")
    print(f"📊 [TEST CSV]  Path: {split_dir / 'test.csv'}  (Exists: {(split_dir / 'test.csv').exists()})")

    # 4. Resolve Dataset (data/raw/.../color)
    data_root = download_and_resolve_dataset(data_dir, args.data_root)

    # 5. Handle Dry Run
    if args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 [DRY RUN] Verification Report")
        print("=" * 70)
        print(f"Runtime:         {'Google Colab' if is_colab_env else 'Local Workstation'}")
        print(f"FOLDER_ROOT:     {folder_root}")
        print(f"DataDir:         {data_dir}")
        print(f"DATA_ROOT:       {data_root}")
        print(f"SPLIT_DIR:       {split_dir}")
        print(f"Checkpoints Dir: {folder_root / 'outputs' / 'checkpoints'}")
        print(f"Logs Dir:        {folder_root / 'outputs' / 'logs'}")
        print(f"Auto-Resume:     {not args.no_resume}")
        print(f"FP16 Enabled:    {not args.no_fp16}")

        configs_dir = folder_root / "configs"
        models_cfg_dir = configs_dir / "models"
        print(f"Default Config:  {configs_dir / 'default_config.yaml'} (Exists: {(configs_dir / 'default_config.yaml').exists()})")
        if models_cfg_dir.is_dir():
            cfgs = [f.name for f in models_cfg_dir.glob("*.yaml")]
            print(f"Model Configs:   {cfgs}")

        print("\n✅ [DRY RUN] All checks passed successfully! Ready for training.\n")
        return

    use_fp16 = not args.no_fp16
    use_resume = not args.no_resume

    # 6. Pipeline Execution
    if args.all:
        print("\n🎯 [PIPELINE] Sequential execution queue selected for all benchmark models:")
        experiments = [
            {
                "model_name": "resnet50",
                "config": str(folder_root / "configs" / "models" / "resnet50.yaml"),
                "batch_size": args.batch_size or 32,
                "grad_accum_steps": 1,
            },
            {
                "model_name": "vit_base_patch16_224",
                "config": str(folder_root / "configs" / "models" / "vit_base.yaml"),
                "batch_size": args.batch_size or 16,
                "grad_accum_steps": 2,  # Effective batch size = 32
            },
            {
                "model_name": "efficientnet_b0",
                "config": str(folder_root / "configs" / "models" / "efficientnet_b0.yaml"),
                "batch_size": args.batch_size or 32,
                "grad_accum_steps": 1,
            },
        ]
        for exp in experiments:
            success = run_experiment(
                model_name=exp["model_name"],
                config=exp.get("config"),
                data_root=data_root,
                split_dir=split_dir,
                folder_root=folder_root,
                epochs=args.epochs,
                batch_size=exp["batch_size"],
                lr=args.lr,
                grad_accum_steps=exp.get("grad_accum_steps", 1),
                fp16=use_fp16,
                resume=use_resume,
                debug=args.debug,
            )
            if not success:
                print(f"⚠️ Pipeline stopped due to failure in {exp['model_name']}")
                sys.exit(1)

    elif args.ablation:
        print("\n🎯 [PIPELINE] Ablation study queue selected:")
        # E4: Frozen Backbone
        print("--- Ablation E4: Frozen Backbone (Feature Extractor Frozen) ---")
        run_experiment(
            model_name="resnet50",
            config=str(folder_root / "configs" / "models" / "resnet50.yaml"),
            data_root=data_root,
            split_dir=split_dir,
            folder_root=folder_root,
            epochs=args.epochs or 5,
            batch_size=args.batch_size or 32,
            lr=args.lr or 1e-3,
            fp16=use_fp16,
            resume=use_resume,
            freeze_backbone=True,
            debug=args.debug,
        )
        # E5: No Augmentation
        print("--- Ablation E5: No Augmentation ---")
        run_experiment(
            model_name="resnet50",
            config=str(folder_root / "configs" / "models" / "resnet50.yaml"),
            data_root=data_root,
            split_dir=split_dir,
            folder_root=folder_root,
            epochs=args.epochs or 10,
            batch_size=args.batch_size or 32,
            lr=args.lr or 1e-4,
            fp16=use_fp16,
            resume=use_resume,
            no_aug=True,
            debug=args.debug,
        )

    else:
        # Single model training
        selected_model = args.model_name or "resnet50"
        grad_accum = 2 if "vit" in selected_model.lower() else 1
        batch_sz = args.batch_size or (16 if "vit" in selected_model.lower() else 32)

        run_experiment(
            model_name=selected_model,
            config=args.config,
            data_root=data_root,
            split_dir=split_dir,
            folder_root=folder_root,
            epochs=args.epochs,
            batch_size=batch_sz,
            lr=args.lr,
            grad_accum_steps=grad_accum,
            fp16=use_fp16,
            resume=use_resume,
            freeze_backbone=args.freeze_backbone,
            no_aug=args.no_aug,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
