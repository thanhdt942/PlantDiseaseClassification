"""Utility functions for reproducibility, hashing, metrics, and logging.
Plant Disease Classification Using Vision Transformers.
"""

import os
import random
import hashlib
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imagehash


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_md5(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file in binary chunks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file in binary chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(img_or_path: Union[str, Path, Image.Image], hash_size: int = 8) -> str:
    """Compute Perceptual Hash (pHash) using imagehash."""
    if isinstance(img_or_path, (str, Path)):
        with Image.open(img_or_path) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    return str(imagehash.phash(img_or_path, hash_size=hash_size))


def get_device() -> torch.device:
    """Detect and return appropriate torch device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
