"""PyTorch Dataset definition and transform utilities for PlantVillage."""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PlantVillageDataset(Dataset):
    """Custom Dataset for PlantVillage using relative file paths and portable DATA_ROOT."""

    def __init__(
        self,
        df_or_csv: Union[pd.DataFrame, str, Path],
        data_root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_col: str = "class_id",
        path_col: str = "rel_path",
    ):
        """
        Args:
            df_or_csv: Pandas DataFrame or path to split CSV (e.g., train.csv).
            data_root: Root directory of the dataset (the `color` folder).
            transform: PyTorch transform.
            target_col: Column containing integer class targets.
            path_col: Column containing relative path from data_root.
        """
        if isinstance(df_or_csv, (str, Path)):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.copy().reset_index(drop=True)

        self.data_root = Path(data_root)
        self.transform = transform
        self.target_col = target_col
        self.path_col = path_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        full_path = self.data_root / rel_path

        # Open image and ensure RGB mode
        with Image.open(full_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row[self.target_col])
        return image, label


def get_default_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    is_train: bool = False,
    no_aug: bool = False,
) -> T.Compose:
    """Standard transformations for ImageNet pretrained models (ResNet, ViT, EfficientNet).
    
    Training: mild data augmentation (random horizontal flip, slight rotation, color jitter).
    If no_aug is True (Ablation E5), use deterministic resize and crop even during training.
    Validation/Test: deterministic resize + center crop + normalization.
    """
    if is_train and not no_aug:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
