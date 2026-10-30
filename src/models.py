"""Model construction and fine-tuning configurations using timm and torchvision."""

from typing import Tuple
import torch
import torch.nn as nn
import timm


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Build and initialize model with custom classification head.
    
    Supported backbones:
    - resnet18, resnet50
    - vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224
    - efficientnet_b0
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep only the classifier head trainable
            if not any(h in name.lower() for h in ["head", "fc", "classifier"]):
                param.requires_grad = False
                
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
