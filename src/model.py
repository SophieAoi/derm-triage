"""Pretrained classifier and class-weighted loss.

Example:
    model = build_model(num_classes=len(train.classes))
    loss_fn = weighted_loss(train.class_counts())
"""

import torch
from torch import nn
from torchvision import models

ARCHITECTURES = {
    # name: (constructor, ImageNet weights)
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(num_classes, arch="resnet50", pretrained=True, dropout=0.0):
    """ImageNet-pretrained ResNet with a new final layer for num_classes.

    dropout adds a dropout layer before the final layer. It is off by default;
    it is there so Monte Carlo dropout can be tried later without changing the code.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, not {arch!r}")
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    in_features = model.fc.in_features
    head = nn.Linear(in_features, num_classes)
    model.fc = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
    return model


def class_weights(class_counts):
    """Inverse-frequency weights: total / (num_classes * count).

    A perfectly balanced dataset gets weight 1 for every class; rarer classes get
    more. This is the same formula as scikit-learn's class_weight="balanced".
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError(f"Every class needs at least one training image, got counts {counts.tolist()}")
    return counts.sum() / (len(counts) * counts)


def weighted_loss(class_counts, device="cpu"):
    """Cross-entropy loss weighted by inverse class frequency in the training split."""
    return nn.CrossEntropyLoss(weight=class_weights(class_counts).to(device))
