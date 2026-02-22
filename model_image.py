"""
Lightweight CNN for ISL letter/number recognition from 128x128 images.
Uses depthwise-separable convolutions for CPU efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_image import IMG_DROPOUT, IMG_SIZE


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, 1, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class SignImageCNN(nn.Module):
    """
    Compact CNN for hand gesture image classification.

    Architecture:
      - 5 conv blocks with increasing channels
      - BatchNorm + ReLU after each conv
      - Global average pooling
      - FC head with dropout

    Input: (batch, 3, 128, 128)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Conv backbone: 3 -> 32 -> 64 -> 128 -> 256
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64x64 -> 32x32
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),

            # Block 3: 32x32 -> 16x16
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),

            # Block 4: 16x16 -> 8x8
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2),

            # Block 5: 8x8 -> 4x4
            DepthwiseSeparableConv(256, 256, stride=2),
        )

        # Global average pooling -> 256-dim vector
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(IMG_DROPOUT)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(IMG_DROPOUT),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
