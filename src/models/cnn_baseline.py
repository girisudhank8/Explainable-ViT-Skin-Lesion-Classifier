import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class BaselineCNN(nn.Module):
    """
    Standard Convolutional Neural Network (ResNet/ConvNeXt style baseline)
    for benchmarking against the Explainable Vision Transformer.
    Includes feature map hooks for Grad-CAM explainability comparison.
    """
    DEFAULT_CLASSES = [
        "MEL (Melanoma)",
        "NV (Melanocytic Nevus)",
        "BCC (Basal Cell Carcinoma)",
        "AKIEC (Actinic Keratosis)",
        "BKL (Benign Keratosis)",
        "DF (Dermatofibroma)",
        "VASC (Vascular Lesion)"
    ]

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        base_channels: int = 32,
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or self.DEFAULT_CLASSES[:num_classes]
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stages
        self.stage1 = ConvBlock(base_channels, base_channels * 2, stride=1)
        self.stage2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        self.stage4 = ConvBlock(base_channels * 8, base_channels * 16, stride=2) # Target layer for Grad-CAM
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(base_channels * 16, num_classes)
        
        # Buffers for Grad-CAM
        self.target_activations: Optional[torch.Tensor] = None
        self.target_gradients: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Save activations for Grad-CAM if hooked
        if x.requires_grad:
            x.register_hook(self._save_gradient)
        self.target_activations = x
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.head(x)
        return logits

    def _save_gradient(self, grad: torch.Tensor):
        self.target_gradients = grad

    def get_target_layer_feature_maps(self) -> Optional[torch.Tensor]:
        return self.target_activations
