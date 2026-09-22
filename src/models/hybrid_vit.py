import math
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_classifier import TransformerBlock


class ConvStem(nn.Module):
    """
    Multi-stage Convolutional Stem (Paper 02: Hybrid CNN-Transformer).
    Extracts fine-grained local dermatological micro-textures (pigment networks, telangiectasia)
    before feeding representations to the Transformer global self-attention blocks.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        # Stage 1: 224x224 -> 112x112
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        # Stage 2: 112x112 -> 56x56
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        # Stage 3: 56x56 -> 14x14
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 224, 224] -> Output: [B, embed_dim, 14, 14]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class HybridViT(nn.Module):
    """
    Hybrid CNN-Vision Transformer for Skin Lesion Diagnosis (Paper 02).
    Combines local CNN receptive field for micro-texture sensitivity with global
    Multi-Head Self-Attention for asymmetric lesion structure and transparent Attention Rollout.
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
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_names = class_names or self.DEFAULT_CLASSES[:num_classes]

        # 1. Convolutional Stem
        self.conv_stem = ConvStem(in_channels=in_channels, embed_dim=embed_dim)
        
        # Grid size after stem: (224/16, 224/16) = (14, 14) -> 196 tokens
        self.grid_size = (img_size // 16, img_size // 16)
        num_tokens = self.grid_size[0] * self.grid_size[1]

        # 2. [CLS] Token & Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Transformer Encoder Blocks with Attention Retention Hooks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.0
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        B = x.shape[0]
        # CNN Feature extraction: [B, D, 14, 14] -> [B, 196, D]
        feat_map = self.conv_stem(x)
        tokens = feat_map.flatten(2).transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer Self-Attention Blocks
        for block in self.blocks:
            tokens = block(tokens, store_attn=store_attn)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)
        return logits

    def get_all_attention_matrices(self) -> List[torch.Tensor]:
        """Returns captured attention matrices across all Transformer blocks."""
        attentions = []
        for block in self.blocks:
            if block.attn.last_attn_weights is not None:
                attentions.append(block.attn.last_attn_weights)
        return attentions
