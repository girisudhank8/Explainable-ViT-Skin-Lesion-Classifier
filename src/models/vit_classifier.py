import math
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class PatchEmbedding(nn.Module):
    """
    Splits image of size (C, H, W) into non-overlapping patches of size (P, P)
    and projects each patch to an embedding vector of dimension D.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with explicit retention of attention weight matrices
    for Attention Rollout and Explainable AI analysis.
    """
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        if store_attn:
            self.last_attn_weights = attn.detach()
            
        attn_out = self.attn_drop(attn)
        x = (attn_out @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Block with Pre-LayerNorm and residual connections.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), store_attn=store_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ExplainableViT(nn.Module):
    """
    Explainable Vision Transformer for Skin Lesion Classification.
    Provides direct access to layer-wise multi-head self-attention maps for Attention Rollout.
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
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.class_names = class_names or self.DEFAULT_CLASSES[:num_classes]

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x, store_attn=store_attn)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        feat = self.forward_features(x, store_attn=store_attn)
        cls_out = feat[:, 0]
        logits = self.head(cls_out)
        return logits

    def get_all_attention_matrices(self) -> List[torch.Tensor]:
        attentions = []
        for block in self.blocks:
            if block.attn.last_attn_weights is not None:
                attentions.append(block.attn.last_attn_weights)
        return attentions

    @classmethod
    def create_small(cls, num_classes: int = 7, img_size: int = 224) -> "ExplainableViT":
        return cls(img_size=img_size, patch_size=16, embed_dim=256, depth=8, num_heads=4, num_classes=num_classes)

    @classmethod
    def create_base(cls, num_classes: int = 7, img_size: int = 224) -> "ExplainableViT":
        return cls(img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=num_classes)


class PretrainedTimmViT(nn.Module):
    """
    Pretrained Vision Transformer (e.g. vit_small_patch16_224, vit_base_patch16_224)
    from `timm` with forward hooks to extract attention maps for Attention Rollout.
    """
    def __init__(self, model_name: str = "vit_small_patch16_224", num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for PretrainedTimmViT. Install via 'pip install timm'.")

        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.captured_attentions: List[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self):
        self.captured_attentions = []
        for block in self.model.blocks:
            attn_module = block.attn
            attn_module.register_forward_hook(self._get_attn_hook())

    def _get_attn_hook(self):
        def hook(module, input, output):
            # timm computes attention inside forward: Q @ K.T * scale -> softmax
            # We can capture the computed attention weights if available or compute from Q, K
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]
            scale = module.scale if hasattr(module, 'scale') else 1.0 / math.sqrt(q.shape[-1])
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self.captured_attentions.append(attn.detach())
        return hook

    def forward(self, x: torch.Tensor, store_attn: bool = True) -> torch.Tensor:
        self.captured_attentions.clear()
        logits = self.model(x)
        return logits

    def get_all_attention_matrices(self) -> List[torch.Tensor]:
        return self.captured_attentions
