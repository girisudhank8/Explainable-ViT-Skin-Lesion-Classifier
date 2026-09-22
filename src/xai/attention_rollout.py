import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class AttentionRollout:
    """
    Implements Attention Rollout for Vision Transformers (Abnar & Zuidema, 2020).
    Recursively tracks and multiplies attention weights across all transformer blocks
    accounting for residual connections to map information flow from image patches to the [CLS] token.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
        add_residual: bool = True,
        residual_weight: float = 0.5
    ):
        """
        Args:
            model: ExplainableViT instance
            head_fusion: How to fuse multi-head attention ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attention values to zero out in each layer (noise reduction)
            add_residual: Whether to add identity matrix I for residual skip connections
            residual_weight: Weight of residual identity matrix (standard is 0.5)
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.add_residual = add_residual
        self.residual_weight = residual_weight

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_layer: Optional[int] = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Calculates the Attention Rollout mask for input_tensor.

        Args:
            input_tensor: [B, C, H, W] image batch (typically B=1 for explainability)
            target_layer: Layer index up to which to compute rollout (None = all layers)

        Returns:
            mask: [B, H, W] numpy array normalized to [0, 1]
            logits: model prediction logits [B, num_classes]
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor, store_attn=True)
            attn_matrices = self.model.get_all_attention_matrices()

        if not attn_matrices:
            raise RuntimeError("No attention matrices captured. Ensure model stores attention weights during forward pass.")

        layers_to_rollout = attn_matrices[:target_layer] if target_layer is not None else attn_matrices
        mask = compute_attention_rollout(
            attn_matrices=layers_to_rollout,
            img_size=(input_tensor.shape[2], input_tensor.shape[3]),
            head_fusion=self.head_fusion,
            discard_ratio=self.discard_ratio,
            add_residual=self.add_residual,
            residual_weight=self.residual_weight
        )
        return mask, logits

    def get_layerwise_rollouts(self, input_tensor: torch.Tensor) -> List[np.ndarray]:
        """
        Computes progressive rollout heatmaps at each transformer depth layer.
        Useful for inspecting how attention evolves from local textures to global pathology.
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor, store_attn=True)
            attn_matrices = self.model.get_all_attention_matrices()

        layer_masks = []
        for l_idx in range(1, len(attn_matrices) + 1):
            mask = compute_attention_rollout(
                attn_matrices=attn_matrices[:l_idx],
                img_size=(input_tensor.shape[2], input_tensor.shape[3]),
                head_fusion=self.head_fusion,
                discard_ratio=self.discard_ratio,
                add_residual=self.add_residual,
                residual_weight=self.residual_weight
            )
            layer_masks.append(mask[0])
        return layer_masks


def compute_attention_rollout(
    attn_matrices: List[torch.Tensor],
    img_size: Tuple[int, int] = (224, 224),
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
    add_residual: bool = True,
    residual_weight: float = 0.5
) -> np.ndarray:
    """
    Pure algorithmic computation of Attention Rollout from a list of layer attention tensors.

    Args:
        attn_matrices: List of tensors, each [B, heads, N+1, N+1]
        img_size: (Height, Width) of the original image
        head_fusion: 'mean', 'max', 'min'
        discard_ratio: bottom quantile of attention weights to suppress
        add_residual: whether to combine with Identity matrix
        residual_weight: weight factor for identity matrix
    """
    B = attn_matrices[0].shape[0]
    num_tokens = attn_matrices[0].shape[-1]
    
    # Initialize rollout with identity matrix for each item in batch: [B, N+1, N+1]
    device = attn_matrices[0].device
    rollout = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(B, 1, 1)

    for attn in attn_matrices:
        # Fuse heads: [B, heads, N+1, N+1] -> [B, N+1, N+1]
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]
        elif head_fusion == "min":
            attn_fused = attn.min(dim=1)[0]
        else:
            attn_fused = attn.mean(dim=1)

        # Discard lowest attention values (thresholding) to reduce noise
        if discard_ratio > 0.0 and discard_ratio < 1.0:
            flat = attn_fused.view(B, -1)
            # Find threshold per sample
            k = int(flat.size(1) * (1.0 - discard_ratio))
            if k > 0:
                threshold = torch.kthvalue(flat, k, dim=1, keepdim=True).values.unsqueeze(-1)
                attn_fused = torch.where(attn_fused < threshold, torch.zeros_like(attn_fused), attn_fused)

        # Account for residual connection: A_hat = (1 - w)*A + w*I
        if add_residual:
            identity = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(B, 1, 1)
            attn_fused = (1.0 - residual_weight) * attn_fused + residual_weight * identity

        # Normalize rows to sum to 1
        row_sums = attn_fused.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)
        attn_fused = attn_fused / row_sums

        # Rollout multiplication: R_{l} = A_{l} * R_{l-1}
        rollout = torch.matmul(attn_fused, rollout)

    # Extract attention from [CLS] token (index 0) to all patch tokens (indices 1:)
    # cls_attention: [B, num_patches]
    cls_attention = rollout[:, 0, 1:]
    num_patches = cls_attention.shape[1]
    grid_size = int(math.isqrt(num_patches))

    # Reshape to 2D grid: [B, 1, grid_size, grid_size]
    cls_attention = cls_attention.reshape(B, 1, grid_size, grid_size)

    # Upsample to original image resolution via bicubic interpolation
    heatmap = F.interpolate(
        cls_attention,
        size=img_size,
        mode="bicubic",
        align_corners=False
    ) # [B, 1, H, W]

    # Convert to numpy and normalize each sample in batch to [0, 1]
    heatmaps_np = heatmap.squeeze(1).cpu().numpy()
    normalized_heatmaps = np.zeros_like(heatmaps_np)

    for i in range(B):
        h = heatmaps_np[i]
        h_min, h_max = h.min(), h.max()
        if h_max - h_min > 1e-8:
            normalized_heatmaps[i] = (h - h_min) / (h_max - h_min)
        else:
            normalized_heatmaps[i] = h

    return normalized_heatmaps
