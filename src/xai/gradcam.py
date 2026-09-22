import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).
    Works with any model:
    - If model has .target_activations / .target_gradients (BaselineCNN) → uses those
    - Otherwise → auto-registers hooks on the last Conv2d layer found in the model
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hook_handles = []
        self._auto_activations = None
        self._auto_gradients = None

    def _find_last_conv(self) -> Optional[nn.Module]:
        """Find the last Conv2d layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self, target_layer: nn.Module):
        """Register forward and backward hooks on a target layer."""
        def forward_hook(module, input, output):
            self._auto_activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._auto_gradients = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, List[int]]] = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Computes Grad-CAM for a given target class or top predicted class per sample.

        Args:
            input_tensor: [B, C, H, W]
            target_class: Integer class index or list of class indices.

        Returns:
            cam: [B, H, W] numpy array normalized to [0, 1]
            logits: model prediction logits (detached)
        """
        self.model.eval()
        self._auto_activations = None
        self._auto_gradients = None

        # Determine whether we need to auto-wire hooks
        use_model_hooks = (
            hasattr(self.model, "target_activations") and
            hasattr(self.model, "target_gradients")
        )

        if not use_model_hooks:
            target_layer = self._find_last_conv()
            if target_layer is None:
                raise RuntimeError("GradCAM: no Conv2d layer found in model.")
            self._register_hooks(target_layer)

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        B = logits.shape[0]

        if target_class is None:
            target_classes = logits.argmax(dim=-1).tolist()
        elif isinstance(target_class, int):
            target_classes = [target_class] * B
        else:
            target_classes = target_class

        score = torch.stack([logits[i, target_classes[i]] for i in range(B)]).sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if use_model_hooks:
            activations = self.model.target_activations
            gradients = self.model.target_gradients
        else:
            activations = self._auto_activations
            gradients = self._auto_gradients
            self._remove_hooks()

        if activations is None or gradients is None:
            raise RuntimeError("GradCAM: Activations or gradients not captured.")

        # Handle 1-D global-avg-pooled features (e.g. EfficientNet after pool)
        # shape may be [B, C] instead of [B, C, H, W] after avgpool — skip those
        if activations.dim() == 2:
            # Fall back to uniform heatmap (model pooled away spatial dims)
            cam_np = np.ones((B, input_tensor.shape[2], input_tensor.shape[3]),
                             dtype=np.float32) * 0.5
            return cam_np, logits.detach()

        # Global average pooling on gradients → alpha weights [B, C, 1, 1]
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam_np = cam.squeeze(1).detach().cpu().numpy()
        normalized_cam = np.zeros_like(cam_np)
        for i in range(B):
            c = cam_np[i]
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-8:
                normalized_cam[i] = (c - c_min) / (c_max - c_min)
            else:
                normalized_cam[i] = np.ones_like(c) * 0.5

        return normalized_cam, logits.detach()
