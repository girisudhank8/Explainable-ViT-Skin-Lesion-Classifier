import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to handle severe class imbalance in dermatological datasets (e.g. ISIC).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C], targets: [B]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples:
    E_n = (1 - beta^n) / (1 - beta)
    Weights = (1 - beta) / (1 - beta^n)
    """
    def __init__(self, samples_per_cls: List[int], beta: float = 0.9999, loss_type: str = "focal", gamma: float = 2.0):
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_cls, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_cls)
        self.weights = weights
        self.loss_type = loss_type
        
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=self.weights, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(inputs, targets)
