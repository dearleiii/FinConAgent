import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    gamma > 0 reduces relative loss for well-classified examples
    alpha balances class weights
    """
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha   # Tensor of shape [num_classes] or scalar
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # normal CE loss per sample

        pt = torch.exp(-ce_loss)            # probability of correct class
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]  # <-- GPU-safe
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
