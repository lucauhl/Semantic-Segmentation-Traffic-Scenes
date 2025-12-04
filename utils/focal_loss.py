import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255, alpha=None, reduction="mean"):
        """
        gamma: Focal exponent, higher gamma decreases the loss value for easily classified examples more strongly
        alpha: Class weight given to each class (optional), has to be a tensor of size C (num_classes)
        ignore_index: LabelId/LabelTrainId to be ignored in loss calculation
        reduction: How the losses get returned (mean, sum or no reduction)
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None and isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            alpha = None

    def forward(self, logits, targets):
        """
        logits: (N, C, H, W)
        targets: (N, H, W)
        """

        # Compute log probabilities along C-axis (N, C, H, W)
        log_probs = F.log_softmax(logits, dim=1)   
        
        # Build valid pixel mask using ignore_index -> (N, H, W)
        valid_mask = (targets != self.ignore_index)   
        safe_targets = targets.clone()
        # Set the pixels with ignore_index to 0, so the next gather operation works
        # This is safe, because we will mask those positions before calculating the loss
        safe_targets[~valid_mask] = 0 

        # Gather log_p_t for each pixel
        # log_probs shape -> (N, C, H, W)
        # safe_targets.unsqueeze(1) -> (N, 1, H, W)
        log_p_t = log_probs.gather(1, safe_targets.unsqueeze(1))
        # squeeze(1) -> (N, H, W) so we can use the valid_mask with shape (N, H, W)
        log_p_t = log_p_t.squeeze(1)
        
        # Keep only valid pixels, ignoring ignore_index positions -> 1D tensor
        log_p_t = log_p_t[valid_mask]

        # p_t = exp(log_p_t)
        p_t = log_p_t.exp()

        # CE = -log(p_t)
        ce_loss = -log_p_t
        
        # Focal term = (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Alpha/class weighting (optional)
        if self.alpha is not None:
            # Gather alpha/weight for each valid pixel
            alpha_t = self.alpha.gather(0, safe_targets[valid_mask])
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss