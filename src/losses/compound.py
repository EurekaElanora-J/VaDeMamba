import torch
import torch.nn as nn
import torch.nn.functional as F

class ForegroundSoftDiceLoss(nn.Module):
    """
    Differentiable foreground Soft Dice loss over the three lesion classes:
      NCR = class 1
      ED  = class 2
      ET  = class 3
    Background class 0 is excluded from the Dice term.
    
    L_SoftDice = 1 - (1/3) * sum_{c in {1, 2, 3}} [ (2 * sum_v P_c(v) Y_c(v) + eps) / (sum_v P_c(v)^2 + sum_v Y_c(v)^2 + eps) ]
    with eps = 1e-5.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, probs, target):
        """
        probs: (B, 4, D, H, W) post-softmax probabilities
        target: (B, D, H, W) integer labels in {0, 1, 2, 3}
        Returns: scalar loss tensor
        """
        B, C, D, H, W = probs.shape
        dice_scores = []
        
        # Target one-hot for classes 1, 2, 3
        for c in [1, 2, 3]:
            p_c = probs[:, c] # (B, D, H, W)
            y_c = (target == c).float() # (B, D, H, W)
            
            # Sum over all voxels (and batch, since B=1)
            intersection = torch.sum(p_c * y_c)
            cardinality = torch.sum(p_c ** 2) + torch.sum(y_c ** 2)
            
            dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
            dice_scores.append(dice_c)
            
        mean_foreground_dice = torch.stack(dice_scores).mean()
        loss = 1.0 - mean_foreground_dice
        return loss


class RevisedExperiment0Loss(nn.Module):
    """
    Coupled deep supervision training objective for Experiment 0:
      L_train = (1/3) * sum_{k=1}^3 [ L_CE^(k) + L_SoftDice^(k) ]
    where:
      L_CE^(k) is standard unweighted CrossEntropyLoss(logits_k, target)
      L_SoftDice^(k) is ForegroundSoftDiceLoss(probs_k, target)
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = ForegroundSoftDiceLoss(eps=eps)

    def forward(self, out, target):
        """
        out: dict containing:
          "logits": {"exit1": logits1, "exit2": logits2, "exit3": logits3}
          "probs":  {"exit1": probs1,  "exit2": probs2,  "exit3": probs3}
        target: (B, D, H, W)
        """
        total_loss = 0.0
        losses_dict = {}
        
        for exit_name in ["exit1", "exit2", "exit3"]:
            logits_k = out["logits"][exit_name]
            probs_k = out["probs"][exit_name]
            
            l_ce = self.ce_loss(logits_k, target)
            l_dice = self.dice_loss(probs_k, target)
            
            losses_dict[f"{exit_name}_ce"] = l_ce.item()
            losses_dict[f"{exit_name}_dice"] = l_dice.item()
            
            total_loss = total_loss + (l_ce + l_dice)
            
        coupled_loss = total_loss / 3.0
        return coupled_loss, losses_dict
