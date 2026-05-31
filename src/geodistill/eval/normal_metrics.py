"""Surface normal evaluation metrics."""

import torch
import torch.nn.functional as F


def compute_normal_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> dict[str, float]:
    """Compute surface normal metrics (mean angular error, angular accuracies)."""
    
    # pred and target are [B, 3, H, W]
    if mask is None:
        # Assuming [B, 1, H, W] mask
        mask = torch.ones_like(target[:, 0:1, :, :], dtype=torch.bool)
        
    pred = pred.permute(0, 2, 3, 1)[mask.squeeze(1)]
    target = target.permute(0, 2, 3, 1)[mask.squeeze(1)]
    
    if pred.numel() == 0:
        return {"mean_angle": 0.0, "median_angle": 0.0, "a11": 0.0, "a22": 0.0, "a30": 0.0}
        
    # Normalize
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    
    # Dot product
    dot = torch.sum(pred * target, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    angles = torch.acos(dot) * 180.0 / torch.pi
    
    mean_angle = angles.mean().item()
    median_angle = angles.median().item()
    
    a11 = (angles < 11.25).float().mean().item()
    a22 = (angles < 22.5).float().mean().item()
    a30 = (angles < 30.0).float().mean().item()
    
    return {
        "mean_angle": mean_angle,
        "median_angle": median_angle,
        "a11": a11,
        "a22": a22,
        "a30": a30
    }
