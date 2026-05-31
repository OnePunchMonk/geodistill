"""Depth evaluation metrics."""

import torch

def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> dict[str, float]:
    """Compute standard depth metrics (AbsRel, RMSE, log10, delta accuracies)."""
    
    if mask is None:
        mask = target > 0
        
    pred = pred[mask]
    target = target[mask]
    
    if pred.numel() == 0:
        return {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}
        
    # Standardize scale using median scaling if needed (for relative depth)
    # pred = pred * (torch.median(target) / torch.median(pred))
    
    thresh = torch.maximum((target / pred), (pred / target))
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()

    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(target)) ** 2)).item()
    
    abs_rel = torch.mean(torch.abs(pred - target) / target).item()
    sq_rel = torch.mean((pred - target) ** 2 / target).item()

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target))).item()
    
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "log10": log10,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3
    }
