"""Transfer evaluation utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from geodistill.eval.depth_metrics import compute_depth_metrics
from geodistill.eval.normal_metrics import compute_normal_metrics

def evaluate_transfer(
    model: nn.Module, 
    decoder: nn.Module, 
    dataloader: DataLoader, 
    device: str
) -> Dict[str, float]:
    """Evaluate zero-shot transfer of the distilled model on a dataset."""
    
    model.eval()
    decoder.eval()
    
    all_depth_metrics = []
    all_normal_metrics = []
    
    pbar = tqdm(dataloader, desc="Evaluating Transfer")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            
            # Forward pass
            outputs = model(images)
            preds = decoder(outputs["features"], (images.shape[2], images.shape[3]))
            
            if "depth" in batch and "depth" in preds:
                targets = batch["depth"].to(device)
                metrics = compute_depth_metrics(preds["depth"], targets)
                all_depth_metrics.append(metrics)
                
            if "normal" in batch and "normal" in preds:
                targets = batch["normal"].to(device)
                metrics = compute_normal_metrics(preds["normal"], targets)
                all_normal_metrics.append(metrics)
                
    results = {}
    
    if all_depth_metrics:
        for k in all_depth_metrics[0].keys():
            results[f"depth_{k}"] = sum(m[k] for m in all_depth_metrics) / len(all_depth_metrics)
            
    if all_normal_metrics:
        for k in all_normal_metrics[0].keys():
            results[f"normal_{k}"] = sum(m[k] for m in all_normal_metrics) / len(all_normal_metrics)
            
    return results
