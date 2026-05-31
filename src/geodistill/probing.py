"""Geometric Probing Toolkit for evaluating implicit 3D knowledge."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable


class GeometricProbe(nn.Module):
    """Linear probe to extract specific geometric properties from frozen features."""

    def __init__(self, in_features: int, out_channels: int, upsample_factor: int = 16):
        super().__init__()
        self.upsample_factor = upsample_factor
        
        # Simple projection and upsampling
        self.proj = nn.Conv2d(in_features, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, C, H/16, W/16]
        """
        x = self.proj(features)
        x = self.upsample(x)
        return x


def run_probing_analysis(
    student_model: nn.Module, 
    probe_model: GeometricProbe,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 5,
    target_key: str = "depth"
):
    """Train the probe on top of frozen student features."""
    
    student_model.eval()
    probe_model.to(device)
    
    for epoch in range(epochs):
        probe_model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Probing Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)
            
            with torch.no_grad():
                # Extract features from the last layer
                outputs = student_model(images)
                # Take the last feature map
                features = outputs["features"][-1]
                
            optimizer.zero_grad()
            preds = probe_model(features)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
    return probe_model
