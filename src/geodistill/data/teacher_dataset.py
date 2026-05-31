"""Dataset that loads pre-computed teacher outputs."""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from geodistill.data.transforms import GeoTransforms


class TeacherDataset(Dataset):
    """Dataset wrapper for loading pre-computed teacher annotations."""

    def __init__(self, data_root: str, split: str = "train", transform: GeoTransforms = None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        self.image_dir = self.data_root / split / "images"
        self.depth_dir = self.data_root / split / "depths"
        self.normal_dir = self.data_root / split / "normals"
        
        # Check if the directory exists
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        
    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]
        base_name = img_path.stem
        
        sample = {}
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        sample["image"] = image
        
        # Load depth if available
        depth_path = self.depth_dir / f"{base_name}.npy"
        if depth_path.exists():
            depth = np.load(depth_path)
            sample["depth"] = depth
            
        # Load normal if available
        normal_path = self.normal_dir / f"{base_name}.npy"
        if normal_path.exists():
            normal = np.load(normal_path)
            sample["normal"] = normal
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
