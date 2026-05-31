"""HM3D Dataset."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from geodistill.data.transforms import GeoTransforms


class HM3DDataset(Dataset):
    """HM3D dataset for evaluation/training."""

    def __init__(self, data_root: str, split: str = "train", transform: GeoTransforms = None):
        self.data_root = Path(data_root) / "hm3d"
        self.split = split
        self.transform = transform
        
        self.image_dir = self.data_root / split / "rgb"
        
        if not self.image_dir.exists():
            print(f"Warning: HM3D directory not found: {self.image_dir}. Mocking dataset.")
            self.image_files = []
        else:
            self.image_files = sorted(list(self.image_dir.glob("*.png")))
            
    def __len__(self) -> int:
        return max(len(self.image_files), 10)

    def __getitem__(self, idx: int) -> dict:
        if not self.image_files:
            image = Image.new("RGB", (512, 512), color=(128, 128, 128))
            sample = {"image": image}
        else:
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert("RGB")
            sample = {"image": image}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
