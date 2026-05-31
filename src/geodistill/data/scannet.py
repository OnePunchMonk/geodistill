"""ScanNet Dataset."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from geodistill.data.transforms import GeoTransforms


class ScanNetDataset(Dataset):
    """ScanNet dataset for evaluation/training."""

    def __init__(self, data_root: str, split: str = "train", transform: GeoTransforms = None):
        self.data_root = Path(data_root) / "scannet"
        self.split = split
        self.transform = transform
        
        self.image_dir = self.data_root / split / "color"
        
        if not self.image_dir.exists():
            print(f"Warning: ScanNet directory not found: {self.image_dir}. Mocking dataset.")
            self.image_files = []
        else:
            self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
            
    def __len__(self) -> int:
        return max(len(self.image_files), 10)

    def __getitem__(self, idx: int) -> dict:
        if not self.image_files:
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))
            sample = {"image": image}
        else:
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert("RGB")
            sample = {"image": image}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
