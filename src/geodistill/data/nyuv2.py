"""NYUv2 Dataset."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from geodistill.data.transforms import GeoTransforms


class NYUv2Dataset(Dataset):
    """NYUv2 dataset for depth and normal evaluation/training."""

    def __init__(self, data_root: str, split: str = "train", transform: GeoTransforms = None):
        self.data_root = Path(data_root) / "nyu_depth_v2"
        self.split = split
        self.transform = transform
        
        self.image_dir = self.data_root / split / "images"
        self.depth_dir = self.data_root / split / "depths"
        
        if not self.image_dir.exists():
            print(f"Warning: NYUv2 directory not found: {self.image_dir}. Mocking dataset for pipeline.")
            self.image_files = []
        else:
            self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
            
    def __len__(self) -> int:
        return max(len(self.image_files), 10)  # Provide 10 dummy items if missing for testing

    def __getitem__(self, idx: int) -> dict:
        if not self.image_files:
            # Mock data if dataset is missing
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))
            depth = np.random.rand(480, 640).astype(np.float32) * 10.0
            sample = {"image": image, "depth": depth}
        else:
            img_path = self.image_files[idx]
            base_name = img_path.stem
            
            image = Image.open(img_path).convert("RGB")
            
            depth_path = self.depth_dir / f"{base_name}.png"
            if depth_path.exists():
                depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
            else:
                depth = np.zeros((image.height, image.width), dtype=np.float32)
                
            sample = {"image": image, "depth": depth}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
