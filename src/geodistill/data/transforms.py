"""Data transformations for GeoDistill."""

import torch
import torchvision.transforms.functional as F
import random
from typing import Dict, Any, Tuple


class GeoTransforms:
    """Transforms for images, depth, and normal maps."""

    def __init__(self, size: Tuple[int, int], augment: bool = False, hflip_prob: float = 0.5):
        self.size = size
        self.augment = augment
        self.hflip_prob = hflip_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]  # PIL Image
        depth = sample.get("depth", None)  # PIL Image or Tensor
        normal = sample.get("normal", None)  # PIL Image or Tensor

        # Resize
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        if depth is not None:
            depth = F.resize(depth, self.size, interpolation=F.InterpolationMode.NEAREST)
        if normal is not None:
            normal = F.resize(normal, self.size, interpolation=F.InterpolationMode.NEAREST)

        # Horizontal Flip
        if self.augment and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if depth is not None:
                depth = F.hflip(depth)
            if normal is not None:
                normal = F.hflip(normal)
                # Need to invert the X component of the normal vector
                # Normal is assumed to be normalized to [-1, 1] as tensor, but here it might still be PIL/image.
                # If it's a tensor (C,H,W):
                if isinstance(normal, torch.Tensor):
                    normal[0, :, :] = -normal[0, :, :]

        # To Tensor
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        if depth is not None and not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        if normal is not None and not isinstance(normal, torch.Tensor):
            # Typically normal maps are loaded as RGB, need to map [0,1] back to [-1,1]
            normal = F.to_tensor(normal)
            normal = (normal - 0.5) * 2.0

        # Normalize Image
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        sample["image"] = image
        if depth is not None:
            sample["depth"] = depth
        if normal is not None:
            sample["normal"] = normal

        return sample
