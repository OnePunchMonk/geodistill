"""Abstract base class for all teacher models.

Defines the common interface that every teacher wrapper must implement,
including inference, output normalization, and confidence estimation.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from geodistill.config import TeacherConfig

logger = logging.getLogger(__name__)


@dataclass
class TeacherOutput:
    """Container for teacher model outputs.

    Attributes:
        depth: Metric depth map of shape (B, 1, H, W).
        normals: Surface normal map of shape (B, 3, H, W), unit-length.
        point_map: 3D point map of shape (B, 3, H, W) in camera coordinates.
        depth_confidence: Per-pixel confidence for depth, shape (B, 1, H, W).
        normal_confidence: Per-pixel confidence for normals, shape (B, 1, H, W).
        features: Optional intermediate features dict {layer_idx: (B, C, H', W')}.
    """

    depth: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None
    point_map: Optional[torch.Tensor] = None
    depth_confidence: Optional[torch.Tensor] = None
    normal_confidence: Optional[torch.Tensor] = None
    features: Optional[dict[int, torch.Tensor]] = None

    def to(self, device: torch.device | str) -> TeacherOutput:
        """Move all tensors to the specified device.

        Args:
            device: Target device.

        Returns:
            Self with all tensors moved.
        """

        def _move(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        self.depth = _move(self.depth)
        self.normals = _move(self.normals)
        self.point_map = _move(self.point_map)
        self.depth_confidence = _move(self.depth_confidence)
        self.normal_confidence = _move(self.normal_confidence)
        if self.features is not None:
            self.features = {k: v.to(device) for k, v in self.features.items()}
        return self

    def resize(self, size: tuple[int, int]) -> TeacherOutput:
        """Resize all spatial tensors to the target size.

        Args:
            size: Target (H, W).

        Returns:
            Self with all spatial tensors resized.
        """
        h, w = size

        def _resize(
            t: Optional[torch.Tensor], mode: str = "bilinear"
        ) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if t.shape[-2:] == (h, w):
                return t
            return F.interpolate(
                t, size=(h, w), mode=mode, align_corners=False if mode != "nearest" else None
            )

        self.depth = _resize(self.depth)
        self.normals = _resize(self.normals)
        self.point_map = _resize(self.point_map)
        self.depth_confidence = _resize(self.depth_confidence)
        self.normal_confidence = _resize(self.normal_confidence)

        # Re-normalize normals after interpolation
        if self.normals is not None:
            self.normals = F.normalize(self.normals, dim=1, eps=1e-6)

        return self


class BaseTeacher(nn.Module, abc.ABC):
    """Abstract base class for teacher model wrappers.

    All teacher models are frozen at construction and run in eval mode.
    They produce structured geometric outputs (depth, normals, point maps)
    along with optional per-pixel confidence estimates.
    """

    def __init__(self, config: TeacherConfig) -> None:
        """Initialize the base teacher.

        Args:
            config: Teacher configuration.
        """
        super().__init__()
        self.config = config
        self._is_loaded = False

    def _freeze(self) -> None:
        """Freeze all parameters and set to eval mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        logger.info("%s: All parameters frozen.", self.__class__.__name__)

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model weights.

        Must be implemented by subclasses. Should set self._is_loaded = True.
        """
        ...

    @abc.abstractmethod
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        """Run teacher inference on a batch of images.

        Args:
            images: RGB images of shape (B, 3, H, W), normalized to [0, 1].

        Returns:
            TeacherOutput with available geometric predictions.
        """
        ...

    @staticmethod
    def depth_to_normals(depth: torch.Tensor, focal_length: float = 1.0) -> torch.Tensor:
        """Derive surface normals from a depth map using Sobel-like finite differences.

        Uses the cross-product of horizontal and vertical depth gradients
        in camera coordinates to compute per-pixel surface normals.

        Args:
            depth: Depth map of shape (B, 1, H, W).
            focal_length: Focal length for unprojection (default 1.0 for relative).

        Returns:
            Surface normals of shape (B, 3, H, W), unit-length.
        """
        b, _, h, w = depth.shape

        # Compute spatial gradients using Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth.dtype,
            device=depth.device,
        ).reshape(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth.dtype,
            device=depth.device,
        ).reshape(1, 1, 3, 3) / 8.0

        # Pad depth to handle borders
        depth_padded = F.pad(depth, (1, 1, 1, 1), mode="replicate")

        dz_dx = F.conv2d(depth_padded, sobel_x)
        dz_dy = F.conv2d(depth_padded, sobel_y)

        # Normal = (-dz/dx, -dz/dy, 1) then normalize
        ones = torch.ones_like(dz_dx) / focal_length
        normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-6)

        return normals

    @staticmethod
    def normalize_depth(
        depth: torch.Tensor,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
    ) -> torch.Tensor:
        """Clamp and normalize depth to a standard range.

        Args:
            depth: Raw depth values of shape (B, 1, H, W).
            min_depth: Minimum valid depth value.
            max_depth: Maximum valid depth value.

        Returns:
            Clamped depth tensor.
        """
        return depth.clamp(min=min_depth, max=max_depth)

    def train(self, mode: bool = True) -> BaseTeacher:
        """Override train to keep teacher always in eval mode.

        Args:
            mode: Ignored; teacher is always in eval mode.

        Returns:
            Self.
        """
        return super().train(False)
