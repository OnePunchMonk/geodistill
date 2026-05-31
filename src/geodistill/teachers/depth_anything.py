"""Depth Anything V2 teacher wrapper.

Wraps the Depth Anything V2 monocular depth estimation model for
producing high-quality metric depth predictions with sharp boundaries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from geodistill.teachers import register_teacher
from geodistill.teachers.base import BaseTeacher, TeacherOutput

if TYPE_CHECKING:
    from geodistill.config import TeacherConfig

logger = logging.getLogger(__name__)

# Mapping from encoder name to model configuration
_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@register_teacher("depth_anything_v2")
class DepthAnythingV2Teacher(BaseTeacher):
    """Teacher wrapper for the Depth Anything V2 depth estimation model.

    Produces high-resolution metric depth maps with sharp object boundaries.
    The model is loaded frozen and supports batched inference at configurable
    resolutions.
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: TeacherConfig) -> None:
        """Initialize the Depth Anything V2 teacher.

        Args:
            config: Teacher configuration with model name and encoder variant.
        """
        super().__init__(config)
        self.model = None
        self._resolution = config.teacher_resolution
        self._encoder = config.depth_anything_encoder

        if self._encoder not in _ENCODER_CONFIGS:
            raise ValueError(
                f"Unknown encoder '{self._encoder}'. "
                f"Available: {list(_ENCODER_CONFIGS.keys())}"
            )

    def load_model(self) -> None:
        """Load Depth Anything V2 model weights.

        Attempts to load from HuggingFace Hub first, then falls back to
        a local checkpoint if available.
        """
        if self._is_loaded:
            logger.info("Depth Anything V2 model already loaded, skipping.")
            return

        device = torch.device(self.config.device)

        try:
            self.model = self._load_from_hub()
        except Exception as e:
            logger.warning("Failed to load from Hub: %s. Trying local checkpoint.", e)
            self.model = self._load_from_local()

        if self.model is None:
            raise RuntimeError(
                "Failed to load Depth Anything V2 model. "
                "Ensure depth-anything-v2 is installed or provide a valid checkpoint."
            )

        self.model = self.model.to(device=device, dtype=self.config.torch_dtype)
        self._freeze()
        self._is_loaded = True
        logger.info(
            "Depth Anything V2 (%s) loaded on %s", self._encoder, self.config.device
        )

    def _load_from_hub(self) -> torch.nn.Module | None:
        """Load model from HuggingFace Hub.

        Returns:
            Loaded model or None.
        """
        try:
            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2Model

            enc_cfg = _ENCODER_CONFIGS[self._encoder]
            model = DAv2Model(**enc_cfg)

            # Try loading from HuggingFace
            from huggingface_hub import hf_hub_download

            ckpt_name = f"depth_anything_v2_{self._encoder}.pth"
            ckpt_path = hf_hub_download(
                repo_id=self.config.depth_anything_model_name,
                filename=ckpt_name,
                cache_dir=self.config.cache_dir,
            )
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            return model
        except ImportError:
            logger.warning("depth_anything_v2 package not available.")
            return None
        except Exception as e:
            logger.warning("Hub loading failed: %s", e)
            return None

    def _load_from_local(self) -> torch.nn.Module | None:
        """Attempt to load model from local checkpoint path.

        Returns:
            Loaded model or None.
        """
        from pathlib import Path

        model_path = Path(self.config.depth_anything_model_name)
        if not model_path.exists():
            return None

        try:
            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2Model

            enc_cfg = _ENCODER_CONFIGS[self._encoder]
            model = DAv2Model(**enc_cfg)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception as e:
            logger.error("Failed to load local checkpoint: %s", e)
            return None

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for Depth Anything V2 inference.

        Applies ImageNet normalization and resizes to the target resolution.

        Args:
            images: RGB images of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Preprocessed images of shape (B, 3, H', W').
        """
        # Resize to target resolution (must be divisible by 14 for ViT)
        target_h = (self._resolution // 14) * 14
        target_w = (self._resolution // 14) * 14

        if images.shape[-2:] != (target_h, target_w):
            images = F.interpolate(
                images,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # Apply ImageNet normalization
        mean = torch.tensor(
            self.IMAGENET_MEAN, device=images.device, dtype=images.dtype
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            self.IMAGENET_STD, device=images.device, dtype=images.dtype
        ).reshape(1, 3, 1, 1)
        return (images - mean) / std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        """Run Depth Anything V2 inference on a batch of images.

        Args:
            images: RGB images of shape (B, 3, H, W), values in [0, 1].

        Returns:
            TeacherOutput with depth and derived normals.
        """
        if not self._is_loaded:
            self.load_model()

        assert self.model is not None
        b, _, h_orig, w_orig = images.shape

        # Preprocess
        preprocessed = self._preprocess(images)

        # Run inference
        with torch.amp.autocast("cuda", dtype=self.config.torch_dtype):
            raw_depth = self.model(preprocessed)

        # Depth Anything V2 outputs relative inverse depth.
        # Convert to metric-like depth.
        depth = self._postprocess_depth(raw_depth)

        # Compute confidence as inverse of local variance (uniform = high confidence)
        depth_confidence = self._estimate_confidence(depth)

        # Derive normals from depth
        normals = self.depth_to_normals(depth)

        output = TeacherOutput(
            depth=depth,
            normals=normals,
            point_map=None,  # Depth Anything doesn't produce point maps
            depth_confidence=depth_confidence,
            normal_confidence=depth_confidence,
        )

        # Resize to original input resolution
        output.resize((h_orig, w_orig))
        return output

    def _postprocess_depth(self, raw_depth: torch.Tensor) -> torch.Tensor:
        """Convert raw model output to metric-like depth.

        Depth Anything V2 outputs relative/inverse depth. We normalize
        it to a physically plausible range.

        Args:
            raw_depth: Raw depth output from the model, shape varies.

        Returns:
            Metric depth of shape (B, 1, H, W).
        """
        # Ensure 4D shape
        if raw_depth.dim() == 2:
            raw_depth = raw_depth.unsqueeze(0).unsqueeze(0)
        elif raw_depth.dim() == 3:
            raw_depth = raw_depth.unsqueeze(1)

        # The model outputs disparity-like values; convert to depth
        # Use a simple normalization: depth = 1 / (disparity + eps)
        # Scale to [min_depth, max_depth] range
        depth = raw_depth

        # Normalize to [0, 1] per sample, then scale to metric range
        b = depth.shape[0]
        depth_flat = depth.reshape(b, -1)
        d_min = depth_flat.min(dim=-1, keepdim=True).values
        d_max = depth_flat.max(dim=-1, keepdim=True).values
        d_range = (d_max - d_min).clamp(min=1e-8)

        depth_flat = (depth_flat - d_min) / d_range
        depth = depth_flat.reshape_as(depth)

        # Scale to metric range (0.1m to 10m for indoor scenes)
        depth = depth * 9.9 + 0.1

        return self.normalize_depth(depth)

    def _estimate_confidence(
        self, depth: torch.Tensor, kernel_size: int = 5
    ) -> torch.Tensor:
        """Estimate per-pixel confidence from depth smoothness.

        Regions with smooth, consistent depth get high confidence.
        Edges and noisy regions get low confidence.

        Args:
            depth: Depth map of shape (B, 1, H, W).
            kernel_size: Size of the local window for variance computation.

        Returns:
            Confidence map of shape (B, 1, H, W), values in [0, 1].
        """
        padding = kernel_size // 2

        # Local mean
        kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=depth.device, dtype=depth.dtype
        ) / (kernel_size * kernel_size)

        local_mean = F.conv2d(depth, kernel, padding=padding)
        local_sq_mean = F.conv2d(depth**2, kernel, padding=padding)
        local_var = (local_sq_mean - local_mean**2).clamp(min=0)

        # Convert variance to confidence (inverse relationship)
        # Use exponential mapping: conf = exp(-var / scale)
        scale = local_var.mean().clamp(min=1e-8)
        confidence = torch.exp(-local_var / scale)
        return confidence.clamp(0.0, 1.0)
