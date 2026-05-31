"""VGGT teacher wrapper.

Wraps the VGGT (Visual Geometry Grounded Transformer) model for
producing 3D point maps, metric depth, and surface normals from
single or multi-view RGB images.
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


@register_teacher("vggt")
class VGGTTeacher(BaseTeacher):
    """Teacher wrapper for the VGGT 3D foundation model.

    VGGT takes one or more RGB images and outputs dense 3D predictions
    including point maps, depth, and confidence scores. This wrapper
    handles single-image inference by treating input as 1-view scenes.

    The model is loaded frozen in eval mode and supports batched inference.
    """

    def __init__(self, config: TeacherConfig) -> None:
        """Initialize the VGGT teacher.

        Args:
            config: Teacher configuration with VGGT model name and settings.
        """
        super().__init__(config)
        self.model = None
        self._resolution = config.teacher_resolution

    def load_model(self) -> None:
        """Load VGGT model weights from HuggingFace or local path.

        Loads the model, moves it to the configured device and dtype,
        freezes all parameters, and sets to eval mode.
        """
        if self._is_loaded:
            logger.info("VGGT model already loaded, skipping.")
            return

        try:
            from vggt.models.vggt import VGGT

            self.model = VGGT.from_pretrained(self.config.vggt_model_name)
        except ImportError:
            logger.warning(
                "vggt package not installed. Attempting to load via "
                "torch.hub or local checkpoint at %s",
                self.config.vggt_model_name,
            )
            # Fallback: try loading a local checkpoint
            self.model = self._load_from_checkpoint(self.config.vggt_model_name)

        if self.model is None:
            raise RuntimeError(
                "Failed to load VGGT model. Install the vggt package or "
                "provide a valid model path."
            )

        device = torch.device(self.config.device)
        self.model = self.model.to(device=device, dtype=self.config.torch_dtype)
        self._freeze()
        self._is_loaded = True
        logger.info(
            "VGGT model loaded on %s with dtype %s",
            self.config.device,
            self.config.dtype,
        )

    def _load_from_checkpoint(self, path: str) -> torch.nn.Module | None:
        """Attempt to load VGGT from a local checkpoint.

        Args:
            path: Path to checkpoint file or directory.

        Returns:
            Loaded model or None if loading fails.
        """
        from pathlib import Path

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.error("VGGT checkpoint not found at %s", path)
            return None

        try:
            from vggt.models.vggt import VGGT

            model = VGGT()
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            return model
        except Exception as e:
            logger.error("Failed to load VGGT from checkpoint: %s", e)
            return None

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for VGGT input.

        VGGT expects images in shape (B, S, 3, H, W) where S is the number
        of views. For single-image inference, S=1.

        Args:
            images: RGB images of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Preprocessed images of shape (B, 1, 3, H, W).
        """
        # Resize to teacher resolution if needed
        if images.shape[-2:] != (self._resolution, self._resolution):
            images = F.interpolate(
                images,
                size=(self._resolution, self._resolution),
                mode="bilinear",
                align_corners=False,
            )

        # VGGT normalization: ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
        images = (images - mean[None, :, None, None]) / std[None, :, None, None]

        # Add view dimension: (B, 3, H, W) -> (B, 1, 3, H, W)
        return images.unsqueeze(1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        """Run VGGT inference on a batch of images.

        Args:
            images: RGB images of shape (B, 3, H, W), values in [0, 1].

        Returns:
            TeacherOutput with point_map, depth, normals, and confidence maps.
        """
        if not self._is_loaded:
            self.load_model()

        assert self.model is not None
        b, _, h_orig, w_orig = images.shape

        # Preprocess
        vggt_input = self._preprocess(images)

        # Run VGGT inference
        with torch.amp.autocast("cuda", dtype=self.config.torch_dtype):
            predictions = self.model(vggt_input)

        # Extract outputs — VGGT returns dict with various predictions
        # Point maps: (B, S, H, W, 3) -> take view 0, permute to (B, 3, H, W)
        point_map = self._extract_point_map(predictions)
        depth = self._extract_depth(predictions, point_map)
        confidence = self._extract_confidence(predictions)
        normals = self.depth_to_normals(depth)

        output = TeacherOutput(
            depth=depth,
            normals=normals,
            point_map=point_map,
            depth_confidence=confidence,
            normal_confidence=confidence,
        )

        # Resize to original input resolution
        output.resize((h_orig, w_orig))
        return output

    def _extract_point_map(self, predictions: dict) -> torch.Tensor:
        """Extract 3D point map from VGGT predictions.

        Args:
            predictions: Raw VGGT model output dictionary.

        Returns:
            Point map of shape (B, 3, H, W).
        """
        # VGGT outputs point maps in world coordinates
        if "world_points" in predictions:
            # Shape: (B, S, H, W, 3) -> (B, 3, H, W) for view 0
            pts = predictions["world_points"][:, 0]  # (B, H, W, 3)
            return pts.permute(0, 3, 1, 2).contiguous()
        elif "point_map" in predictions:
            pts = predictions["point_map"]
            if pts.dim() == 5:
                pts = pts[:, 0]
            if pts.shape[-1] == 3:
                return pts.permute(0, 3, 1, 2).contiguous()
            return pts
        else:
            # Fallback: construct from depth if available
            logger.warning("No point map found in VGGT output, constructing from depth.")
            depth = self._extract_depth_raw(predictions)
            return self._depth_to_point_map(depth)

    def _extract_depth(
        self, predictions: dict, point_map: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract metric depth from VGGT predictions.

        Depth can be derived from the Z-channel of the point map or
        directly from a depth prediction head.

        Args:
            predictions: Raw VGGT model output dictionary.
            point_map: Pre-extracted point map of shape (B, 3, H, W).

        Returns:
            Depth map of shape (B, 1, H, W).
        """
        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 5:
                depth = depth[:, 0]
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            elif depth.shape[1] != 1:
                depth = depth[:, :1]
            return self.normalize_depth(depth)

        # Derive from point map Z-channel
        if point_map is not None:
            depth = point_map[:, 2:3, :, :]  # Z-channel
            return self.normalize_depth(depth.abs())

        raise RuntimeError("Cannot extract depth from VGGT predictions.")

    def _extract_depth_raw(self, predictions: dict) -> torch.Tensor:
        """Extract raw depth from predictions for fallback point map construction.

        Args:
            predictions: VGGT output dictionary.

        Returns:
            Depth tensor of shape (B, 1, H, W).
        """
        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 5:
                depth = depth[:, 0]
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            return depth
        raise RuntimeError("No depth found in VGGT predictions for fallback.")

    def _extract_confidence(self, predictions: dict) -> torch.Tensor | None:
        """Extract per-pixel confidence from VGGT predictions.

        Args:
            predictions: Raw VGGT model output dictionary.

        Returns:
            Confidence map of shape (B, 1, H, W) or None.
        """
        for key in ("confidence", "conf", "point_confidence", "depth_confidence"):
            if key in predictions:
                conf = predictions[key]
                if conf.dim() == 5:
                    conf = conf[:, 0]
                if conf.dim() == 3:
                    conf = conf.unsqueeze(1)
                elif conf.shape[1] != 1:
                    conf = conf[:, :1]
                return conf.sigmoid()  # Ensure [0, 1]

        return None

    def _depth_to_point_map(
        self,
        depth: torch.Tensor,
        fx: float = 500.0,
        fy: float = 500.0,
    ) -> torch.Tensor:
        """Convert depth map to 3D point map assuming pinhole camera.

        Args:
            depth: Depth map of shape (B, 1, H, W).
            fx: Focal length in x.
            fy: Focal length in y.

        Returns:
            Point map of shape (B, 3, H, W).
        """
        b, _, h, w = depth.shape
        device = depth.device
        dtype = depth.dtype

        cx, cy = w / 2.0, h / 2.0

        u = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, -1)
        v = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(-1, w)

        x = (u - cx) / fx * depth[:, 0]
        y = (v - cy) / fy * depth[:, 0]
        z = depth[:, 0]

        return torch.stack([x, y, z], dim=1)
