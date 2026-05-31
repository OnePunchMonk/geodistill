"""Multi-teacher ensemble.

Combines outputs from multiple teacher models (VGGT + Depth Anything V2)
using uncertainty-weighted averaging and per-region confidence selection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from geodistill.teachers import register_teacher
from geodistill.teachers.base import BaseTeacher, TeacherOutput
from geodistill.teachers.depth_anything import DepthAnythingV2Teacher
from geodistill.teachers.vggt import VGGTTeacher

if TYPE_CHECKING:
    from geodistill.config import TeacherConfig

logger = logging.getLogger(__name__)


@register_teacher("multi")
class MultiTeacher(BaseTeacher):
    """Multi-teacher ensemble combining VGGT and Depth Anything V2.

    Aggregates geometric predictions from multiple teachers using
    uncertainty-weighted averaging for depth and confidence-based
    selection for normals. The ensemble leverages each teacher's
    strengths: VGGT for 3D structure and Depth Anything V2 for
    sharp depth boundaries.
    """

    def __init__(self, config: TeacherConfig) -> None:
        """Initialize the multi-teacher ensemble.

        Args:
            config: Teacher configuration. Both VGGT and Depth Anything
                    settings should be populated.
        """
        super().__init__(config)
        self.vggt_teacher = VGGTTeacher(config)
        self.depth_anything_teacher = DepthAnythingV2Teacher(config)
        self._use_confidence = config.use_confidence_weighting

    def load_model(self) -> None:
        """Load all constituent teacher models.

        Both VGGT and Depth Anything V2 models are loaded, frozen,
        and placed in eval mode.
        """
        if self._is_loaded:
            return

        logger.info("Loading multi-teacher ensemble...")

        try:
            self.vggt_teacher.load_model()
            self._has_vggt = True
        except Exception as e:
            logger.warning("Failed to load VGGT teacher: %s. Continuing without it.", e)
            self._has_vggt = False

        try:
            self.depth_anything_teacher.load_model()
            self._has_da = True
        except Exception as e:
            logger.warning(
                "Failed to load Depth Anything V2 teacher: %s. Continuing without it.",
                e,
            )
            self._has_da = False

        if not self._has_vggt and not self._has_da:
            raise RuntimeError("No teacher models could be loaded.")

        self._freeze()
        self._is_loaded = True
        logger.info(
            "Multi-teacher loaded. VGGT: %s, Depth Anything V2: %s",
            self._has_vggt,
            self._has_da,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        """Run ensemble inference on a batch of images.

        Runs all available teachers and combines their outputs using
        uncertainty-weighted averaging.

        Args:
            images: RGB images of shape (B, 3, H, W), values in [0, 1].

        Returns:
            TeacherOutput with combined depth, normals, point maps, and
            aggregated confidence maps.
        """
        if not self._is_loaded:
            self.load_model()

        b, _, h, w = images.shape
        outputs: list[TeacherOutput] = []

        # Collect outputs from available teachers
        if self._has_vggt:
            vggt_out = self.vggt_teacher(images)
            vggt_out.resize((h, w))
            outputs.append(vggt_out)

        if self._has_da:
            da_out = self.depth_anything_teacher(images)
            da_out.resize((h, w))
            outputs.append(da_out)

        if len(outputs) == 1:
            return outputs[0]

        # Combine depth predictions with uncertainty-weighted averaging
        depth = self._combine_depth(outputs)

        # Combine normals: use confidence-weighted average
        normals = self._combine_normals(outputs)

        # Point map comes from VGGT only (Depth Anything doesn't produce it)
        point_map = None
        if self._has_vggt and outputs[0].point_map is not None:
            point_map = outputs[0].point_map

        # Aggregate confidence as the maximum confidence across teachers
        depth_confidence = self._aggregate_confidence(
            [o.depth_confidence for o in outputs]
        )
        normal_confidence = self._aggregate_confidence(
            [o.normal_confidence for o in outputs]
        )

        return TeacherOutput(
            depth=depth,
            normals=normals,
            point_map=point_map,
            depth_confidence=depth_confidence,
            normal_confidence=normal_confidence,
        )

    def _combine_depth(self, outputs: list[TeacherOutput]) -> torch.Tensor:
        """Combine depth predictions using uncertainty-weighted averaging.

        If confidence weighting is enabled, depth maps are weighted by
        their respective confidence maps. Otherwise, simple averaging is used.

        Args:
            outputs: List of TeacherOutput from individual teachers.

        Returns:
            Combined depth map of shape (B, 1, H, W).
        """
        depths = [o.depth for o in outputs if o.depth is not None]
        if not depths:
            raise RuntimeError("No depth predictions available from any teacher.")

        if len(depths) == 1:
            return depths[0]

        # Align scales between teachers using least-squares
        depths = self._align_depth_scales(depths)

        if not self._use_confidence:
            return torch.stack(depths).mean(dim=0)

        # Uncertainty-weighted combination
        confidences = []
        for o in outputs:
            if o.depth is not None:
                conf = (
                    o.depth_confidence
                    if o.depth_confidence is not None
                    else torch.ones_like(o.depth)
                )
                confidences.append(conf)

        weights = torch.stack(confidences, dim=0)  # (N, B, 1, H, W)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        depth_stack = torch.stack(depths, dim=0)
        combined = (depth_stack * weights).sum(dim=0)
        return combined

    def _combine_normals(self, outputs: list[TeacherOutput]) -> torch.Tensor:
        """Combine normal predictions using confidence-weighted averaging.

        After averaging, normals are re-normalized to unit length.

        Args:
            outputs: List of TeacherOutput from individual teachers.

        Returns:
            Combined normal map of shape (B, 3, H, W), unit-length.
        """
        normals = [o.normals for o in outputs if o.normals is not None]
        if not normals:
            raise RuntimeError("No normal predictions available from any teacher.")

        if len(normals) == 1:
            return normals[0]

        if not self._use_confidence:
            combined = torch.stack(normals).mean(dim=0)
            return F.normalize(combined, dim=1, eps=1e-6)

        # Confidence-weighted averaging
        confidences = []
        for o in outputs:
            if o.normals is not None:
                conf = (
                    o.normal_confidence
                    if o.normal_confidence is not None
                    else torch.ones(
                        o.normals.shape[0],
                        1,
                        o.normals.shape[2],
                        o.normals.shape[3],
                        device=o.normals.device,
                        dtype=o.normals.dtype,
                    )
                )
                confidences.append(conf)

        weights = torch.stack(confidences, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        normal_stack = torch.stack(normals, dim=0)
        combined = (normal_stack * weights).sum(dim=0)
        return F.normalize(combined, dim=1, eps=1e-6)

    def _align_depth_scales(
        self, depths: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Align depth scales across teachers using median-based normalization.

        All depth maps are scaled so that their median depth matches
        the first teacher's median depth.

        Args:
            depths: List of depth tensors of shape (B, 1, H, W).

        Returns:
            Scale-aligned depth tensors.
        """
        if len(depths) <= 1:
            return depths

        # Use first teacher as reference
        ref = depths[0]
        ref_median = ref.flatten(1).median(dim=1).values.reshape(-1, 1, 1, 1)
        ref_median = ref_median.clamp(min=1e-6)

        aligned = [ref]
        for d in depths[1:]:
            d_median = d.flatten(1).median(dim=1).values.reshape(-1, 1, 1, 1)
            d_median = d_median.clamp(min=1e-6)
            scale = ref_median / d_median
            aligned.append(d * scale)

        return aligned

    def _aggregate_confidence(
        self, confidences: list[torch.Tensor | None]
    ) -> torch.Tensor | None:
        """Aggregate confidence maps by taking element-wise maximum.

        Args:
            confidences: List of confidence tensors or None values.

        Returns:
            Aggregated confidence of shape (B, 1, H, W) or None.
        """
        valid = [c for c in confidences if c is not None]
        if not valid:
            return None

        if len(valid) == 1:
            return valid[0]

        return torch.stack(valid).max(dim=0).values
