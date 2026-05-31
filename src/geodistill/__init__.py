"""GeoDistill: Geometry-Grounded Knowledge Distillation from 3D Foundation Models.

Distills structured geometric targets (metric depth, surface normals, point maps)
from large 3D foundation models (VGGT, Depth Anything V2) into lightweight ViT
students equipped with LoRA adapters.
"""

__version__ = "0.1.0"
__author__ = "GeoDistill Team"

from geodistill.config import (
    GeoDistillConfig,
    LoRAConfig,
    LossConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
)

__all__ = [
    "GeoDistillConfig",
    "LoRAConfig",
    "LossConfig",
    "StudentConfig",
    "TeacherConfig",
    "TrainingConfig",
]
