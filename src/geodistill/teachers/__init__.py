"""Teacher model registry and factory.

Provides a unified interface for instantiating teacher models by name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from geodistill.teachers.base import BaseTeacher

if TYPE_CHECKING:
    from geodistill.config import TeacherConfig

logger = logging.getLogger(__name__)

_TEACHER_REGISTRY: dict[str, type[BaseTeacher]] = {}


def register_teacher(name: str):
    """Decorator to register a teacher class under a given name.

    Args:
        name: Registry key for the teacher class.
    """

    def decorator(cls: type[BaseTeacher]) -> type[BaseTeacher]:
        _TEACHER_REGISTRY[name] = cls
        return cls

    return decorator


def build_teacher(config: TeacherConfig) -> BaseTeacher:
    """Build a teacher model from configuration.

    Args:
        config: Teacher configuration specifying model type and parameters.

    Returns:
        Instantiated and frozen teacher model.

    Raises:
        ValueError: If the teacher type is not registered.
    """
    # Lazy imports to trigger registration
    from geodistill.teachers import depth_anything, multi_teacher, vggt  # noqa: F401

    teacher_name = config.teacher_type.value
    if teacher_name not in _TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown teacher type '{teacher_name}'. "
            f"Available: {list(_TEACHER_REGISTRY.keys())}"
        )

    teacher_cls = _TEACHER_REGISTRY[teacher_name]
    teacher = teacher_cls(config)
    logger.info("Built teacher: %s", teacher_name)
    return teacher


__all__ = ["BaseTeacher", "build_teacher", "register_teacher"]
