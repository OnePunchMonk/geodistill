"""Configuration dataclasses for the GeoDistill pipeline.

All configurable parameters are defined here as frozen or mutable dataclasses
to provide type-safe, documented configuration throughout the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TeacherType(str, Enum):
    """Enumeration of supported teacher model types."""

    VGGT = "vggt"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    MULTI = "multi"


class BackboneType(str, Enum):
    """Enumeration of supported student backbone types."""

    VIT_SMALL_16 = "vit_small_patch16_224"
    VIT_BASE_16 = "vit_base_patch16_224"
    VIT_SMALL_14 = "vit_small_patch14_518"
    VIT_BASE_14 = "vit_base_patch14_518"


class PretrainSource(str, Enum):
    """Source of pretrained weights for the student backbone."""

    DINOV2 = "dinov2"
    MAE = "mae"
    IMAGENET = "imagenet"
    NONE = "none"


@dataclass
class TeacherConfig:
    """Configuration for teacher model(s).

    Attributes:
        teacher_type: Which teacher model to use.
        vggt_model_name: HuggingFace model name or local path for VGGT.
        depth_anything_model_name: Model name for Depth Anything V2.
        depth_anything_encoder: Encoder variant (vits, vitb, vitl, vitg).
        teacher_resolution: Input resolution for the teacher model.
        use_confidence_weighting: Whether to use teacher confidence maps.
        cache_dir: Directory to cache teacher model weights.
        device: Device for teacher inference.
        dtype: Data type for teacher inference (float32 or float16).
    """

    teacher_type: TeacherType = TeacherType.MULTI
    vggt_model_name: str = "facebook/vggt-1b"
    depth_anything_model_name: str = "depth-anything/Depth-Anything-V2-Large"
    depth_anything_encoder: str = "vitl"
    teacher_resolution: int = 518
    use_confidence_weighting: bool = True
    cache_dir: Optional[str] = None
    device: str = "cuda"
    dtype: str = "float32"

    @property
    def torch_dtype(self) -> "torch.dtype":
        import torch

        return torch.float16 if self.dtype == "float16" else torch.float32


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) injection.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA (effective scale = alpha / rank).
        dropout: Dropout probability applied to LoRA layers.
        target_modules: Which projection matrices to inject LoRA into.
        layers: Which transformer block indices get LoRA. None means all layers.
        use_rslora: Whether to use rank-stabilized LoRA scaling.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["qkv"])
    layers: Optional[list[int]] = None
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        """Compute the effective LoRA scaling factor."""
        if self.use_rslora:
            import math

            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


@dataclass
class StudentConfig:
    """Configuration for the student model.

    Attributes:
        backbone: Which ViT backbone to use.
        pretrain_source: Source of pretrained weights.
        pretrained_weights_path: Optional explicit path to pretrained weights.
        freeze_backbone: Whether to freeze the backbone (only LoRA + decoder train).
        lora: LoRA configuration.
        num_registers: Number of post-hoc register tokens (0 to disable).
        feature_layers: Which transformer block indices to extract features from.
        input_resolution: Input image resolution for the student.
        drop_path_rate: Stochastic depth rate.
        use_gradient_checkpointing: Whether to enable gradient checkpointing.
    """

    backbone: BackboneType = BackboneType.VIT_BASE_14
    pretrain_source: PretrainSource = PretrainSource.DINOV2
    pretrained_weights_path: Optional[str] = None
    freeze_backbone: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    num_registers: int = 0
    feature_layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11])
    input_resolution: int = 518
    drop_path_rate: float = 0.0
    use_gradient_checkpointing: bool = False

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension based on backbone type."""
        if "small" in self.backbone.value:
            return 384
        return 768

    @property
    def num_heads(self) -> int:
        """Return the number of attention heads based on backbone type."""
        if "small" in self.backbone.value:
            return 6
        return 12

    @property
    def patch_size(self) -> int:
        """Return the patch size based on backbone type."""
        if "patch14" in self.backbone.value:
            return 14
        return 16

    @property
    def num_blocks(self) -> int:
        """Return the number of transformer blocks."""
        return 12


@dataclass
class DecoderConfig:
    """Configuration for the DPT decoder head.

    Attributes:
        hidden_dim: Hidden feature dimension in the decoder.
        predict_depth: Whether to include the depth prediction head.
        predict_normals: Whether to include the normal prediction head.
        predict_point_map: Whether to include the point map prediction head.
        fusion_type: Type of feature fusion ('add', 'concat', 'bilinear').
        upsample_mode: Interpolation mode for upsampling.
        num_fusion_layers: Number of convolutional layers per fusion block.
    """

    hidden_dim: int = 256
    predict_depth: bool = True
    predict_normals: bool = True
    predict_point_map: bool = False
    fusion_type: str = "add"
    upsample_mode: str = "bilinear"
    num_fusion_layers: int = 2


@dataclass
class LossConfig:
    """Configuration for multi-target distillation losses.

    Attributes:
        lambda_depth: Weight for depth loss.
        lambda_normal: Weight for surface normal loss.
        lambda_point_map: Weight for point map loss.
        lambda_feature_kl: Weight for feature-level KL divergence loss.
        lambda_geometric_consistency: Weight for geometric consistency loss.
        lambda_edge_aware: Weight for edge-aware smoothness loss.
        depth_loss_type: Type of depth loss ('si_mse', 'l1', 'berhu').
        normal_loss_type: Type of normal loss ('cosine', 'l1').
        feature_kl_temperature: Temperature for KL divergence softening.
        feature_kl_layers: Which layers to apply feature KL loss on.
        grad_matching_weight: Weight for gradient-matching term in depth loss.
        use_uncertainty_weighting: Whether to use learned uncertainty weighting.
    """

    lambda_depth: float = 1.0
    lambda_normal: float = 1.0
    lambda_point_map: float = 0.5
    lambda_feature_kl: float = 0.1
    lambda_geometric_consistency: float = 0.5
    lambda_edge_aware: float = 0.2
    depth_loss_type: str = "si_mse"
    normal_loss_type: str = "cosine"
    feature_kl_temperature: float = 4.0
    feature_kl_layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11])
    grad_matching_weight: float = 0.5
    use_uncertainty_weighting: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation.

    Attributes:
        dataset_name: Name of the dataset to use.
        data_root: Root directory for dataset files.
        teacher_cache_dir: Directory with pre-computed teacher outputs.
        train_split: Name of the training split.
        val_split: Name of the validation split.
        image_size: Target image size (H, W) after transforms.
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for data loading.
        augment: Whether to apply data augmentation.
        color_jitter: Strength of color jitter augmentation.
        random_crop_scale: Scale range for random resized crop.
        horizontal_flip_prob: Probability of horizontal flip.
    """

    dataset_name: str = "nyuv2"
    data_root: str = "./data"
    teacher_cache_dir: Optional[str] = None
    train_split: str = "train"
    val_split: str = "test"
    image_size: tuple[int, int] = (518, 518)
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True
    color_jitter: float = 0.2
    random_crop_scale: tuple[float, float] = (0.8, 1.0)
    horizontal_flip_prob: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes:
        max_epochs: Maximum number of training epochs.
        max_steps: Maximum number of training steps (overrides epochs if set).
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW optimizer.
        warmup_steps: Number of linear warmup steps.
        lr_scheduler: Learning rate scheduler type.
        min_lr_ratio: Minimum LR as a ratio of peak LR for cosine schedule.
        gradient_clip_norm: Maximum gradient norm for clipping.
        mixed_precision: Mixed precision training mode ('no', 'fp16', 'bf16').
        seed: Random seed for reproducibility.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluation runs.
        save_interval: Steps between checkpoint saves.
        output_dir: Directory for checkpoints and logs.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity/team name.
        wandb_run_name: Optional run name for W&B.
        resume_from: Path to checkpoint to resume from.
    """

    max_epochs: int = 50
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.01
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 2000
    output_dir: str = "./outputs"
    wandb_project: str = "geodistill"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    resume_from: Optional[str] = None


@dataclass
class GeoDistillConfig:
    """Top-level configuration aggregating all sub-configs.

    Attributes:
        teacher: Teacher model configuration.
        student: Student model configuration.
        decoder: Decoder head configuration.
        loss: Loss function configuration.
        data: Data loading configuration.
        training: Training loop configuration.
    """

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is inconsistent.
        """
        # Ensure feature layers are valid block indices
        max_blocks = self.student.num_blocks
        for layer_idx in self.student.feature_layers:
            if layer_idx < 0 or layer_idx >= max_blocks:
                raise ValueError(
                    f"Feature layer index {layer_idx} is out of range "
                    f"[0, {max_blocks - 1}] for backbone {self.student.backbone.value}."
                )

        # Ensure KL layers are a subset of feature layers
        for layer_idx in self.loss.feature_kl_layers:
            if layer_idx not in self.student.feature_layers:
                raise ValueError(
                    f"Feature KL layer {layer_idx} must be in student "
                    f"feature_layers {self.student.feature_layers}."
                )

        # Ensure point map loss weight is 0 if decoder doesn't predict point maps
        if self.loss.lambda_point_map > 0 and not self.decoder.predict_point_map:
            logger.warning(
                "lambda_point_map > 0 but decoder.predict_point_map is False. "
                "Enabling point map prediction in decoder."
            )
            self.decoder.predict_point_map = True

        # Validate mixed precision mode
        valid_modes = {"no", "fp16", "bf16"}
        if self.training.mixed_precision not in valid_modes:
            raise ValueError(
                f"Invalid mixed_precision mode '{self.training.mixed_precision}'. "
                f"Must be one of {valid_modes}."
            )

        logger.info("Configuration validated successfully.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeoDistillConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated GeoDistillConfig instance.
        """
        from omegaconf import OmegaConf

        raw = OmegaConf.load(path)
        cfg_dict = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(cfg_dict, dict)

        teacher_cfg = TeacherConfig(**cfg_dict.get("teacher", {}))
        lora_cfg = LoRAConfig(**cfg_dict.get("student", {}).pop("lora", {}))
        student_kwargs = cfg_dict.get("student", {})
        if "backbone" in student_kwargs:
            student_kwargs["backbone"] = BackboneType(student_kwargs["backbone"])
        if "pretrain_source" in student_kwargs:
            student_kwargs["pretrain_source"] = PretrainSource(
                student_kwargs["pretrain_source"]
            )
        if "image_size" in student_kwargs:
            student_kwargs["image_size"] = tuple(student_kwargs["image_size"])
        student_cfg = StudentConfig(lora=lora_cfg, **student_kwargs)
        decoder_cfg = DecoderConfig(**cfg_dict.get("decoder", {}))
        loss_cfg = LossConfig(**cfg_dict.get("loss", {}))
        data_kwargs = cfg_dict.get("data", {})
        if "image_size" in data_kwargs:
            data_kwargs["image_size"] = tuple(data_kwargs["image_size"])
        if "random_crop_scale" in data_kwargs:
            data_kwargs["random_crop_scale"] = tuple(data_kwargs["random_crop_scale"])
        data_cfg = DataConfig(**data_kwargs)
        training_cfg = TrainingConfig(**cfg_dict.get("training", {}))

        config = cls(
            teacher=teacher_cfg,
            student=student_cfg,
            decoder=decoder_cfg,
            loss=loss_cfg,
            data=data_cfg,
            training=training_cfg,
        )
        config.validate()
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to write the YAML configuration file.
        """
        import dataclasses

        from omegaconf import OmegaConf

        def _to_serializable(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                for f in dataclasses.fields(obj):
                    val = getattr(obj, f.name)
                    result[f.name] = _to_serializable(val)
                return result
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        serialized = _to_serializable(self)
        conf = OmegaConf.create(serialized)  # type: ignore[arg-type]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(conf, path)
        logger.info("Configuration saved to %s", path)
