"""Student ViT model with LoRA adapters for GeoDistill."""

import math
from typing import Optional

import torch
import torch.nn as nn
import timm

from geodistill.config import StudentConfig, PretrainSource


class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation) injection."""

    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.05, use_rslora: bool = False):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        
        self.rank = rank
        self.alpha = alpha
        self.use_rslora = use_rslora
        
        if self.use_rslora:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.dropout(x)
        lora_out = nn.functional.linear(lora_out, self.lora_A)
        lora_out = nn.functional.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling


class LoRAQKV(nn.Module):
    """LoRA injection specifically for Q and V in QKV projection."""

    def __init__(self, qkv_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.05, use_rslora: bool = False):
        super().__init__()
        self.in_features = qkv_layer.in_features
        self.out_features = qkv_layer.out_features
        self.weight = qkv_layer.weight
        self.bias = qkv_layer.bias
        self.embed_dim = self.out_features // 3

        self.rank = rank
        self.scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank

        # Only adapt Q and V
        self.lora_A_q = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(self.embed_dim, rank))
        self.lora_A_v = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(self.embed_dim, rank))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = nn.functional.linear(x, self.weight, self.bias)
        
        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply LoRA to Q and V
        lora_x = self.dropout(x)
        delta_q = nn.functional.linear(nn.functional.linear(lora_x, self.lora_A_q), self.lora_B_q) * self.scaling
        delta_v = nn.functional.linear(nn.functional.linear(lora_x, self.lora_A_v), self.lora_B_v) * self.scaling
        
        q = q + delta_q
        v = v + delta_v
        
        return torch.cat([q, k, v], dim=-1)


def inject_lora(model: nn.Module, config: StudentConfig):
    """Inject LoRA adapters into the backbone model."""
    for i, block in enumerate(model.blocks):
        if config.lora.layers is not None and i not in config.lora.layers:
            continue
            
        if "qkv" in config.lora.target_modules:
            block.attn.qkv = LoRAQKV(
                block.attn.qkv, 
                rank=config.lora.rank, 
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                use_rslora=config.lora.use_rslora
            )
            
        if "fc1" in config.lora.target_modules:
            block.mlp.fc1 = LoRALinear(
                block.mlp.fc1,
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                use_rslora=config.lora.use_rslora
            )
            
        if "fc2" in config.lora.target_modules:
            block.mlp.fc2 = LoRALinear(
                block.mlp.fc2,
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                use_rslora=config.lora.use_rslora
            )

    return model


class GeoDistillStudent(nn.Module):
    """GeoDistill Student Model (ViT + LoRA)."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        # Determine model name for timm
        model_name = config.backbone.value
        
        # Adjust for dinov2 / registers
        if config.pretrain_source == PretrainSource.DINOV2:
            # Note: timm uses different names for dinov2
            if "small_patch14" in model_name:
                model_name = "vit_small_patch14_reg4_dinov2.lvd142m" if config.num_registers > 0 else "vit_small_patch14_dinov2.lvd142m"
            elif "base_patch14" in model_name:
                model_name = "vit_base_patch14_reg4_dinov2.lvd142m" if config.num_registers > 0 else "vit_base_patch14_dinov2.lvd142m"
                
        # Initialize backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=(config.pretrain_source != PretrainSource.NONE),
            img_size=config.input_resolution,
            drop_path_rate=config.drop_path_rate,
            dynamic_img_size=True
        )

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if config.use_gradient_checkpointing:
            self.backbone.set_grad_checkpointing()

        # Inject LoRA
        self.backbone = inject_lora(self.backbone, config)
        
        # Ensure LoRA parameters require gradients
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass extracting multi-scale features.
        
        Returns:
            dict containing:
                - features: List of intermediate block features
                - patch_tokens: Final patch tokens
                - cls_token: Final class token
        """
        B, C, H, W = x.shape
        
        patch_h, patch_w = H // self.config.patch_size, W // self.config.patch_size
        
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        features = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.config.feature_layers:
                # Remove cls and register tokens to get spatial features
                num_extra_tokens = 1 + self.config.num_registers
                feat = x[:, num_extra_tokens:]
                feat = feat.reshape(B, patch_h, patch_w, -1).permute(0, 3, 1, 2)
                features.append(feat.contiguous())

        x = self.backbone.norm(x)
        
        num_extra_tokens = 1 + self.config.num_registers
        cls_token = x[:, 0]
        patch_tokens = x[:, num_extra_tokens:]
        
        return {
            "features": features,
            "patch_tokens": patch_tokens,
            "cls_token": cls_token
        }
