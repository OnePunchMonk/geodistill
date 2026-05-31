"""DPT Decoder Head for GeoDistill."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geodistill.config import DecoderConfig, StudentConfig


class ResidualConvUnit(nn.Module):
    """Residual Convolution Unit (RCU) for DPT."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out


class FeatureFusionBlock(nn.Module):
    """Feature Fusion Block (FFB) for combining multi-scale features."""

    def __init__(self, features: int, fusion_type: str = "add"):
        super().__init__()
        self.fusion_type = fusion_type
        
        self.res_conv_1 = ResidualConvUnit(features)
        self.res_conv_2 = ResidualConvUnit(features)
        
        if self.fusion_type == "concat":
            self.project = nn.Conv2d(features * 2, features, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        res_x = self.res_conv_1(x)
        
        if x.shape[2:] != skip.shape[2:]:
            res_x = F.interpolate(res_x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            
        if self.fusion_type == "add":
            out = res_x + skip
        elif self.fusion_type == "concat":
            out = torch.cat([res_x, skip], dim=1)
            out = self.project(out)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
        out = self.res_conv_2(out)
        return out


class Scratch(nn.Module):
    """Scratch layers to project ViT features to decoder hidden dimensions."""

    def __init__(self, in_shape: list[int], out_shape: int):
        super().__init__()
        
        self.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        out.append(self.layer1_rn(features[0]))
        out.append(self.layer2_rn(features[1]))
        out.append(self.layer3_rn(features[2]))
        out.append(self.layer4_rn(features[3]))
        return out


class DPTDecoder(nn.Module):
    """Dense Prediction Transformer Decoder for multiple geometric targets."""

    def __init__(self, decoder_config: DecoderConfig, student_config: StudentConfig):
        super().__init__()
        self.config = decoder_config
        
        embed_dim = student_config.embed_dim
        features = decoder_config.hidden_dim
        
        # Scratch networks to project from embed_dim to hidden_dim
        # We assume 4 feature maps are passed (from feature_layers)
        self.scratch = Scratch([embed_dim] * 4, features)
        
        # Feature Fusion Blocks
        self.fusion1 = FeatureFusionBlock(features, decoder_config.fusion_type)
        self.fusion2 = FeatureFusionBlock(features, decoder_config.fusion_type)
        self.fusion3 = FeatureFusionBlock(features, decoder_config.fusion_type)
        self.fusion4 = FeatureFusionBlock(features, decoder_config.fusion_type)

        # Output heads
        if decoder_config.predict_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features // 2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)  # Depth is positive
            )
            
        if decoder_config.predict_normals:
            self.normal_head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features // 2, 3, kernel_size=1, stride=1, padding=0)
            )
            
        if decoder_config.predict_point_map:
            self.point_map_head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features // 2, 3, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, features: list[torch.Tensor], img_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Forward pass of the decoder.
        
        Args:
            features: List of 4 feature maps from the backbone.
                Shapes typically: [B, C, H/16, W/16]
            img_shape: (H, W) of the original input image.
            
        Returns:
            Dictionary with predicted targets (depth, normal, point_map) at original resolution.
        """
        assert len(features) == 4, f"Expected 4 feature maps, got {len(features)}"
        
        # Project features
        projected_features = self.scratch(features)
        layer_1, layer_2, layer_3, layer_4 = projected_features
        
        # We process from lowest resolution to highest resolution
        # But wait, ViT typically has same spatial resolution across all layers: H/16, W/16.
        # DPT usually uses a reshape+conv to get hierarchical scales.
        # Here, we will just use convolutions to mimic standard FPN-like scaling.
        # Since all features are H/16, W/16, we'll progressively upsample.
        
        # Base representation from last layer
        x = layer_4
        
        # Fuse and upsample
        x = self.fusion4(x, layer_3)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) # H/8
        
        x = self.fusion3(x, layer_2)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) # H/4
        
        x = self.fusion2(x, layer_1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False) # H/2
        
        # Final upsample to H, W happens via interpolation later or in the head
        
        outputs = {}
        
        if self.config.predict_depth:
            depth = self.depth_head(x)
            depth = F.interpolate(depth, size=img_shape, mode="bilinear", align_corners=False)
            outputs["depth"] = depth
            
        if self.config.predict_normals:
            normal = self.normal_head(x)
            normal = F.interpolate(normal, size=img_shape, mode="bilinear", align_corners=False)
            normal = F.normalize(normal, p=2, dim=1)  # L2 normalize
            outputs["normal"] = normal
            
        if self.config.predict_point_map:
            point_map = self.point_map_head(x)
            point_map = F.interpolate(point_map, size=img_shape, mode="bilinear", align_corners=False)
            outputs["point_map"] = point_map
            
        return outputs
