"""Multi-Target Distillation Losses for GeoDistill."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geodistill.config import LossConfig


class DepthLoss(nn.Module):
    """Scale-Invariant MSE Loss for Depth Distillation."""

    def __init__(self, loss_type: str = "si_mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
            
        pred = pred[mask]
        target = target[mask]
        
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        if self.loss_type == "si_mse":
            log_pred = torch.log(pred + 1e-6)
            log_target = torch.log(target + 1e-6)
            diff = log_pred - log_target
            
            # Scale-Invariant MSE
            loss = torch.mean(diff ** 2) - 0.5 * (torch.mean(diff) ** 2)
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unsupported depth loss type: {self.loss_type}")
            
        return loss


class NormalLoss(nn.Module):
    """Loss for Surface Normal Distillation."""

    def __init__(self, loss_type: str = "cosine"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred: [B, 3, H, W]
        target: [B, 3, H, W]
        """
        if mask is None:
            mask = torch.ones_like(target[:, 0:1, :, :], dtype=torch.bool)
            
        # Broadcast mask
        mask = mask.expand_as(pred)
        
        pred_valid = pred[mask].view(-1, 3)
        target_valid = target[mask].view(-1, 3)
        
        if pred_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        if self.loss_type == "cosine":
            # Cosine distance loss (1 - cosine_similarity)
            loss = 1.0 - F.cosine_similarity(pred_valid, target_valid, dim=1).mean()
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_valid, target_valid)
        else:
            raise ValueError(f"Unsupported normal loss type: {self.loss_type}")
            
        return loss


class PointMapLoss(nn.Module):
    """Loss for Point Map Distillation."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(target[:, 0:1, :, :], dtype=torch.bool)
            
        mask = mask.expand_as(pred)
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        if pred_valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        return F.l1_loss(pred_valid, target_valid)


class FeatureKLLoss(nn.Module):
    """Feature-level KL Divergence Loss for dense feature distillation."""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features: list[torch.Tensor], teacher_features: list[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # s_feat, t_feat: [B, C, H, W]
            B, C, H, W = s_feat.shape
            
            s_flat = s_feat.view(B, C, -1) / self.temperature
            t_flat = t_feat.view(B, C, -1) / self.temperature
            
            s_log_prob = F.log_softmax(s_flat, dim=-1)
            t_prob = F.softmax(t_flat, dim=-1)
            
            kl = F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (self.temperature ** 2)
            loss += kl
            
        return loss / len(student_features)


class GeometricConsistencyLoss(nn.Module):
    """Loss enforcing consistency between predicted depth and predicted normals.
    
    Gradient of depth should be orthogonal to the surface normal.
    """

    def __init__(self):
        super().__init__()

    def forward(self, depth: torch.Tensor, normal: torch.Tensor, intrinsics: torch.Tensor = None) -> torch.Tensor:
        """
        depth: [B, 1, H, W]
        normal: [B, 3, H, W]
        """
        # Simplistic geometric consistency: ∇depth · normal_{xy} terms
        
        grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        # Pad to match original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        # Construct approximate surface tangent from depth gradients
        # tangent_x ≈ [1, 0, ∂z/∂x]
        # tangent_y ≈ [0, 1, ∂z/∂y]
        # These should be orthogonal to normal: T · N = 0
        
        # In actual 3D space, this requires camera intrinsics. 
        # A simpler approach without intrinsics is penalizing the dot product of depth gradients and normal x/y components
        # N_x * dx + N_y * dy + N_z ≈ 0 (under certain assumptions)
        
        n_x = normal[:, 0:1, :, :]
        n_y = normal[:, 1:2, :, :]
        n_z = normal[:, 2:3, :, :]
        
        consistency = torch.abs(n_x * grad_x + n_y * grad_y + n_z)
        return consistency.mean()


class EdgeAwareLoss(nn.Module):
    """Edge-aware smoothness loss for depth maps."""

    def __init__(self):
        super().__init__()

    def forward(self, depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        depth: [B, 1, H, W]
        image: [B, 3, H, W]
        """
        depth_grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        depth_grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        img_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        img_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), dim=1, keepdim=True))
        
        smoothness_x = torch.abs(depth_grad_x) * weights_x
        smoothness_y = torch.abs(depth_grad_y) * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()


class GeoDistillLoss(nn.Module):
    """Composite loss function for the GeoDistill pipeline."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        self.depth_loss = DepthLoss(config.depth_loss_type)
        self.normal_loss = NormalLoss(config.normal_loss_type)
        self.point_map_loss = PointMapLoss()
        self.feature_kl_loss = FeatureKLLoss(config.feature_kl_temperature)
        self.geometric_consistency_loss = GeometricConsistencyLoss()
        self.edge_aware_loss = EdgeAwareLoss()
        
        if config.use_uncertainty_weighting:
            # Learnable weights for each task loss
            self.log_vars = nn.Parameter(torch.zeros(6))
        else:
            self.log_vars = None

    def forward(
        self, 
        student_preds: dict[str, torch.Tensor | list[torch.Tensor]], 
        teacher_targets: dict[str, torch.Tensor | list[torch.Tensor]],
        images: torch.Tensor = None,
        masks: dict[str, torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0.0
        
        masks = masks or {}
        
        # Depth Loss
        if "depth" in student_preds and "depth" in teacher_targets:
            mask = masks.get("depth", None)
            l_depth = self.depth_loss(student_preds["depth"], teacher_targets["depth"], mask)
            losses["loss_depth"] = l_depth
            total_loss += self.config.lambda_depth * l_depth
            
        # Normal Loss
        if "normal" in student_preds and "normal" in teacher_targets:
            mask = masks.get("normal", None)
            l_norm = self.normal_loss(student_preds["normal"], teacher_targets["normal"], mask)
            losses["loss_normal"] = l_norm
            total_loss += self.config.lambda_normal * l_norm
            
        # Point Map Loss
        if "point_map" in student_preds and "point_map" in teacher_targets:
            mask = masks.get("point_map", None)
            l_pts = self.point_map_loss(student_preds["point_map"], teacher_targets["point_map"], mask)
            losses["loss_point_map"] = l_pts
            total_loss += self.config.lambda_point_map * l_pts
            
        # Feature KL Loss
        if "features" in student_preds and "features" in teacher_targets:
            l_feat = self.feature_kl_loss(student_preds["features"], teacher_targets["features"])
            losses["loss_feature"] = l_feat
            total_loss += self.config.lambda_feature_kl * l_feat
            
        # Geometric Consistency Loss
        if "depth" in student_preds and "normal" in student_preds:
            l_geo = self.geometric_consistency_loss(student_preds["depth"], student_preds["normal"])
            losses["loss_geo_consistency"] = l_geo
            total_loss += self.config.lambda_geometric_consistency * l_geo
            
        # Edge-Aware Loss
        if "depth" in student_preds and images is not None:
            l_edge = self.edge_aware_loss(student_preds["depth"], images)
            losses["loss_edge_aware"] = l_edge
            total_loss += self.config.lambda_edge_aware * l_edge
            
        losses["loss"] = total_loss
        return losses
