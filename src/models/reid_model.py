from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_backbone(pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        backbone = resnet50(weights=weights)
    except Exception as exc:
        warnings.warn(f"Falling back to random-initialized ResNet50: {exc}")
        backbone = resnet50(weights=None)
    return backbone


class ReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        backbone = build_resnet50_backbone(pretrained)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return logits, embedding


class CFTAttentionModule(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(64, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_map = self.channel_attention(features)
        spatial_map = self.spatial_attention(features)
        attended = features * channel_map * spatial_map
        return self.refine(attended) + features


class PositionAwareAttentionModule(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 32) -> None:
        super().__init__()
        reduced_channels = max(32, channels // reduction)
        self.shared_projection = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )
        self.height_gate = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.width_gate = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height_context = features.mean(dim=3, keepdim=True)
        width_context = features.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        combined = torch.cat([height_context, width_context], dim=2)
        projected = self.shared_projection(combined)
        height_tokens, width_tokens = torch.split(
            projected,
            [height_context.size(2), width_context.size(2)],
            dim=2,
        )
        width_tokens = width_tokens.permute(0, 1, 3, 2)

        height_attention = torch.sigmoid(self.height_gate(height_tokens))
        width_attention = torch.sigmoid(self.width_gate(width_tokens))
        return features * height_attention * width_attention + features


class SEBlock(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(32, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(self.squeeze(features))
        return features * scale + features


class DistinguishabilityEnhancementModule(nn.Module):
    def __init__(self, input_dim: int = 2048, embedding_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(embedding_dim * 2, input_dim // 2)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.shortcut = nn.Linear(input_dim, embedding_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(pooled_features)
        shortcut = self.shortcut(pooled_features)
        gate = self.gate(pooled_features)
        enhanced = shortcut + (projected * gate)
        return self.bn(enhanced)


class LocalStripeBranch(nn.Module):
    def __init__(self, channels: int = 2048, embedding_dim: int = 512, num_stripes: int = 3) -> None:
        super().__init__()
        self.num_stripes = max(2, num_stripes)
        self.pool = nn.AdaptiveAvgPool2d((self.num_stripes, 1))
        self.projection = nn.Sequential(
            nn.Linear(channels * self.num_stripes, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.projection(pooled)


class GlobalLocalFusion(nn.Module):
    def __init__(self, embedding_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, global_embedding: torch.Tensor, local_embedding: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([global_embedding, local_embedding], dim=1))


class DADNetReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        attention_reduction: int = 16,
        position_attention_reduction: int = 32,
        use_se_block: bool = False,
        se_reduction: int = 16,
        dem_dropout: float = 0.1,
        use_local_branch: bool = False,
        num_local_stripes: int = 3,
        local_branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        backbone = build_resnet50_backbone(pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.cft = CFTAttentionModule(channels=2048, reduction=attention_reduction)
        self.position_attention = PositionAwareAttentionModule(
            channels=2048,
            reduction=position_attention_reduction,
        )
        self.se_block = SEBlock(channels=2048, reduction=se_reduction) if use_se_block else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dem = DistinguishabilityEnhancementModule(
            input_dim=2048,
            embedding_dim=embedding_dim,
            dropout=dem_dropout,
        )
        self.local_branch = (
            LocalStripeBranch(
                channels=2048,
                embedding_dim=embedding_dim,
                num_stripes=num_local_stripes,
            )
            if use_local_branch
            else None
        )
        self.global_local_fusion = (
            GlobalLocalFusion(embedding_dim=embedding_dim, dropout=local_branch_dropout)
            if use_local_branch
            else nn.Identity()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        attended = self.cft(features)
        attended = self.position_attention(attended)
        attended = self.se_block(attended)
        pooled = torch.flatten(self.pool(attended), 1)
        embedding = self.dem(pooled)
        if self.local_branch is not None:
            local_embedding = self.local_branch(attended)
            embedding = self.global_local_fusion(embedding, local_embedding)
        logits = self.classifier(embedding)
        return logits, embedding


def build_model_from_config(config: dict[str, Any], num_classes: int, pretrained: bool | None = None) -> nn.Module:
    model_config = config["model"]
    variant = model_config.get("variant", "baseline").lower()
    if pretrained is None:
        pretrained = bool(model_config.get("pretrained", True))

    if variant == "baseline":
        return ReIDModel(
            num_classes=num_classes,
            embedding_dim=model_config["embedding_dim"],
            pretrained=pretrained,
        )

    if variant == "dadnet":
        return DADNetReIDModel(
            num_classes=num_classes,
            embedding_dim=model_config["embedding_dim"],
            pretrained=pretrained,
            attention_reduction=model_config.get("attention_reduction", 16),
            position_attention_reduction=model_config.get("position_attention_reduction", 32),
            use_se_block=model_config.get("use_se_block", False),
            se_reduction=model_config.get("se_reduction", 16),
            dem_dropout=model_config.get("dem_dropout", 0.1),
            use_local_branch=model_config.get("use_local_branch", False),
            num_local_stripes=model_config.get("num_local_stripes", 3),
            local_branch_dropout=model_config.get("local_branch_dropout", 0.0),
        )

    raise ValueError(f"Unsupported model variant: {variant}")
