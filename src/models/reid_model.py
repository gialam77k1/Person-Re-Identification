from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            warnings.warn(f"Falling back to random-initialized ResNet50: {exc}")
            backbone = resnet50(weights=None)

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
