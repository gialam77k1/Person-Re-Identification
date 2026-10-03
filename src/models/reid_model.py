from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b16_backbone(pretrained: bool) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        backbone = vit_b_16(weights=weights)
    except Exception as exc:
        warnings.warn(f"Falling back to random-initialized ViT-B/16: {exc}")
        backbone = vit_b_16(weights=None)
    backbone.heads = nn.Identity()
    return backbone


class ViTReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.1,
        use_local_branch: bool = True,
        num_local_stripes: int = 4,
        local_branch_dropout: float = 0.1,
        use_bnneck: bool = True,
        use_auxiliary_branch_loss: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = build_vit_b16_backbone(pretrained)
        vit_feature_dim = 768
        self.use_local_branch = use_local_branch
        self.num_local_stripes = max(2, num_local_stripes)
        self.use_bnneck = use_bnneck
        self.use_auxiliary_branch_loss = use_auxiliary_branch_loss

        self.cls_embedding = nn.Sequential(
            nn.LayerNorm(vit_feature_dim),
            nn.Linear(vit_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(vit_feature_dim),
            nn.Linear(vit_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_local_branch:
            self.local_embedding = nn.Sequential(
                nn.LayerNorm(vit_feature_dim * self.num_local_stripes),
                nn.Linear(vit_feature_dim * self.num_local_stripes, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
                nn.Dropout(local_branch_dropout),
            )
            fusion_input_dim = embedding_dim * 3
        else:
            self.local_embedding = None
            fusion_input_dim = embedding_dim * 2

        if self.use_bnneck:
            self.fusion_projection = nn.Linear(fusion_input_dim, embedding_dim)
            self.bottleneck = nn.BatchNorm1d(embedding_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.fusion = None
        else:
            self.fusion_projection = None
            self.bottleneck = None
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=not self.use_bnneck)
        if self.use_bnneck:
            nn.init.normal_(self.classifier.weight, std=0.001)

        num_branches = 3 if self.use_local_branch else 2
        self.auxiliary_classifiers = (
            nn.ModuleList(
                [nn.Linear(embedding_dim, num_classes, bias=False) for _ in range(num_branches)]
            )
            if self.use_auxiliary_branch_loss
            else None
        )
        if self.auxiliary_classifiers is not None:
            for auxiliary_classifier in self.auxiliary_classifiers:
                nn.init.normal_(auxiliary_classifier.weight, std=0.001)

    def _forward_tokens(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone._process_input(inputs)
        batch_size = tokens.shape[0]
        class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        encoded = self.backbone.encoder(tokens)
        return encoded[:, 0], encoded[:, 1:]

    def _build_embeddings(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        cls_token, patch_tokens = self._forward_tokens(inputs)
        cls_embedding = self.cls_embedding(cls_token)
        patch_embedding = self.patch_embedding(patch_tokens.mean(dim=1))

        embeddings = [cls_embedding, patch_embedding]
        if self.use_local_branch and self.local_embedding is not None:
            grid_size = int(patch_tokens.size(1) ** 0.5)
            patch_grid = patch_tokens.transpose(1, 2).reshape(
                patch_tokens.size(0),
                patch_tokens.size(2),
                grid_size,
                grid_size,
            )
            stripe_features = torch.flatten(
                nn.functional.adaptive_avg_pool2d(patch_grid, (self.num_local_stripes, 1)),
                1,
            )
            local_embedding = self.local_embedding(stripe_features)
            embeddings.append(local_embedding)

        concatenated = torch.cat(embeddings, dim=1)
        if self.use_bnneck:
            if self.fusion_projection is None or self.bottleneck is None:
                raise RuntimeError("BNNeck modules were not initialized")
            metric_embedding = self.fusion_projection(concatenated)
            inference_embedding = self.bottleneck(metric_embedding)
        else:
            if self.fusion is None:
                raise RuntimeError("Fusion module was not initialized")
            inference_embedding = self.fusion(concatenated)
            metric_embedding = inference_embedding
        return inference_embedding, metric_embedding, embeddings

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inference_embedding, _, _ = self._build_embeddings(inputs)
        logits = self.classifier(inference_embedding)
        return logits, inference_embedding

    def forward_training(self, inputs: torch.Tensor) -> dict[str, Any]:
        inference_embedding, metric_embedding, branch_embeddings = self._build_embeddings(inputs)
        logits = self.classifier(inference_embedding)
        auxiliary_logits = []
        if self.auxiliary_classifiers is not None:
            auxiliary_logits = [
                classifier(branch_embedding)
                for classifier, branch_embedding in zip(self.auxiliary_classifiers, branch_embeddings)
            ]
        return {
            "logits": logits,
            "embedding": inference_embedding,
            "metric_embedding": metric_embedding,
            "auxiliary_logits": auxiliary_logits,
            "auxiliary_embeddings": branch_embeddings if auxiliary_logits else [],
        }


def build_model_from_config(
    config: dict[str, Any],
    num_classes: int,
    pretrained: bool | None = None,
) -> nn.Module:
    model_config = config["model"]
    variant = str(model_config.get("variant", "vit")).lower()
    backbone = str(model_config.get("backbone", "vit_b_16")).lower()
    if variant != "vit" or backbone != "vit_b_16":
        raise ValueError(
            "This repository only supports the production ViT-B/16 ReID model "
            f"(received variant={variant!r}, backbone={backbone!r})."
        )
    if pretrained is None:
        pretrained = bool(model_config.get("pretrained", True))

    return ViTReIDModel(
        num_classes=num_classes,
        embedding_dim=model_config["embedding_dim"],
        pretrained=pretrained,
        dropout=model_config.get("dropout", 0.1),
        use_local_branch=model_config.get("use_local_branch", True),
        num_local_stripes=model_config.get("num_local_stripes", 4),
        local_branch_dropout=model_config.get("local_branch_dropout", 0.1),
        use_bnneck=model_config.get("use_bnneck", True),
        use_auxiliary_branch_loss=model_config.get("use_auxiliary_branch_loss", True),
    )
