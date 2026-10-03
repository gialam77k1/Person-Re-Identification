from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3, normalize_embeddings: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        distances = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos

        eye = torch.eye(mask_pos.size(0), dtype=torch.bool, device=mask_pos.device)
        mask_pos = mask_pos & ~eye

        hardest_pos = distances.masked_fill(~mask_pos, float("-inf")).max(dim=1).values
        hardest_neg = distances.masked_fill(~mask_neg, float("inf")).min(dim=1).values

        valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
        if not valid.any():
            return embeddings.new_tensor(0.0)

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(embeddings.device)
        labels = labels.to(embeddings.device)
        centers_batch = centers[labels]
        return ((embeddings - centers_batch) ** 2).sum(dim=1).mean()


class ReIDLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        ce_weight: float = 1.0,
        triplet_weight: float = 1.0,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.0,
        center_loss_weight: float = 0.0,
        auxiliary_loss_weight: float = 0.0,
        normalize_triplet_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.center_loss_weight = center_loss_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = BatchHardTripletLoss(
            margin=triplet_margin,
            normalize_embeddings=normalize_triplet_embeddings,
        )
        self.center = CenterLoss(num_classes=num_classes, feat_dim=embedding_dim) if center_loss_weight > 0 else None

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metric_embeddings: torch.Tensor | None = None,
        auxiliary_logits: list[torch.Tensor] | None = None,
        auxiliary_embeddings: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        metric_embeddings = embeddings if metric_embeddings is None else metric_embeddings
        ce_loss = self.ce(logits, labels)
        triplet_loss = self.triplet(metric_embeddings, labels)
        center_loss = embeddings.new_tensor(0.0)
        auxiliary_loss = embeddings.new_tensor(0.0)
        total_loss = (self.ce_weight * ce_loss) + (self.triplet_weight * triplet_loss)

        auxiliary_logits = auxiliary_logits or []
        auxiliary_embeddings = auxiliary_embeddings or []
        if self.auxiliary_loss_weight > 0 and auxiliary_logits and auxiliary_embeddings:
            if len(auxiliary_logits) != len(auxiliary_embeddings):
                raise ValueError("Auxiliary logits and embeddings must have the same length")
            auxiliary_ce = torch.stack(
                [self.ce(branch_logits, labels) for branch_logits in auxiliary_logits]
            ).mean()
            auxiliary_triplet = torch.stack(
                [self.triplet(branch_embedding, labels) for branch_embedding in auxiliary_embeddings]
            ).mean()
            auxiliary_loss = (self.ce_weight * auxiliary_ce) + (
                self.triplet_weight * auxiliary_triplet
            )
            total_loss = total_loss + (self.auxiliary_loss_weight * auxiliary_loss)

        if self.center is not None:
            center_loss = self.center(metric_embeddings, labels)
            total_loss = total_loss + (self.center_loss_weight * center_loss)
        return total_loss, ce_loss, triplet_loss, center_loss, auxiliary_loss
