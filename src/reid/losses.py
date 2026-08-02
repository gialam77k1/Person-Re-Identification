from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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


class ReIDLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        triplet_weight: float = 1.0,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=triplet_margin)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self.ce(logits, labels)
        triplet_loss = self.triplet(embeddings, labels)
        total_loss = (self.ce_weight * ce_loss) + (self.triplet_weight * triplet_loss)
        return total_loss, ce_loss, triplet_loss
