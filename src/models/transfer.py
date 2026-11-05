from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from src.common.utils import resolve_path

if TYPE_CHECKING:
    import torch


TASK_HEAD_PREFIXES = ("classifier.", "auxiliary_classifiers.")


def is_task_head_parameter(name: str) -> bool:
    return name.startswith(TASK_HEAD_PREFIXES)


def load_transfer_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    reset_classifier_heads: bool = True,
    expected_epoch: int | None = None,
) -> dict[str, Any]:
    import torch

    resolved_checkpoint = resolve_path(checkpoint_path)
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Transfer checkpoint not found: {resolved_checkpoint}")

    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("Transfer checkpoint is missing model_state_dict")

    source_epoch = checkpoint.get("epoch")
    if expected_epoch is not None and int(source_epoch) != int(expected_epoch):
        raise ValueError(
            f"Expected transfer checkpoint epoch {expected_epoch}, received {source_epoch}"
        )

    target_state = model.state_dict()
    transferable: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name, value in source_state.items():
        if reset_classifier_heads and is_task_head_parameter(name):
            skipped[name] = "task_head_reset"
            continue
        if name not in target_state:
            skipped[name] = "not_in_target_model"
            continue
        if tuple(value.shape) != tuple(target_state[name].shape):
            skipped[name] = (
                f"shape_mismatch:{tuple(value.shape)}->{tuple(target_state[name].shape)}"
            )
            continue
        transferable[name] = value

    backbone_keys = [name for name in transferable if name.startswith("backbone.")]
    if not backbone_keys:
        raise ValueError("Transfer checkpoint did not provide any compatible backbone weights")

    incompatible = model.load_state_dict(transferable, strict=False)
    return {
        "checkpoint": str(resolved_checkpoint),
        "source_epoch": int(source_epoch) if source_epoch is not None else None,
        "loaded_parameter_tensors": len(transferable),
        "loaded_backbone_tensors": len(backbone_keys),
        "reset_classifier_heads": bool(reset_classifier_heads),
        "skipped": skipped,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def set_vit_lower_blocks_trainable(
    model: torch.nn.Module,
    *,
    frozen_lower_blocks: int,
) -> dict[str, int]:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("Model does not expose a 'backbone' module")

    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise AttributeError("ViT backbone does not expose encoder.layers")

    total_blocks = len(layers)
    if frozen_lower_blocks < 0 or frozen_lower_blocks > total_blocks:
        raise ValueError(
            f"frozen_lower_blocks must be between 0 and {total_blocks}, "
            f"received {frozen_lower_blocks}"
        )

    for parameter in backbone.parameters():
        parameter.requires_grad = True

    patch_projection = getattr(backbone, "conv_proj", None)
    if patch_projection is not None:
        for parameter in patch_projection.parameters():
            parameter.requires_grad = False

    class_token = getattr(backbone, "class_token", None)
    if class_token is not None:
        class_token.requires_grad = False

    position_embedding = getattr(encoder, "pos_embedding", None)
    if position_embedding is not None:
        position_embedding.requires_grad = False

    for index, block in enumerate(layers):
        trainable = index >= frozen_lower_blocks
        for parameter in block.parameters():
            parameter.requires_grad = trainable

    return {
        "total_blocks": total_blocks,
        "frozen_lower_blocks": frozen_lower_blocks,
        "trainable_upper_blocks": total_blocks - frozen_lower_blocks,
    }
