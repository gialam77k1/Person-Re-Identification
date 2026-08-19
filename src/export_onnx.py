from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.utils import (
    configure_torch_home,
    ensure_dir,
    infer_device,
    resolve_path,
    save_json,
    tee_output,
)
from src.models.reid_model import build_model_from_config


class EmbeddingExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, embedding = self.model(inputs)
        return embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dadnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.image_height=256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config, args.set, command_name="export-onnx")
    configure_torch_home()
    ensure_dir(config["artifacts"]["logs_dir"])
    ensure_dir(config["artifacts"]["exports_dir"])

    log_path = Path(config["artifacts"]["logs_dir"]) / "export_onnx.log"
    with tee_output(log_path):
        run_export_command(config, args.checkpoint, args.output, args.opset)


def run_export_command(config: dict, checkpoint_path: str, output_path: str, opset: int) -> None:
    export_dir = Path(config["artifacts"]["exports_dir"])
    export_path = Path(output_path) if output_path else export_dir / "model_embedding.onnx"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logging console output to {Path(config['artifacts']['logs_dir']) / 'export_onnx.log'}")
    print(f"Exporting ONNX model to {export_path}")

    device = infer_device(config["device"])
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"]
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise KeyError("Checkpoint is missing classifier.weight, cannot infer num_classes for export.")

    model = build_model_from_config(
        config,
        num_classes=int(classifier_weight.shape[0]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = EmbeddingExportWrapper(model).to(device)
    wrapper.eval()

    height = int(config["data"]["image_height"])
    width = int(config["data"]["image_width"])
    dummy_input = torch.randn(1, 3, height, width, device=device)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
        )

    metadata = {
        "run_slug": config["runtime"]["run_slug"],
        "run_root": config["runtime"]["run_root"],
        "dataset": config["data"]["dataset"]["name"],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "onnx_path": str(export_path.resolve()),
        "opset": opset,
        "input_shape": [1, 3, height, width],
        "output_name": "embeddings",
        "embedding_dim": int(config["model"]["embedding_dim"]),
        "loaded_epoch": checkpoint.get("epoch"),
        "model_variant": config["model"].get("variant", "baseline"),
    }
    save_json(metadata, export_dir / "onnx_export_manifest.json")
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")
    print(metadata)


if __name__ == "__main__":
    main()
