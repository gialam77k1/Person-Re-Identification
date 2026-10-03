from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
from src.data.dataset import build_dataset, build_transforms
from src.models.reid_model import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dadnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--split", choices=("train", "query", "gallery"), default="query")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.dataset.name=market1501",
    )
    return parser.parse_args()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")

    config = load_runtime_config(args.config, args.set, command_name="verify-onnx")
    configure_torch_home()
    ensure_dir(config["artifacts"]["logs_dir"])
    ensure_dir(config["artifacts"]["metrics_dir"])

    log_path = Path(config["artifacts"]["logs_dir"]) / "verify_onnx.log"
    with tee_output(log_path):
        run_verification(config, args)


def run_verification(config: dict, args: argparse.Namespace) -> None:
    checkpoint_path = resolve_path(args.checkpoint)
    onnx_path = resolve_path(args.onnx_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    device = infer_device(config["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise KeyError("Checkpoint is missing classifier.weight, cannot infer num_classes.")

    model = build_model_from_config(
        config,
        num_classes=int(classifier_weight.shape[0]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_transform = build_transforms(
        config["data"]["image_height"],
        config["data"]["image_width"],
        preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
    )
    dataset = build_dataset(config, args.split, transform=test_transform, relabel=False)
    selected_count = min(args.num_samples, len(dataset))
    if selected_count == 0:
        raise RuntimeError(f"Dataset split '{args.split}' contains no samples.")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    sample_results = []
    all_pytorch_outputs = []
    all_onnx_outputs = []
    for index in range(selected_count):
        sample = dataset[index]
        inputs = sample["image"].unsqueeze(0)
        with torch.inference_mode():
            _, pytorch_embedding = model(inputs.to(device))
        pytorch_output = pytorch_embedding.detach().cpu().numpy().astype(np.float32)[0]
        onnx_output = session.run(
            [output_name],
            {input_name: inputs.numpy().astype(np.float32, copy=False)},
        )[0][0].astype(np.float32, copy=False)

        absolute_error = np.abs(pytorch_output - onnx_output)
        sample_results.append(
            {
                "index": index,
                "path": str(Path(sample["path"]).resolve()),
                "max_abs_error": float(absolute_error.max()),
                "mean_abs_error": float(absolute_error.mean()),
                "cosine_similarity": cosine_similarity(pytorch_output, onnx_output),
            }
        )
        all_pytorch_outputs.append(pytorch_output)
        all_onnx_outputs.append(onnx_output)

    pytorch_outputs = np.stack(all_pytorch_outputs)
    onnx_outputs = np.stack(all_onnx_outputs)
    absolute_error = np.abs(pytorch_outputs - onnx_outputs)
    passed = bool(np.allclose(pytorch_outputs, onnx_outputs, atol=args.atol, rtol=args.rtol))

    results = {
        "run_slug": config["runtime"]["run_slug"],
        "dataset": config["data"]["dataset"]["name"],
        "split": args.split,
        "num_samples": selected_count,
        "checkpoint": str(checkpoint_path),
        "loaded_epoch": checkpoint.get("epoch"),
        "onnx_path": str(onnx_path),
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
        "input_name": input_name,
        "input_shape": session.get_inputs()[0].shape,
        "output_name": output_name,
        "output_shape": session.get_outputs()[0].shape,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in sample_results),
        "passed": passed,
        "samples": sample_results,
    }
    output_path = Path(config["artifacts"]["metrics_dir"]) / "onnx_parity.json"
    save_json(results, output_path)
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")
    print(results)

    if not passed:
        raise RuntimeError(
            "PyTorch-ONNX parity check failed: "
            f"max_abs_error={results['max_abs_error']:.6g}, atol={args.atol}, rtol={args.rtol}"
        )


if __name__ == "__main__":
    main()
