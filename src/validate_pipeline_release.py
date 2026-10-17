from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a reproducible release manifest.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipeline-params", default="params.yaml")
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--evaluation-metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--triton-root", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_spec(role: str, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "repository_path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    params_path = resolve(args.pipeline_params)
    data_manifest_path = resolve(args.data_manifest)
    training_metrics_path = resolve(args.training_metrics)
    evaluation_metrics_path = resolve(args.evaluation_metrics)
    checkpoint_path = resolve(args.checkpoint)
    onnx_path = resolve(args.onnx)
    external_weights_path = onnx_path.with_name(f"{onnx_path.name}.data")
    parity_path = resolve(args.parity)
    reference_manifest_path = resolve(args.reference_manifest)
    triton_root = resolve(args.triton_root)
    preprocessing_source = resolve(args.preprocessing)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    data_manifest = read_json(data_manifest_path)
    training_metrics = read_json(training_metrics_path)
    evaluation_metrics = read_json(evaluation_metrics_path)
    parity = read_json(parity_path)
    reference_manifest = read_json(reference_manifest_path)
    if not data_manifest.get("passed"):
        raise RuntimeError("Dataset validation did not pass.")
    if not parity.get("passed"):
        raise RuntimeError("PyTorch-ONNX parity validation did not pass.")

    reference_files = [
        resolve(reference_manifest[key])
        for key in ("embeddings_path", "pids_path", "camids_path", "paths_path")
    ]
    for reference_file in reference_files:
        if not reference_file.is_file():
            raise FileNotFoundError(reference_file)

    triton_model_root = triton_root / "reid_embedding"
    triton_onnx = triton_model_root / "1" / "model.onnx"
    triton_external = triton_model_root / "1" / external_weights_path.name
    triton_config = triton_model_root / "config.pbtxt"
    if sha256_file(triton_onnx) != sha256_file(onnx_path):
        raise RuntimeError("Packaged Triton ONNX graph differs from export output.")
    if sha256_file(triton_external) != sha256_file(external_weights_path):
        raise RuntimeError("Packaged Triton external weights differ from export output.")
    if not triton_config.is_file():
        raise FileNotFoundError(triton_config)

    preprocessing_target = output_root / "preprocessing.json"
    shutil.copy2(preprocessing_source, preprocessing_target)
    files = [
        file_spec("pytorch_checkpoint", checkpoint_path),
        file_spec("onnx_graph", onnx_path),
        file_spec("onnx_external_weights", external_weights_path),
        file_spec("preprocessing_contract", preprocessing_target),
        file_spec("reference_embeddings", reference_files[0]),
        file_spec("reference_pids", reference_files[1]),
        file_spec("reference_camids", reference_files[2]),
        file_spec("reference_paths", reference_files[3]),
        file_spec("triton_config", triton_config),
    ]
    manifest = {
        "schema_version": 1,
        "release": {
            "release_id": args.release_id,
            "status": "validated",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "source": {
            "git_commit": git_commit(),
            "dataset_dvc_revision": data_manifest["dataset_dvc_revision"],
            "config_path": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "pipeline_params_path": relative(params_path),
            "pipeline_params_sha256": sha256_file(params_path),
            "checkpoint_epoch": training_metrics["best_epoch"],
        },
        "contract": {
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 512],
            "preprocessing_path": relative(preprocessing_target),
        },
        "quality_metrics": {
            key: evaluation_metrics[key]
            for key in ("rank1", "rank5", "mAP", "mINP")
        },
        "files": files,
        "validation": {
            "data": True,
            "pytorch_onnx_parity": parity,
            "reference_num_samples": reference_manifest["num_samples"],
            "reference_feature_dim": reference_manifest["feature_dim"],
            "triton_package": True,
        },
    }
    manifest_path = save_json(manifest, output_root / "model_manifest.json")
    report = {
        "passed": True,
        "release_id": args.release_id,
        "git_commit": manifest["source"]["git_commit"],
        "dataset_dvc_revision": manifest["source"]["dataset_dvc_revision"],
        "config_sha256": manifest["source"]["config_sha256"],
        "manifest_path": relative(manifest_path),
        "validated_files": len(files),
    }
    save_json(report, output_root / "validation_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
