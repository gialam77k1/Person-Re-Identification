from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def import_checkpoint(
    *,
    checkpoint_path: Path,
    training_metrics_path: Path,
    mlflow_run_path: Path,
    transfer_report_path: Path,
    effective_config_path: Path,
    data_manifest_path: Path,
    output_root: Path,
) -> dict:
    for path in (
        checkpoint_path,
        training_metrics_path,
        mlflow_run_path,
        transfer_report_path,
        effective_config_path,
        data_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    training = read_json(training_metrics_path)
    mlflow_run = read_json(mlflow_run_path)
    transfer = read_json(transfer_report_path)
    data = read_json(data_manifest_path)
    release_id = "market1501-duke-vit-bnneck:v2"
    if training.get("release_id") != release_id:
        raise ValueError("Training metrics do not belong to the v2 release")
    if mlflow_run.get("status") != "FINISHED" or not mlflow_run.get("run_id"):
        raise ValueError("Training MLflow run must be FINISHED and have a run_id")
    if int(transfer.get("source_epoch", -1)) != 48:
        raise ValueError("v2 checkpoint must originate from model v1 epoch 48")
    if transfer.get("reset_classifier_heads") is not True:
        raise ValueError("v2 training did not reset the Market-1501 classifier")
    if data.get("passed") is not True:
        raise ValueError("Release dataset validation did not pass")

    output_root.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_root / "best_model.pth"
    shutil.copy2(checkpoint_path, output_checkpoint)
    for source, name in (
        (training_metrics_path, "metrics_v2.json"),
        (mlflow_run_path, "mlflow_run.json"),
        (transfer_report_path, "transfer_report.json"),
        (effective_config_path, "effective_config.json"),
    ):
        shutil.copy2(source, output_root / name)

    datasets = data.get("datasets", {})
    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "source_checkpoint": str(checkpoint_path),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": sha256_file(output_checkpoint),
        "checkpoint_size_bytes": output_checkpoint.stat().st_size,
        "best_epoch": training.get("best_epoch"),
        "git_commit": current_commit(),
        "training_git_commit": mlflow_run.get("git_commit"),
        "mlflow_run_id": mlflow_run["run_id"],
        "training_config_sha256": sha256_file(effective_config_path),
        "dataset_revisions": {
            name: spec.get("dataset_dvc_revision") for name, spec in datasets.items()
        },
    }
    (output_root / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--mlflow-run", required=True)
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = import_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        training_metrics_path=Path(args.training_metrics),
        mlflow_run_path=Path(args.mlflow_run),
        transfer_report_path=Path(args.transfer_report),
        effective_config_path=Path(args.effective_config),
        data_manifest_path=Path(args.data_manifest),
        output_root=Path(args.output_root),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
