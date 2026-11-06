from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.validate_msmt17_v2_training import (
    EXPECTED_BASE_RELEASE_ID,
    EXPECTED_DATASET,
    EXPECTED_DATASET_VERSION,
    EXPECTED_EPOCHS,
    EXPECTED_RELEASE_ID,
    validate_training_contract,
)


REQUIRED_FILES = (
    "checkpoints/best_model.pth",
    "checkpoints/last_model.pth",
    "logs/effective_config.json",
    "logs/mlflow_run.json",
    "logs/transfer_report.json",
    "metrics/metrics_v2.json",
)
CUDA_FAILURE_MARKERS = (
    "cuda out of memory",
    "cuda error",
    "outofmemoryerror",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_finite_metrics(value: Any, path: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {path}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_metrics(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_metrics(item, f"{path}.{key}")


def validate_training_completion(run_root: Path, *, train_log: Path | None = None) -> dict:
    missing = [relative for relative in REQUIRED_FILES if not (run_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required v2 training artifacts: {', '.join(missing)}")

    for relative in ("checkpoints/best_model.pth", "checkpoints/last_model.pth"):
        if (run_root / relative).stat().st_size <= 0:
            raise ValueError(f"Checkpoint is empty: {relative}")

    effective_config = _load_json(run_root / "logs/effective_config.json")
    approved_contract = {
        "base_release_id": EXPECTED_BASE_RELEASE_ID,
        "target_dataset": EXPECTED_DATASET,
        "target_dataset_version": EXPECTED_DATASET_VERSION,
        "target_epochs": EXPECTED_EPOCHS,
    }
    validate_training_contract(effective_config, approved_contract)

    mlflow_run = _load_json(run_root / "logs/mlflow_run.json")
    if mlflow_run.get("status") != "FINISHED":
        raise ValueError(f"MLflow run is not FINISHED: {mlflow_run.get('status')!r}")
    if not mlflow_run.get("run_id"):
        raise ValueError("MLflow run metadata is missing run_id")

    transfer = _load_json(run_root / "logs/transfer_report.json")
    if transfer.get("enabled") is not True:
        raise ValueError("Transfer learning was not enabled")
    if int(transfer.get("source_epoch", -1)) != 48:
        raise ValueError("Transfer checkpoint must be model v1 epoch 48")
    if transfer.get("reset_classifier_heads") is not True:
        raise ValueError("Market-1501 classifier heads were not reset")
    if int(transfer.get("loaded_backbone_tensors", 0)) <= 0:
        raise ValueError("No compatible v1 backbone tensors were loaded")

    metrics = _load_json(run_root / "metrics/metrics_v2.json")
    _require_finite_metrics(metrics)
    if metrics.get("release_id") != EXPECTED_RELEASE_ID:
        raise ValueError("Training metrics contain the wrong release_id")
    if metrics.get("base_release_id") != EXPECTED_BASE_RELEASE_ID:
        raise ValueError("Training metrics contain the wrong base_release_id")

    configured_epochs = int(metrics.get("configured_epochs", 0))
    completed_epochs = int(metrics.get("completed_epochs", 0))
    best_epoch = int(metrics.get("best_epoch", 0))
    stopped_early = metrics.get("stopped_early") is True
    stop_reason = str(metrics.get("stop_reason", ""))
    history = metrics.get("history")
    if configured_epochs != EXPECTED_EPOCHS:
        raise ValueError(f"Expected {EXPECTED_EPOCHS} configured epochs")
    if not 1 <= best_epoch <= completed_epochs <= configured_epochs:
        raise ValueError("Invalid best/completed/configured epoch relationship")
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise ValueError("Training history length does not match completed_epochs")
    if stopped_early:
        if completed_epochs >= configured_epochs or not stop_reason.startswith("early_stopping:"):
            raise ValueError("Early-stopped run is missing a valid early stopping reason")
    elif completed_epochs != configured_epochs or stop_reason != "max_epochs_completed":
        raise ValueError("Non-early-stopped run must complete all configured epochs")

    resolved_log = train_log or run_root / "logs/train.log"
    if resolved_log.is_file():
        log_text = resolved_log.read_text(encoding="utf-8", errors="replace").lower()
        marker = next((item for item in CUDA_FAILURE_MARKERS if item in log_text), None)
        if marker:
            raise ValueError(f"Training log contains CUDA failure marker: {marker}")

    return {
        "status": "PASS",
        "release_id": EXPECTED_RELEASE_ID,
        "mlflow_run_id": mlflow_run["run_id"],
        "mlflow_status": mlflow_run["status"],
        "configured_epochs": configured_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "best_rank1": metrics.get("best_rank1"),
        "best_mAP": metrics.get("best_mAP"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "required_artifacts": list(REQUIRED_FILES),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_training_completion(
        Path(args.run_root),
        train_log=Path(args.train_log) if args.train_log else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
