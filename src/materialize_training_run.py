from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable


ARTIFACT_PATHS = {
    "checkpoints/best_model.pth": "checkpoints/best_model.pth",
    "metrics/metrics_v2.json": "metrics/metrics_v2.json",
    "config/effective_config.json": "logs/effective_config.json",
    "provenance/mlflow_run.json": "logs/mlflow_run.json",
    "provenance/transfer_report.json": "logs/transfer_report.json",
}
LAST_CHECKPOINT_ARTIFACT = "checkpoints/last_model.pth"


def materialize_training_run(
    *,
    run_id: str,
    release_id: str,
    run_root: Path,
    tracking_uri: str,
    client: Any | None = None,
    downloader: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    mlflow_module = None
    if client is None or downloader is None:
        import mlflow as mlflow_module
        from mlflow import MlflowClient

        mlflow_module.set_tracking_uri(tracking_uri)
    active_client = client or MlflowClient()
    run = active_client.get_run(run_id)
    if str(run.info.status).upper() != "FINISHED":
        raise ValueError(
            f"MLflow training run {run_id!r} is {run.info.status!r}, not FINISHED."
        )
    recorded_release = str(
        run.data.params.get("release_id") or run.data.tags.get("release.id") or ""
    )
    if recorded_release != release_id:
        raise ValueError(
            f"MLflow training run release mismatch: {recorded_release!r} != {release_id!r}."
        )
    download = downloader or (
        lambda uri: mlflow_module.artifacts.download_artifacts(artifact_uri=uri)
    )
    restored: dict[str, str] = {}
    for artifact_path, relative_destination in ARTIFACT_PATHS.items():
        downloaded = Path(download(f"runs:/{run_id}/{artifact_path}"))
        if not downloaded.is_file() or downloaded.stat().st_size <= 0:
            raise ValueError(f"MLflow artifact is missing or empty: {artifact_path}")
        destination = run_root / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
        restored[artifact_path] = str(destination)
    last_destination = run_root / LAST_CHECKPOINT_ARTIFACT
    last_destination.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_source = LAST_CHECKPOINT_ARTIFACT
    try:
        downloaded_last = Path(download(f"runs:/{run_id}/{LAST_CHECKPOINT_ARTIFACT}"))
        if not downloaded_last.is_file() or downloaded_last.stat().st_size <= 0:
            raise ValueError("downloaded last checkpoint is empty")
        shutil.copy2(downloaded_last, last_destination)
    except Exception:
        # Older completed runs only persisted the best checkpoint in MLflow.
        # They can still resume post-training safely because no optimizer step
        # is performed; preserve the required release layout explicitly.
        shutil.copy2(run_root / "checkpoints" / "best_model.pth", last_destination)
        last_checkpoint_source = "checkpoints/best_model.pth (legacy recovery fallback)"
    restored[LAST_CHECKPOINT_ARTIFACT] = str(last_destination)
    report = {
        "status": "PASS",
        "mode": "resume_post_training",
        "run_id": run_id,
        "mlflow_run_id": run_id,
        "release_id": release_id,
        "last_checkpoint_source": last_checkpoint_source,
        "restored": restored,
    }
    output = run_root / "logs" / "training_restore_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a completed candidate training run without retraining."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_training_run(
                run_id=args.run_id,
                release_id=args.release_id,
                run_root=Path(args.run_root),
                tracking_uri=args.tracking_uri,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
