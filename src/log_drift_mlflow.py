from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def drift_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "drift_detected": 1.0 if report.get("drift_detected") is True else 0.0,
        "reference_samples": float(report["reference_samples"]),
        "current_samples": float(report["current_samples"]),
        "embedding_dim": float(report["embedding_dim"]),
    }
    for name, check in report["checks"].items():
        metrics[f"drift_{name}"] = float(check["value"])
        metrics[f"threshold_{name}"] = float(check["maximum"])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Store an already-computed drift report in MLflow.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--prometheus-metrics", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--experiment", default="person-reid-drift")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import mlflow

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    release_id = str(report["release_id"])
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"duke-drift-{release_id.replace(':', '-')}") as run:
        mlflow.set_tags(
            {
                "run.kind": "embedding-drift",
                "release.id": release_id,
                "drift.status": str(report["status"]),
                "dataset": str(provenance.get("dataset", "unknown")),
                "git.commit": str(provenance.get("git_commit", "unknown")),
                "dvc.dataset_revision": str(
                    provenance.get("dataset_dvc_revision", "unknown")
                ),
                "detector": "generate_drift_report.py",
            }
        )
        mlflow.log_metrics(drift_metrics(report))
        mlflow.log_params(
            {
                "sample_seed": provenance.get("sample_seed"),
                "split": provenance.get("split"),
                "minimum_current_samples": report.get("minimum_current_samples"),
                "model_sha256": report.get("model_artifact", {}).get("sha256"),
            }
        )
        mlflow.log_artifact(str(report_path), artifact_path="drift")
        mlflow.log_artifact(args.prometheus_metrics, artifact_path="drift")
        mlflow.log_artifact(args.policy, artifact_path="config")
        result = {
            "schema_version": 1,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment": args.experiment,
            "run_id": run.info.run_id,
            "release_id": release_id,
            "drift_status": report["status"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
