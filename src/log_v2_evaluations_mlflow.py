from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("rank1", "rank5", "mAP", "mINP")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bundle(
    msmt17: dict[str, Any], market1501: dict[str, Any], gate: dict[str, Any], run: dict[str, Any]
) -> tuple[str, str]:
    release_ids = {msmt17.get("release_id"), market1501.get("release_id"), gate.get("release_id")}
    if len(release_ids) != 1 or None in release_ids:
        raise ValueError("Evaluation bundle contains mismatched release IDs")
    if not run.get("run_id") or not run.get("tracking_uri"):
        raise ValueError("Training MLflow metadata is incomplete")
    if run.get("status") != "FINISHED":
        raise ValueError("Training MLflow run is not FINISHED")
    if gate.get("mlflow_run_id") != run.get("run_id"):
        raise ValueError("Forgetting gate does not point to the training MLflow run")
    return str(run["run_id"]), str(release_ids.pop())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmt17", required=True)
    parser.add_argument("--market1501", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--mlflow-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "msmt17": Path(args.msmt17),
        "market1501": Path(args.market1501),
        "gate": Path(args.gate),
        "policy": Path(args.policy),
    }
    msmt17 = read_json(paths["msmt17"])
    market1501 = read_json(paths["market1501"])
    gate = read_json(paths["gate"])
    run = read_json(Path(args.mlflow_run))
    run_id, release_id = validate_bundle(msmt17, market1501, gate, run)

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(run["tracking_uri"])
    client = MlflowClient()
    client.get_run(run_id)
    for dataset, report in (("msmt17", msmt17), ("market1501", market1501)):
        for metric in METRICS:
            client.log_metric(run_id, f"evaluation.{dataset}.{metric}", float(report[metric]))
        client.log_metric(run_id, f"evaluation.{dataset}.query_count", float(report["query_count"]))
        client.log_metric(run_id, f"evaluation.{dataset}.gallery_count", float(report["gallery_count"]))
        client.log_artifact(run_id, str(paths[dataset]), artifact_path=f"evaluation/{dataset}")
    client.set_tag(run_id, "evaluation.release_id", release_id)
    client.set_tag(run_id, "evaluation.catastrophic_forgetting", gate["status"])
    client.log_metric(run_id, "evaluation.catastrophic_forgetting_passed", 1.0 if gate.get("passed") else 0.0)
    client.log_artifact(run_id, str(paths["gate"]), artifact_path="evaluation/gate")
    client.log_artifact(run_id, str(paths["policy"]), artifact_path="evaluation/gate")

    result = {
        "status": "PASS",
        "run_id": run_id,
        "release_id": release_id,
        "tracking_uri": run["tracking_uri"],
        "artifacts": [str(path) for path in paths.values()],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
