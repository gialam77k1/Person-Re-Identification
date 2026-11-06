from __future__ import annotations

import argparse
import json
from pathlib import Path

def validate_candidate_inputs(report: dict, manifest: dict, alias: str) -> tuple[str, str, str]:
    release_id = manifest.get("release", {}).get("release_id")
    if alias != "candidate":
        raise ValueError("Retraining workflow may only assign the candidate alias.")
    if report.get("status") != "PASS" or report.get("passed") is not True:
        raise ValueError("Only a PASS validation report can be registered as candidate.")
    if report.get("release_id") != release_id:
        raise ValueError("Validation report and release manifest do not match.")
    run_id = manifest.get("source", {}).get("mlflow_run_id")
    if not run_id:
        raise ValueError("Release manifest is missing the MLflow run ID.")
    return str(release_id).partition(":")[0], str(release_id), str(run_id)


def main() -> int:
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser(description="Register a validated retraining candidate in MLflow.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--alias", default="candidate")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    model_name, release_id, run_id = validate_candidate_inputs(report, manifest, args.alias)
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name, tags={"task": "person-re-identification"})
    versions = client.search_model_versions(f"name = '{model_name}'")
    version = next((item for item in versions if item.tags.get("release.id") == release_id), None)
    if version is None:
        version = client.create_model_version(name=model_name, source=f"runs:/{run_id}/checkpoints", run_id=run_id, tags={"release.id": release_id, "validation.status": "PASS"})
    client.set_registered_model_alias(model_name, args.alias, version.version)
    print(json.dumps({"registered_model": model_name, "version": str(version.version), "alias": args.alias, "release_id": release_id, "run_id": run_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
