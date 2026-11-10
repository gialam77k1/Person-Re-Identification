from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_inputs(report: dict, manifest: dict, alias: str) -> tuple[str, str, str]:
    release_id = manifest.get("release", {}).get("release_id")
    if alias != "candidate":
        raise ValueError("Retraining workflow may only assign the candidate alias.")
    if report.get("status") != "PASS" or report.get("passed") is not True:
        raise ValueError("Only a PASS validation report can be registered as candidate.")
    if manifest.get("release", {}).get("status") != "validated" or manifest.get("validation", {}).get("status") != "PASS":
        raise ValueError("The v2 release manifest has not passed validation.")
    if report.get("release_id") != release_id:
        raise ValueError("Validation report and release manifest do not match.")
    run_id = manifest.get("source", {}).get("mlflow_run_id")
    if not release_id or not run_id:
        raise ValueError("Release manifest is missing the release ID or MLflow run ID.")
    return str(release_id).partition(":")[0], str(release_id), str(run_id)


def build_lineage_tags(
    manifest: dict[str, Any],
    validation_path: Path,
    manifest_path: Path,
    publication: dict[str, Any],
) -> dict[str, str]:
    source = manifest.get("source", {})
    revisions = source.get("dataset_revisions", {})
    required = {
        "git.commit": source.get("git_commit"),
        "training.git.commit": source.get("training_git_commit"),
        "dvc.duke.revision": revisions.get("duke"),
        "dvc.market1501.revision": revisions.get("market1501"),
        "config.sha256": source.get("config_sha256"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Release lineage is incomplete: {', '.join(missing)}")
    if len(str(source.get("config_sha256"))) != 64:
        raise ValueError("Release config SHA256 is invalid.")
    if publication.get("status") != "PASS" or publication.get("release_id") != manifest.get("release", {}).get("release_id"):
        raise ValueError("DVC publication report must PASS for the same release.")
    if publication.get("model_manifest", {}).get("sha256") != sha256(manifest_path):
        raise ValueError("Published model manifest checksum does not match.")
    if publication.get("validation_report", {}).get("sha256") != sha256(validation_path):
        raise ValueError("Published validation report checksum does not match.")

    metric_tags: dict[str, str] = {}
    for dataset in ("duke", "market1501"):
        values = manifest.get("quality_metrics", {}).get(dataset, {})
        for metric in ("rank1", "rank5", "mAP", "mINP"):
            value = values.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {dataset}.{metric} metric in release manifest.")
            metric_tags[f"metrics.{dataset}.{metric}"] = str(float(value))

    checkpoint = next((item for item in manifest.get("files", []) if item.get("role") == "pytorch_checkpoint"), {})
    if len(str(checkpoint.get("sha256", ""))) != 64:
        raise ValueError("Release manifest is missing the checkpoint checksum.")
    return {
        **{name: str(value) for name, value in required.items()},
        **metric_tags,
        "checkpoint.sha256": str(checkpoint["sha256"]),
        "manifest.sha256": sha256(manifest_path),
        "validation.sha256": sha256(validation_path),
        "dvc.remote": str(publication.get("dvc", {}).get("remote")),
        "dvc.stage": str(publication.get("dvc", {}).get("stage")),
        "validation.status": "PASS",
        "registry.alias.policy": "candidate-only",
    }


def main() -> int:
    import mlflow
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser(description="Register a validated and published retraining candidate in MLflow.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--dvc-publication", required=True)
    parser.add_argument("--mlflow-run", required=True)
    parser.add_argument("--alias", default="candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    validation_path = Path(args.validation_report)
    publication_path = Path(args.dvc_publication)
    manifest = read_json(manifest_path)
    report = read_json(validation_path)
    publication = read_json(publication_path)
    run_metadata = read_json(Path(args.mlflow_run))
    model_name, release_id, run_id = validate_candidate_inputs(report, manifest, args.alias)
    if run_metadata.get("run_id") != run_id or run_metadata.get("status") != "FINISHED":
        raise ValueError("MLflow training run must be FINISHED and match the release manifest.")
    tracking_uri = run_metadata.get("tracking_uri")
    if not tracking_uri:
        raise ValueError("MLflow run metadata is missing tracking_uri.")
    tags = {
        "release.id": release_id,
        **build_lineage_tags(manifest, validation_path, manifest_path, publication),
        "dvc.publication.sha256": sha256(publication_path),
    }

    mlflow.set_tracking_uri(str(tracking_uri))
    client = MlflowClient()
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run {run_id} is not FINISHED: {run.info.status}")
    for path in (manifest_path, validation_path, publication_path):
        client.log_artifact(run_id, str(path), artifact_path="registry/release")
        # Retain the original location for already-deployed v2 consumers.
        client.log_artifact(run_id, str(path), artifact_path="registry/v2")
    client.set_tag(run_id, "registry.candidate.release_id", release_id)
    client.set_tag(run_id, "registry.candidate.validation", "PASS")

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            model_name,
            tags={"task": "person-re-identification", "architecture": "vit_b_16", "registry.alias.policy": "candidate-only"},
        )
    versions = client.search_model_versions(f"name = '{model_name}'")
    version = next((item for item in versions if item.tags.get("release.id") == release_id), None)
    if version is not None and version.run_id != run_id:
        raise ValueError(f"Release {release_id} is already linked to a different MLflow run.")
    if version is None:
        version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/checkpoints",
            run_id=run_id,
            tags=tags,
            description=f"Validated DVC-backed candidate release {release_id}.",
        )
    else:
        for key, value in tags.items():
            client.set_model_version_tag(model_name, version.version, key, value)
    client.set_registered_model_alias(model_name, args.alias, version.version)
    result = {
        "status": "REGISTERED",
        "registered_model": model_name,
        "version": str(version.version),
        "alias": args.alias,
        "release_id": release_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "source": f"runs:/{run_id}/checkpoints",
        "lineage_tags": tags,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
