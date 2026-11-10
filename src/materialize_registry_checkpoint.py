from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def materialize(
    release_id: str,
    destination: Path | None,
    tracking_uri: str,
    manifest_destination: Path | None = None,
) -> dict[str, str]:
    import mlflow
    from mlflow import MlflowClient

    model_name = release_id.partition(":")[0]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    version = next(
        (
            item
            for item in versions
            if item.tags.get("release.id") == release_id
            or item.tags.get("release_id") == release_id
        ),
        None,
    )
    if version is None or not version.run_id:
        raise ValueError(f"MLflow Registry has no immutable version for {release_id!r}")
    result = {
        "release_id": release_id,
        "registered_model": model_name,
        "registry_version": str(version.version),
        "run_id": str(version.run_id),
    }
    if destination is not None:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{version.run_id}/checkpoints/best_model.pth"
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
        result["checkpoint"] = str(destination)
    if manifest_destination is not None:
        try:
            manifest = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{version.run_id}/registry/release/model_manifest.json"
                )
            )
        except Exception:
            # v2 was registered before the version-neutral registry path existed.
            manifest = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{version.run_id}/registry/v2/model_manifest.json"
                )
            )
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, manifest_destination)
        result["manifest"] = str(manifest_destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a champion checkpoint by immutable release ID."
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--manifest-destination")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = materialize(
        args.release_id,
        Path(args.destination) if args.destination else None,
        args.tracking_uri,
        Path(args.manifest_destination) if args.manifest_destination else None,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
