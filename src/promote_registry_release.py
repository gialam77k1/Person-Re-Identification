from __future__ import annotations

import argparse
import json
from pathlib import Path


def promote(release_id: str, alias: str, tracking_uri: str) -> dict[str, str]:
    import mlflow
    from mlflow import MlflowClient

    model_name = release_id.partition(":")[0]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    version = next(
        (item for item in versions if item.tags.get("release.id") == release_id), None
    )
    if version is None:
        raise ValueError(f"Registry release not found: {release_id}")
    if version.tags.get("validation.status") != "PASS":
        raise ValueError("Only a validated registry version may become champion")
    client.set_registered_model_alias(model_name, alias, version.version)
    client.set_model_version_tag(model_name, version.version, "promotion.alias", alias)
    return {
        "status": "PROMOTED",
        "release_id": release_id,
        "registered_model": model_name,
        "version": str(version.version),
        "alias": alias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated MLflow release alias.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = promote(args.release_id, args.alias, args.tracking_uri)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
