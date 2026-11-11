from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


class ReleasePublicationError(RuntimeError):
    pass


def resolve_stage(
    publication: dict[str, Any], expected_release_id: str, expected_slug: str
) -> str:
    if publication.get("status") != "PASS":
        raise ReleasePublicationError("Candidate DVC publication did not PASS.")
    if publication.get("release_id") != expected_release_id:
        raise ReleasePublicationError("Candidate publication release_id does not match.")
    stage = publication.get("dvc", {}).get("stage")
    if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
        raise ReleasePublicationError("Candidate publication has an invalid DVC stage.")
    manifest_path = publication.get("model_manifest", {}).get("path")
    expected_manifest = f"artifacts/pipeline/{expected_slug}/release/model_manifest.json"
    if manifest_path != expected_manifest:
        raise ReleasePublicationError("Candidate publication path does not match release_slug.")
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate candidate publication metadata and return its DVC stage."
    )
    parser.add_argument("--publication", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-slug", required=True)
    args = parser.parse_args()
    path = Path(args.publication)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleasePublicationError("DVC publication report must be a JSON object.")
    print(resolve_stage(value, args.release_id, args.release_slug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
