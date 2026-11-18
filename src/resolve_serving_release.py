from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


def release_slug(release_id: str) -> str:
    if not re.fullmatch(r"[^:\s]+:[^:\s]+", release_id):
        raise ValueError(f"Invalid serving release_id: {release_id!r}")
    return release_id.replace(":", "-").replace("_", "-")


def resolve(api_url: str) -> tuple[str, str]:
    url = f"{api_url.rstrip('/')}/model-info"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resolve the active serving release from {url}: {exc}") from exc
    release_id = payload.get("release_id") if isinstance(payload, dict) else None
    if not isinstance(release_id, str) or not release_id:
        raise ValueError("Inference API /model-info returned no release_id")
    return release_id, release_slug(release_id)


def resolve_reference(
    deployment_state: Path | None, release_id: str, slug: str
) -> Path | None:
    if deployment_state is None or not deployment_state.is_file():
        return None
    try:
        state = json.loads(deployment_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read deployment state {deployment_state}: {exc}") from exc
    if state.get("release_id") != release_id:
        raise ValueError(
            "Deployment state release does not match the inference API: "
            f"{state.get('release_id')!r} != {release_id!r}"
        )
    root_value = state.get("serving_artifacts_root")
    if not isinstance(root_value, str) or not root_value:
        return None
    root = Path(root_value).resolve()
    reference = (
        root
        / "pipeline"
        / slug
        / "reference"
        / "embeddings"
        / "reference_embeddings.npy"
    ).resolve()
    try:
        reference.relative_to(root)
    except ValueError as exc:
        raise ValueError("Resolved reference artifact escapes the serving bundle") from exc
    if not reference.is_file():
        raise ValueError(f"Active serving reference artifact is missing: {reference}")
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the active release and its pipeline artifact slug."
    )
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--deployment-state")
    args = parser.parse_args()
    try:
        release_id, slug = resolve(args.api_url)
        reference = resolve_reference(
            Path(args.deployment_state) if args.deployment_state else None,
            release_id,
            slug,
        )
        print(f"{release_id}\t{slug}\t{reference or ''}")
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
