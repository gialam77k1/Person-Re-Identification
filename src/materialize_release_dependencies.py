from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


class ReleaseDependencyError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_repository_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseDependencyError(
            f"Release dependency escapes the repository: {value}"
        ) from exc
    return candidate


def manifest_artifact(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    for item in manifest.get("files", []):
        if isinstance(item, dict) and item.get("role") == role:
            return item
    raise ReleaseDependencyError(f"Manifest has no {role} artifact")


def reconstruct_validation_policy(
    *, manifest: dict[str, Any], repository_path: str, source_repository_root: Path
) -> bytes:
    source_commit = str(manifest.get("source", {}).get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ReleaseDependencyError("Manifest source Git commit is invalid")
    try:
        source = subprocess.run(
            ["git", "show", f"{source_commit}:{repository_path}"],
            cwd=source_repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ReleaseDependencyError(
            f"Cannot restore {repository_path} from source commit {source_commit}"
        ) from exc
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ReleaseDependencyError("Validation policy must be a YAML mapping")
    release_id = manifest.get("release", {}).get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise ReleaseDependencyError("Manifest release_id is missing")
    # prepare_iterative_release.py applies this exact, deterministic mutation.
    payload["release_id"] = release_id
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def materialize_validation_policy(
    *,
    manifest: dict[str, Any],
    repository_root: Path,
    candidate_root: Path | None = None,
    source_repository_root: Path | None = None,
) -> Path:
    artifact = manifest_artifact(manifest, "validation_policy")
    repository_path = str(artifact.get("repository_path", ""))
    expected_sha = str(artifact.get("sha256", ""))
    if not repository_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ReleaseDependencyError("Validation policy manifest entry is invalid")
    destination = safe_repository_path(repository_root, repository_path)

    content: bytes | None = None
    if candidate_root is not None:
        candidate = safe_repository_path(candidate_root, repository_path)
        if candidate.is_file():
            content = candidate.read_bytes()
    if content is None:
        content = reconstruct_validation_policy(
            manifest=manifest,
            repository_path=repository_path,
            source_repository_root=source_repository_root or repository_root,
        )
    if sha256_bytes(content) != expected_sha:
        raise ReleaseDependencyError(
            "Restored validation policy checksum does not match the candidate manifest"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.release.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore immutable non-DVC dependencies for a model release."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--candidate-root")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ReleaseDependencyError("Release manifest must be a JSON object")
    destination = materialize_validation_policy(
        manifest=manifest,
        repository_root=Path(args.repository_root).resolve(),
        candidate_root=(
            Path(args.candidate_root).resolve() if args.candidate_root else None
        ),
        source_repository_root=Path(args.repository_root).resolve(),
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
