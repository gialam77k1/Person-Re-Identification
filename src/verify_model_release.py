from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="model_releases/market1501-vit-bnneck/v1/model_manifest.json",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results = []
    failed = False
    for file_spec in manifest["files"]:
        paths_to_verify = [("repository", file_spec["repository_path"])]
        if file_spec.get("deployment_path"):
            paths_to_verify.append(("deployment", file_spec["deployment_path"]))

        for path_kind, relative_path in paths_to_verify:
            file_path = ROOT / relative_path
            exists = file_path.is_file()
            actual_size = file_path.stat().st_size if exists else None
            actual_sha256 = sha256_file(file_path) if exists else None
            passed = (
                exists
                and actual_size == file_spec["size_bytes"]
                and actual_sha256 == file_spec["sha256"]
            )
            failed = failed or not passed
            results.append(
                {
                    "role": file_spec["role"],
                    "path_kind": path_kind,
                    "path": str(file_path),
                    "exists": exists,
                    "expected_size_bytes": file_spec["size_bytes"],
                    "actual_size_bytes": actual_size,
                    "expected_sha256": file_spec["sha256"],
                    "actual_sha256": actual_sha256,
                    "passed": passed,
                }
            )

    summary = {
        "release_id": manifest["release"]["release_id"],
        "source_git_commit": manifest["source"]["git_commit"],
        "manifest": str(manifest_path),
        "passed": not failed,
        "files": results,
    }
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
