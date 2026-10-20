from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class ReleaseError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ReleaseError(f"Artifact path escapes repository: {value}") from exc
    return candidate


def validate_candidate(
    validation_report_path: Path,
    manifest_path: Path,
    expected_release_id: str,
    *,
    verify_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(validation_report_path)
    manifest = read_json(manifest_path)
    errors: list[str] = []

    if report.get("status") != "PASS" or report.get("passed") is not True:
        errors.append("validation report is not PASS")
    if report.get("release_id") != expected_release_id:
        errors.append("validation report release_id does not match")
    failed_checks = report.get("failed_checks")
    if not isinstance(failed_checks, list) or failed_checks:
        errors.append("validation report contains failed or invalid checks")

    release = manifest.get("release", {})
    validation = manifest.get("validation", {})
    source = manifest.get("source", {})
    if release.get("release_id") != expected_release_id:
        errors.append("manifest release_id does not match")
    if release.get("status") != "validated" or validation.get("status") != "PASS":
        errors.append("manifest is not validated")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit", ""))):
        errors.append("manifest source Git commit is invalid")

    files = manifest.get("files")
    report_files = report.get("validated_artifacts")
    if not isinstance(files, list) or not files:
        errors.append("manifest has no artifact checksums")
        files = []
    if not isinstance(report_files, list):
        errors.append("validation report has no validated artifacts")
        report_files = []
    report_checksums = {
        item.get("role"): item.get("sha256")
        for item in report_files
        if isinstance(item, dict)
    }
    for item in files:
        if not isinstance(item, dict):
            errors.append("manifest contains an invalid artifact entry")
            continue
        role = str(item.get("role", "unknown"))
        expected_sha = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            errors.append(f"artifact {role} has an invalid checksum")
            continue
        if report_checksums.get(role) != expected_sha:
            errors.append(f"artifact {role} differs between report and manifest")
        if verify_files:
            path = resolve_repository_path(str(item.get("repository_path", "")))
            if not path.is_file():
                errors.append(f"artifact {role} is missing: {path}")
            elif sha256_file(path) != expected_sha:
                errors.append(f"artifact {role} checksum mismatch")

    if errors:
        raise ReleaseError("; ".join(errors))
    return report, manifest


def release_version_slug(release_id: str) -> str:
    _, separator, version = release_id.partition(":")
    if not separator or not version:
        raise ReleaseError("release_id must use the name:version format")
    slug = re.sub(r"[^a-z0-9_.-]+", "-", version.lower()).strip("-.")
    if not slug:
        raise ReleaseError("release_id has no usable version")
    return slug


def build_image_reference(repository: str, git_commit: str, release_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{7,40}", git_commit.lower()):
        raise ReleaseError("image Git commit must contain 7 to 40 hexadecimal characters")
    return f"{repository}:{git_commit[:7].lower()}-model-{release_version_slug(release_id)}"


def validate_image_reference(image: str, git_commit: str, release_id: str) -> None:
    expected_suffix = f":{git_commit[:7].lower()}-model-{release_version_slug(release_id)}"
    if not image.lower().endswith(expected_suffix):
        raise ReleaseError(f"image must end with {expected_suffix}")


def write_deployment_state(state_root: Path, environment: str, record: dict[str, Any]) -> None:
    directory = state_root / environment
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / "current.json"
    previous = directory / "previous.json"
    if current.is_file():
        shutil.copy2(current, previous)
    temporary = directory / "current.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(current)
    (directory / "pending.json").unlink(missing_ok=True)


def write_pending_state(state_root: Path, environment: str, record: dict[str, Any]) -> None:
    directory = state_root / environment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pending.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def select_rollback_record(state_root: Path, environment: str) -> dict[str, Any]:
    directory = state_root / environment
    target = directory / ("current.json" if (directory / "pending.json").is_file() else "previous.json")
    if not target.is_file():
        raise ReleaseError(f"No rollback target recorded for {environment}")
    return read_json(target)


def api_json(api_url: str, endpoint: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}{endpoint}", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"API request failed for {endpoint}: {exc}") from exc


def wait_for_release(api_url: str, release_id: str, attempts: int = 30) -> None:
    last_error = "API did not become ready"
    for attempt in range(attempts):
        try:
            ready = api_json(api_url, "/ready")
            model_info = api_json(api_url, "/model-info")
            if ready.get("status") == "ready" and model_info.get("release_id") == release_id:
                return
            last_error = f"ready={ready.get('status')}, release_id={model_info.get('release_id')}"
        except ReleaseError as exc:
            last_error = str(exc)
        if attempt + 1 < attempts:
            time.sleep(2)
    raise ReleaseError(f"Deployment smoke test failed: {last_error}")


def deploy(args: argparse.Namespace, *, is_rollback: bool = False) -> dict[str, Any]:
    report_path = Path(args.validation_report).resolve()
    manifest_path = Path(args.manifest).resolve()
    _, manifest = validate_candidate(report_path, manifest_path, args.release_id)
    validate_image_reference(args.image, args.image_git_commit, args.release_id)

    triton_repository = Path(args.triton_model_repository).resolve()
    if not triton_repository.is_dir():
        raise ReleaseError(f"Triton model repository is missing: {triton_repository}")
    try:
        manifest_relative = manifest_path.relative_to(ROOT)
        triton_relative = triton_repository.relative_to(ROOT)
    except ValueError as exc:
        raise ReleaseError("Manifest and Triton repository must be inside the repository") from exc

    record = {
        "schema_version": 1,
        "environment": args.environment,
        "image": args.image,
        "image_git_commit": args.image_git_commit,
        "release_id": args.release_id,
        "model_git_commit": manifest["source"]["git_commit"],
        "manifest": str(manifest_relative),
        "validation_report": str(report_path.relative_to(ROOT)),
        "triton_model_repository": str(triton_relative),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "rollback": is_rollback,
    }
    state_root = Path(args.state_root)
    write_pending_state(state_root, args.environment, record)

    compose_environment = os.environ.copy()
    compose_environment.update(
        {
            "API_IMAGE": args.image,
            "API_MODEL_MANIFEST_PATH": str(manifest_relative),
            "API_TRITON_MODEL_REPOSITORY": str(triton_relative),
            "TRITON_MODEL_REPOSITORY": str(triton_relative),
        }
    )
    compose = [
        "docker",
        "compose",
        "--env-file",
        args.env_file,
        "--file",
        args.compose_file,
    ]
    if not args.skip_pull:
        subprocess.run(["docker", "pull", args.image], check=True, cwd=ROOT)
    subprocess.run(
        compose + ["up", "-d", "--no-build", "--force-recreate", "triton-server", "inference-api"],
        check=True,
        cwd=ROOT,
        env=compose_environment,
    )
    wait_for_release(args.api_url, args.release_id)

    write_deployment_state(state_root, args.environment, record)
    return record


def rollback(args: argparse.Namespace) -> dict[str, Any]:
    previous = select_rollback_record(Path(args.state_root), args.environment)
    release_root = Path(previous["manifest"]).parents[1]
    subprocess.run(
        [sys.executable, "-m", "dvc", "pull", str(release_root)],
        check=True,
        cwd=ROOT,
    )
    args.image = previous["image"]
    args.image_git_commit = previous["image_git_commit"]
    args.release_id = previous["release_id"]
    args.manifest = previous["manifest"]
    args.validation_report = previous["validation_report"]
    args.triton_model_repository = previous["triton_model_repository"]
    return deploy(args, is_rollback=True)


def add_shared_deploy_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--environment", choices=("staging", "production"), required=True)
    parser.add_argument("--image", required=required)
    parser.add_argument("--image-git-commit", required=required)
    parser.add_argument("--release-id", required=required)
    parser.add_argument("--validation-report", required=required)
    parser.add_argument("--manifest", required=required)
    parser.add_argument("--triton-model-repository", required=required)
    parser.add_argument("--compose-file", default="docker-compose.mlops.yml")
    parser.add_argument("--env-file", default=".env.mlops")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument(
        "--state-root",
        default=os.environ.get("DEPLOYMENT_STATE_ROOT", "../person-reid-deployments"),
    )
    parser.add_argument("--skip-pull", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate, deploy, and roll back model releases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Verify a candidate without deploying it.")
    verify.add_argument("--release-id", required=True)
    verify.add_argument("--validation-report", required=True)
    verify.add_argument("--manifest", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy a validated release.")
    add_shared_deploy_arguments(deploy_parser, required=True)

    rollback_parser = subparsers.add_parser("rollback", help="Restore the previous recorded release.")
    add_shared_deploy_arguments(rollback_parser, required=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "verify":
            validate_candidate(
                Path(args.validation_report).resolve(),
                Path(args.manifest).resolve(),
                args.release_id,
            )
            print(f"Candidate validation: PASS ({args.release_id})")
            return 0
        record = deploy(args) if args.command == "deploy" else rollback(args)
        print(json.dumps(record, indent=2))
        return 0
    except (ReleaseError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
