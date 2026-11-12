from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.config import ReleaseContract, verify_serving_model_artifacts


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
    current_lock = directory / "current.dvc.lock"
    previous_lock = directory / "previous.dvc.lock"
    if current.is_file():
        shutil.copy2(current, previous)
    if current_lock.is_file():
        shutil.copy2(current_lock, previous_lock)
    temporary = directory / "current.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(current)
    repository_lock = ROOT / "dvc.lock"
    if not repository_lock.is_file():
        raise ReleaseError("The immutable candidate dvc.lock is missing.")
    shutil.copy2(repository_lock, current_lock)
    (directory / "pending.json").unlink(missing_ok=True)


def write_pending_state(state_root: Path, environment: str, record: dict[str, Any]) -> None:
    directory = state_root / environment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pending.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def select_rollback_record(state_root: Path, environment: str) -> tuple[dict[str, Any], Path]:
    directory = state_root / environment
    prefix = "current" if (directory / "pending.json").is_file() else "previous"
    target = directory / f"{prefix}.json"
    lock = directory / f"{prefix}.dvc.lock"
    if not target.is_file():
        raise ReleaseError(f"No rollback target recorded for {environment}")
    if not lock.is_file():
        raise ReleaseError(f"No immutable DVC lock recorded for {environment} rollback")
    return read_json(target), lock


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


def preflight_serving_release(
    manifest_path: Path,
    triton_repository: Path,
    expected_release_id: str,
) -> tuple[str, str]:
    try:
        triton_manifest = read_json(triton_repository / "triton_model_manifest.json")
        model_name = str(triton_manifest.get("model_name") or "")
        model_version = str(triton_manifest.get("model_version") or "")
        if not model_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_version):
            raise ValueError("Triton model manifest has an invalid model name or version.")
        contract = ReleaseContract.load(manifest_path)
        if contract.release_id != expected_release_id:
            raise ValueError("Serving contract release_id does not match the deployment request.")
        verify_serving_model_artifacts(
            contract,
            triton_repository,
            model_name,
            model_version,
        )
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"Serving preflight failed: {exc}") from exc
    return model_name, model_version


def require_healthy_qdrant(container_name: str = "reid-qdrant") -> None:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            container_name,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    status = result.stdout.strip()
    if result.returncode != 0 or status != "running/healthy":
        detail = result.stderr.strip() or status or "container not found"
        raise ReleaseError(f"Qdrant must be running and healthy before deployment: {detail}")


def materialize_serving_bundle(
    state_root: Path,
    release_root: Path,
    manifest_path: Path,
    triton_repository: Path,
) -> tuple[Path, Path]:
    """Copy a validated release outside the ephemeral Actions checkout.

    The API and Triton must not bind-mount files that actions/checkout can
    remove after the deployment job. The manifest digest makes the bundle
    immutable and allows staging and production to reuse the exact bytes.
    """
    artifacts_root = (ROOT / "artifacts").resolve()
    try:
        release_relative = release_root.resolve().relative_to(artifacts_root)
        triton_relative = triton_repository.resolve().relative_to(artifacts_root)
    except ValueError as exc:
        raise ReleaseError("Serving release must be located under repository artifacts/") from exc

    digest = sha256_file(manifest_path)
    serving_root = state_root.resolve() / "serving"
    bundle_root = serving_root / digest[:16]
    persistent_artifacts = bundle_root / "artifacts"
    persistent_release = persistent_artifacts / release_relative
    persistent_triton = persistent_artifacts / triton_relative
    marker = bundle_root / "bundle.json"

    if bundle_root.exists():
        metadata = read_json(marker) if marker.is_file() else {}
        if (
            metadata.get("manifest_sha256") != digest
            or not persistent_release.is_dir()
            or not persistent_triton.is_dir()
        ):
            raise ReleaseError(f"Existing serving bundle is incomplete: {bundle_root}")
        return persistent_artifacts, persistent_triton

    serving_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=serving_root))
    try:
        temporary_artifacts = temporary / "artifacts"
        destination = temporary_artifacts / release_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(release_root, destination)
        temporary_marker = temporary / "bundle.json"
        temporary_marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_sha256": digest,
                    "release_path": str(release_relative),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(bundle_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return persistent_artifacts, persistent_triton


def build_compose_environment(
    base: dict[str, str],
    *,
    image: str,
    manifest_relative: Path,
    triton_relative: Path,
    triton_repository: Path,
    serving_artifacts_root: Path,
    model_name: str,
    model_version: str,
) -> dict[str, str]:
    environment = base.copy()
    environment.update(
        {
            "API_IMAGE": image,
            # These paths are resolved inside the inference image, where the
            # repository artifacts directory is mounted at /app/artifacts.
            "API_MODEL_MANIFEST_PATH": str(manifest_relative),
            "API_TRITON_MODEL_REPOSITORY": str(triton_relative),
            "SERVING_ARTIFACTS_ROOT": str(serving_artifacts_root.resolve()),
            "TRITON_MODEL_NAME": model_name,
            "TRITON_MODEL_VERSION": model_version,
            # Docker Compose resolves a relative host path without ./ as a
            # named volume. Always pass the canonical host path for Triton.
            "TRITON_MODEL_REPOSITORY": str(triton_repository.resolve()),
        }
    )
    return environment


def deploy(args: argparse.Namespace, *, is_rollback: bool = False) -> dict[str, Any]:
    report_path = Path(args.validation_report).resolve()
    manifest_path = Path(args.manifest).resolve()
    _, manifest = validate_candidate(report_path, manifest_path, args.release_id)
    validate_image_reference(args.image, args.image_git_commit, args.release_id)

    triton_repository = Path(args.triton_model_repository).resolve()
    if not triton_repository.is_dir():
        raise ReleaseError(f"Triton model repository is missing: {triton_repository}")
    model_name, model_version = preflight_serving_release(
        manifest_path, triton_repository, args.release_id
    )
    require_healthy_qdrant(os.environ.get("QDRANT_CONTAINER_NAME", "reid-qdrant"))
    release_root = manifest_path.parents[1]
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
        "dvc_stage": "package_triton",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "rollback": is_rollback,
        "triton_model_name": model_name,
        "triton_model_version": model_version,
    }
    publication_path = release_root / "release" / "dvc_publication.json"
    if publication_path.is_file():
        publication = read_json(publication_path)
        stage = publication.get("dvc", {}).get("stage")
        if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
            raise ReleaseError("DVC publication report has an invalid stage")
        record["dvc_stage"] = stage
    state_root = Path(args.state_root).resolve()
    serving_artifacts_root, persistent_triton_repository = materialize_serving_bundle(
        state_root,
        release_root,
        manifest_path,
        triton_repository,
    )
    record["serving_artifacts_root"] = str(serving_artifacts_root)
    record["persistent_triton_model_repository"] = str(persistent_triton_repository)
    write_pending_state(state_root, args.environment, record)

    compose_environment = build_compose_environment(
        dict(os.environ),
        image=args.image,
        manifest_relative=manifest_relative,
        triton_relative=triton_relative,
        triton_repository=persistent_triton_repository,
        serving_artifacts_root=serving_artifacts_root,
        model_name=model_name,
        model_version=model_version,
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
        compose
        + [
            "up",
            "-d",
            "--no-build",
            "--no-deps",
            "--force-recreate",
            "triton-server",
            "inference-api",
        ],
        check=True,
        cwd=ROOT,
        env=compose_environment,
    )
    try:
        wait_for_release(args.api_url, args.release_id, attempts=90)
    except ReleaseError:
        subprocess.run(compose + ["ps"], cwd=ROOT, env=compose_environment, check=False)
        subprocess.run(
            compose + ["logs", "--no-color", "--tail", "200", "triton-server", "inference-api"],
            cwd=ROOT,
            env=compose_environment,
            check=False,
        )
        raise

    write_deployment_state(state_root, args.environment, record)
    return record


def rollback(args: argparse.Namespace) -> dict[str, Any]:
    previous, previous_lock = select_rollback_record(Path(args.state_root), args.environment)
    stage = str(previous.get("dvc_stage", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stage):
        raise ReleaseError("Rollback state has an invalid DVC stage")
    shutil.copy2(previous_lock, ROOT / "dvc.lock")
    subprocess.run(
        [sys.executable, "-m", "dvc", "pull", "--with-deps", stage],
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
