from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class WorkspaceResetError(ValueError):
    pass


def release_slug(release_id: str) -> str:
    name, separator, version = release_id.partition(":")
    if not separator or not name or not version.startswith("v") or not version[1:].isdigit():
        raise WorkspaceResetError("release_id must use the name:vN format")
    return f"{name}-{version}"


def guarded_candidate_path(
    workspace: Path, requested: Path, *, family: str, expected_slug: str
) -> Path:
    root = workspace.resolve()
    allowed_parent = (root / "artifacts" / family).resolve()
    candidate = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    if candidate.parent != allowed_parent or candidate.name != expected_slug:
        raise WorkspaceResetError(
            f"Refusing to reset candidate path outside artifacts/{family}/{expected_slug}: "
            f"{candidate}"
        )
    return candidate


def reset_candidate_workspace(
    *, workspace: Path, release_id: str, run_root: Path, pipeline_root: Path
) -> list[str]:
    slug = release_slug(release_id)
    targets = (
        guarded_candidate_path(workspace, run_root, family="duke", expected_slug=slug),
        guarded_candidate_path(
            workspace, pipeline_root, family="pipeline", expected_slug=slug
        ),
    )
    removed: list[str] = []
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove stale candidate outputs before a fresh, non-resume training run."
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pipeline-root", required=True)
    args = parser.parse_args()
    removed = reset_candidate_workspace(
        workspace=Path(args.workspace),
        release_id=args.release_id,
        run_root=Path(args.run_root),
        pipeline_root=Path(args.pipeline_root),
    )
    print(f"Candidate workspace ready; removed {len(removed)} stale path(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
