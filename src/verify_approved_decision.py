from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_approved_decision(
    decision: dict[str, Any],
    evidence_manifest: dict[str, Any],
    *,
    decision_sha256: str,
    manifest_sha256: str,
    cycle_id: str,
    source_release: str,
    target_release: str,
    dataset: str,
    target_epochs: int,
) -> None:
    if canonical_sha256(decision) != decision_sha256:
        raise ValueError("Approved decision checksum does not match Control API dispatch.")
    if decision.get("status") != "RETRAIN_APPROVED":
        raise ValueError("Decision status is not RETRAIN_APPROVED.")
    if decision.get("cycle_id") != cycle_id:
        raise ValueError("Decision cycle ID does not match this workflow.")
    if decision.get("base_release_id") != source_release:
        raise ValueError("Decision source release does not match this workflow.")
    if decision.get("target_release_id") != target_release:
        raise ValueError("Decision target release does not match this workflow.")
    if str(decision.get("target_dataset", "")).lower() != dataset.lower():
        raise ValueError("Decision target dataset does not match this workflow.")
    if int(decision.get("target_epochs", 0)) != target_epochs:
        raise ValueError("Decision epoch limit does not match the approved training contract.")
    if not str(decision.get("approved_by") or "").strip():
        raise ValueError("Decision is missing the approver identity.")
    try:
        datetime.fromisoformat(str(decision["approved_at"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Decision is missing a valid approval timestamp.") from exc
    evidence = decision.get("evidence", {})
    if evidence.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Decision does not approve this evidence manifest checksum.")
    if evidence_manifest.get("cycle_id") != cycle_id:
        raise ValueError("Evidence manifest cycle ID does not match this workflow.")
    if str(evidence.get("workflow_run_id")) != str(
        evidence_manifest.get("workflow_run_id")
    ):
        raise ValueError("Decision and evidence workflow run IDs do not match.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Control API retraining decision.")
    parser.add_argument("--decision", required=True)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-epochs", type=int, required=True)
    args = parser.parse_args()
    try:
        decision_path = Path(args.decision)
        manifest_path = Path(args.evidence_manifest)
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_approved_decision(
            decision,
            manifest,
            decision_sha256=args.decision_sha256,
            manifest_sha256=file_sha256(manifest_path),
            cycle_id=args.cycle_id,
            source_release=args.source_release,
            target_release=args.target_release,
            dataset=args.dataset,
            target_epochs=args.target_epochs,
        )
        print(
            json.dumps(
                {
                    "status": decision["status"],
                    "approved_by": decision["approved_by"],
                    "approved_at": decision["approved_at"],
                    "decision_sha256": args.decision_sha256,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
