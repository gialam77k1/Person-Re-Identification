from __future__ import annotations

import hashlib
import json
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.retraining_decision_gate import build_retraining_decision


class ApprovalEvidenceError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _by_basename(files: dict[str, bytes]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for name, content in files.items():
        basename = Path(name).name
        if basename in result:
            raise ApprovalEvidenceError(f"Duplicate evidence filename: {basename}")
        result[basename] = content
    return result


def create_approved_decision(
    *,
    files: dict[str, bytes],
    policy: dict[str, Any],
    cycle_id: str,
    actor: str,
    source_release: str,
    target_release: str,
    dataset: str,
    evidence_run_id: str,
    evidence_artifact: str,
) -> dict[str, Any]:
    bundle = _by_basename(files)
    required = {"evidence_manifest.json", "drift_report.json", "evaluation_report.json"}
    missing = required - set(bundle)
    if missing:
        raise ApprovalEvidenceError(f"Evidence artifact is missing: {sorted(missing)}")
    try:
        manifest = json.loads(bundle["evidence_manifest.json"])
        drift = json.loads(bundle["drift_report.json"])
        evaluation = json.loads(bundle["evaluation_report.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApprovalEvidenceError(f"Evidence contains invalid JSON: {exc}") from exc
    if manifest.get("cycle_id") != cycle_id:
        raise ApprovalEvidenceError("Evidence cycle ID does not match the approval request.")
    if str(manifest.get("workflow_run_id")) != str(evidence_run_id):
        raise ApprovalEvidenceError("Evidence workflow run ID does not match the approval request.")
    if manifest.get("release_id") != source_release:
        raise ApprovalEvidenceError("Evidence source release does not match the cycle.")
    if str(manifest.get("dataset", "")).lower() != dataset.lower():
        raise ApprovalEvidenceError("Evidence dataset does not match the cycle.")
    for role, spec in manifest.get("evidence", {}).items():
        name = Path(str(spec.get("path", ""))).name
        content = bundle.get(name)
        if content is None or sha256_bytes(content) != spec.get("sha256"):
            raise ApprovalEvidenceError(f"Evidence checksum mismatch for {role}.")
    effective_policy = copy.deepcopy(policy)
    effective_policy.setdefault("candidate", {})["base_release_id"] = source_release
    decision = build_retraining_decision(drift, evaluation, effective_policy, actor)
    if decision.get("status") != "RETRAIN_APPROVED":
        raise ApprovalEvidenceError(
            f"Decision gate returned {decision.get('status')}, retraining remains blocked."
        )
    approved_at = datetime.now(timezone.utc).isoformat()
    decision.update(
        {
            "cycle_id": cycle_id,
            "target_release_id": target_release,
            "approved_at": approved_at,
            "evidence": {
                "workflow_run_id": str(evidence_run_id),
                "artifact_name": evidence_artifact,
                "manifest_sha256": sha256_bytes(bundle["evidence_manifest.json"]),
                "drift_report_sha256": sha256_bytes(bundle["drift_report.json"]),
                "evaluation_report_sha256": sha256_bytes(bundle["evaluation_report.json"]),
                "dataset_dvc_revision": manifest.get("dataset_dvc_revision"),
                "source_git_commit": manifest.get("source_git_commit"),
            },
        }
    )
    return decision


def load_policy(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
