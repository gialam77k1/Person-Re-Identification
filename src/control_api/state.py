from __future__ import annotations

from enum import Enum


class CycleStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    PASS = "PASS"
    FAIL = "FAIL"
    CANCELLED = "CANCELLED"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


PIPELINE_STAGES = (
    "validate_data",
    "extract_embeddings",
    "detect_drift",
    "evaluate_v1",
    "retraining_approval",
    "train_v2",
    "evaluate_v2",
    "validate_release",
    "export_onnx",
    "publish_artifacts",
    "register_candidate",
    "deploy_staging",
    "staging_test",
    "production_approval",
)
