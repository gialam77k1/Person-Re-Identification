from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_embeddings(path: Path, release_id: str | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        rows = []
        paths = [path]
        rotated = path.with_suffix(f"{path.suffix}.previous")
        if rotated.is_file():
            paths.insert(0, rotated)
        for source in paths:
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        if release_id is None or record.get("release_id") == release_id:
                            rows.append(record["embedding"])
        values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2D embedding array in {path}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"NaN or Inf found in {path}.")
    return values.astype(np.float64)


def row_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm embedding found.")
    return values / norms[:, None], norms


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    combined = np.sort(np.concatenate((reference, current)))
    reference_cdf = np.searchsorted(np.sort(reference), combined, side="right") / reference.size
    current_cdf = np.searchsorted(np.sort(current), combined, side="right") / current.size
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def build_drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    release_id: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if reference.shape[1] != current.shape[1]:
        raise ValueError("Reference and current embedding dimensions differ.")
    minimum = int(policy["minimum_current_samples"])
    if current.shape[0] < minimum:
        raise ValueError(f"Current sample has {current.shape[0]} rows; at least {minimum} required.")

    reference_normalized, reference_norms = row_normalize(reference)
    current_normalized, current_norms = row_normalize(current)
    reference_mean = reference_normalized.mean(axis=0)
    current_mean = current_normalized.mean(axis=0)
    reference_centroid = reference_mean / max(np.linalg.norm(reference_mean), 1e-12)
    current_centroid = current_mean / max(np.linalg.norm(current_mean), 1e-12)
    reference_scores = reference_normalized @ reference_centroid
    current_scores = current_normalized @ reference_centroid
    reference_std = reference_normalized.std(axis=0)

    metrics = {
        "centroid_cosine_distance": float(1.0 - np.dot(reference_centroid, current_centroid)),
        "standardized_mean_shift": float(
            np.mean(np.abs(current_mean - reference_mean) / np.maximum(reference_std, 1e-6))
        ),
        "norm_mean_delta": float(abs(current_norms.mean() - reference_norms.mean())),
        "ks_statistic": ks_statistic(reference_scores, current_scores),
    }
    thresholds = policy["thresholds"]
    checks = {}
    for name, value in metrics.items():
        maximum = float(thresholds[f"{name}_max"])
        checks[name] = {"value": value, "maximum": maximum, "passed": value <= maximum}
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 1,
        "release_id": release_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "DRIFT",
        "drift_detected": not passed,
        "reference_samples": int(reference.shape[0]),
        "current_samples": int(current.shape[0]),
        "embedding_dim": int(reference.shape[1]),
        "checks": checks,
    }


def prometheus_text(report: dict[str, Any]) -> str:
    release = str(report["release_id"]).replace('"', '\\"')
    labels = f'release_id="{release}"'
    timestamp = datetime.fromisoformat(report["generated_at"]).timestamp()
    lines = [
        "# HELP reid_embedding_drift_detected Whether the latest drift report failed its policy.",
        "# TYPE reid_embedding_drift_detected gauge",
        f"reid_embedding_drift_detected{{{labels}}} {int(report['drift_detected'])}",
        "# HELP reid_embedding_drift_last_run_timestamp_seconds Unix time of the latest drift report.",
        "# TYPE reid_embedding_drift_last_run_timestamp_seconds gauge",
        f"reid_embedding_drift_last_run_timestamp_seconds{{{labels}}} {timestamp:.0f}",
        f"reid_embedding_drift_current_samples{{{labels}}} {report['current_samples']}",
    ]
    for name, check in report["checks"].items():
        if not math.isfinite(float(check["value"])):
            raise ValueError(f"Non-finite drift metric: {name}")
        lines.append(f'reid_embedding_drift_metric{{{labels},metric="{name}"}} {check["value"]:.10g}')
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recent embeddings with the release reference set.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--config", default="configs/drift_detection.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prometheus-output", required=True)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        policy = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        report = build_drift_report(
            load_embeddings(Path(args.reference)),
            load_embeddings(Path(args.current), release_id=args.release_id),
            args.release_id,
            policy,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        metrics = Path(args.prometheus_output)
        metrics.parent.mkdir(parents=True, exist_ok=True)
        temporary = metrics.with_suffix(f"{metrics.suffix}.tmp")
        temporary.write_text(prometheus_text(report), encoding="utf-8")
        temporary.replace(metrics)
        print(f"Embedding drift: {report['status']} ({report['current_samples']} samples)")
        return 2 if args.fail_on_drift and report["drift_detected"] else 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
