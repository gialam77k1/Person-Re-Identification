from __future__ import annotations

import math
from typing import Any

from src.control_api.store import CycleStore


def _label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def render_control_metrics(store: CycleStore) -> str:
    """Expose lifecycle evidence in Prometheus text format.

    Only the latest completed drift callback is exported, preventing old cycles
    from leaving multiple equally-current drift series in the dashboard.
    """
    lines = [
        "# HELP reid_control_api_up Control API metrics endpoint availability.",
        "# TYPE reid_control_api_up gauge",
        "reid_control_api_up 1",
    ]
    event = store.latest_stage_metrics("detect_drift")
    if event is None:
        return "\n".join(lines) + "\n"

    metrics = event["payload"]["metrics"]
    labels = (
        f'cycle_id="{_label(event["cycle_id"])}",'
        f'dataset="{_label(event.get("dataset", "unknown"))}",'
        f'source_release="{_label(event.get("source_release", "unknown"))}"'
    )
    detected = 1.0 if metrics.get("drift_detected") is True else 0.0
    lines.extend(
        [
            "# HELP reid_embedding_drift_detected Latest lifecycle drift decision (1=drift).",
            "# TYPE reid_embedding_drift_detected gauge",
            f"reid_embedding_drift_detected{{{labels}}} {detected:g}",
        ]
    )

    checks = metrics.get("checks", {})
    if isinstance(checks, dict):
        lines.extend(
            [
                "# HELP reid_embedding_drift_metric Latest drift detector metric value.",
                "# TYPE reid_embedding_drift_metric gauge",
            ]
        )
        for name, result in sorted(checks.items()):
            if not isinstance(result, dict):
                continue
            metric_labels = f'{labels},metric="{_label(name)}"'
            lines.append(
                f"reid_embedding_drift_metric{{{metric_labels}}} "
                f"{_number(result.get('value')):.12g}"
            )
    return "\n".join(lines) + "\n"
