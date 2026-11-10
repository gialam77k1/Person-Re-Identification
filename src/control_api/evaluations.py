from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.control_api.config import ControlSettings
from src.control_api.store import CycleStore


SPLITS = ("query", "bounding_box_test", "bounding_box_train")
DATASET_DIR_ALIASES = {"market1501": "Market-1501-v15.09.15"}
RANKING_METRICS = ("rank1", "rank5", "rank10", "rank20", "mAP", "mINP")


class EvaluationService:
    def __init__(self, settings: ControlSettings, store: CycleStore) -> None:
        self.settings = settings
        self.store = store

    def overview(self) -> dict[str, Any]:
        versions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for cycle in self.store.list_cycles(limit=50):
            v2_event = self.store.latest_cycle_stage_metrics(cycle["id"], "evaluate_v2")
            if v2_event:
                metrics = v2_event["payload"]["metrics"]
                for domain, label in (("duke", "DukeMTMC-reID"), ("market1501", "Market-1501")):
                    report = metrics.get(domain)
                    self._append_report(
                        versions,
                        seen,
                        report,
                        release_id=cycle["target_release"],
                        model_version=self._release_version(cycle["target_release"]),
                        domain=label,
                        cycle_id=cycle["id"],
                    )
            v1_event = self.store.latest_cycle_stage_metrics(cycle["id"], "evaluate_v1")
            if v1_event:
                metrics = v1_event["payload"]["metrics"]
                report = metrics.get("evaluation")
                self._append_report(
                    versions,
                    seen,
                    report,
                    release_id=cycle["source_release"],
                    model_version=self._release_version(cycle["source_release"]),
                    domain="DukeMTMC-reID",
                    cycle_id=cycle["id"],
                )

        for report_path in self._report_paths():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            self._append_report(
                versions,
                seen,
                report,
                release_id=str(
                    report.get("release_id")
                    or str(report.get("run_slug", "evaluation")).removesuffix("-deployment")
                ),
                model_version=self._release_version(str(report.get("release_id") or "model")),
                domain=self._domain_label(str(report.get("dataset") or "evaluation")),
                cycle_id=None,
            )

        dataset = self.settings.allowed_datasets[0]
        return {
            "dataset": dataset,
            "dataset_samples": self.dataset_samples(dataset),
            "identity_camera_groups": self.identity_camera_groups(dataset),
            "versions": versions,
        }

    def dataset_samples(self, dataset: str, limit: int = 12) -> list[dict[str, Any]]:
        root = self._dataset_dir(dataset)
        query_dir = root / "query"
        if not query_dir.is_dir():
            return []
        images = sorted(
            path for path in query_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(images) > limit:
            step = max(len(images) // limit, 1)
            images = images[::step][:limit]
        return [
            {
                "image_path": f"query/{path.name}",
                "person_id": self._filename_field(path.name, 0),
                "camera_id": self._filename_field(path.name, 1, prefix="c"),
            }
            for path in images
        ]

    def identity_camera_groups(
        self,
        dataset: str,
        limit: int = 4,
        cameras_per_identity: int = 6,
    ) -> list[dict[str, Any]]:
        """Return deterministic examples of one identity observed by multiple cameras."""
        root = self._dataset_dir(dataset)
        identities: dict[int, dict[int, dict[str, Any]]] = {}
        for split in SPLITS:
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                person_id = self._filename_field(path.name, 0)
                camera_id = self._filename_field(path.name, 1, prefix="c")
                if person_id is None or camera_id is None or person_id < 0:
                    continue
                identities.setdefault(person_id, {}).setdefault(
                    camera_id,
                    {
                        "image_path": f"{split}/{path.name}",
                        "person_id": person_id,
                        "camera_id": camera_id,
                    },
                )

        candidates = [
            (person_id, cameras)
            for person_id, cameras in identities.items()
            if len(cameras) >= 2
        ]
        candidates.sort(key=lambda item: (-len(item[1]), item[0]))
        return [
            {
                "person_id": person_id,
                "camera_count": len(cameras),
                "images": [cameras[camera_id] for camera_id in sorted(cameras)[:cameras_per_identity]],
            }
            for person_id, cameras in candidates[:limit]
        ]

    def resolve_image(self, dataset: str, image_path: str) -> Path:
        root = self._dataset_dir(dataset).resolve()
        relative = Path(image_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid evaluation image path")
        if relative.parts[0] not in SPLITS:
            raise ValueError("Evaluation image must belong to a dataset split")
        resolved = (root / relative).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("Evaluation image escapes dataset root")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _append_report(
        self,
        target: list[dict[str, Any]],
        seen: set[tuple[str, str]],
        report: Any,
        *,
        release_id: str,
        model_version: str,
        domain: str,
        cycle_id: str | None,
    ) -> None:
        if not isinstance(report, dict):
            return
        key = (release_id, domain)
        if key in seen:
            return
        seen.add(key)
        dataset = str(report.get("dataset") or self.settings.allowed_datasets[0])
        preview = report.get("retrieval_preview") or self._legacy_preview(report)
        base_metrics = self._protocol_metrics(report, "base")
        reranked_metrics = self._protocol_metrics(report, "rerank")
        primary_protocol = "reranked" if report.get("use_rerank") and reranked_metrics else "base"
        primary_metrics = reranked_metrics if primary_protocol == "reranked" else base_metrics
        target.append(
            {
                "release_id": release_id,
                "model_version": model_version,
                "domain": domain,
                "dataset": dataset,
                "cycle_id": cycle_id,
                "metrics": {
                    **primary_metrics,
                    "query_count": report.get("query_count", report.get("num_query_samples")),
                    "gallery_count": report.get("gallery_count", report.get("num_gallery_samples")),
                },
                "primary_protocol": primary_protocol,
                "protocols": {
                    "base": base_metrics,
                    "reranked": reranked_metrics or None,
                },
                "git_commit": report.get("git_commit"),
                "dataset_dvc_revision": report.get("dataset_dvc_revision"),
                "generated_at": report.get("generated_at"),
                "retrieval_preview": self._normalize_preview(preview),
            }
        )

    @staticmethod
    def _protocol_metrics(report: dict[str, Any], suffix: str) -> dict[str, Any]:
        """Return one explicit metric protocol while supporting legacy reports."""
        result: dict[str, Any] = {}
        for metric in RANKING_METRICS:
            suffixed = report.get(f"{metric}_{suffix}")
            if suffixed is not None:
                result[metric] = suffixed
            elif suffix == "base":
                # Reports without re-ranking predate explicit protocol fields;
                # their canonical values are base retrieval metrics.
                result[metric] = report.get(metric)
        return result

    def _report_paths(self) -> list[Path]:
        root = self.settings.evaluation_artifacts_root
        patterns = (
            "cross-domain/market1501-v1-on-duke/metrics/v1_on_duke.json",
            "duke/*/evaluations/*/metrics/evaluation_*.json",
            "pipeline/*/deployment/metrics/deployment_retrieval_latest.json",
        )
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
        return paths

    def _dataset_dir(self, dataset: str) -> Path:
        directory = DATASET_DIR_ALIASES.get(dataset, dataset)
        if dataset not in self.settings.allowed_datasets and dataset not in DATASET_DIR_ALIASES:
            raise ValueError("Dataset is not allowed")
        return self.settings.dataset_root / directory

    @staticmethod
    def _release_version(release_id: str) -> str:
        match = re.search(r":v(\d+)(?:\b|-)", release_id)
        return f"v{match.group(1)}" if match else release_id

    @staticmethod
    def _domain_label(dataset: str) -> str:
        return {
            "market1501": "Market-1501",
            "dukemtmc-reid": "DukeMTMC-reID",
        }.get(dataset, dataset)

    @staticmethod
    def _filename_field(name: str, index: int, prefix: str = "") -> int | None:
        parts = name.split("_")
        if index >= len(parts):
            return None
        value = parts[index]
        if prefix and value.startswith(prefix):
            value = value[len(prefix) :]
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _portable_path(value: Any) -> str:
        path = Path(str(value))
        for split in SPLITS:
            if split in path.parts:
                return Path(*path.parts[path.parts.index(split) :]).as_posix()
        return path.name

    def _normalize_preview(self, preview: Any) -> list[dict[str, Any]]:
        if not isinstance(preview, list):
            return []
        normalized = []
        for item in preview[:12]:
            if not isinstance(item, dict) or not isinstance(item.get("query"), dict):
                continue
            query = dict(item["query"])
            query["image_path"] = self._portable_path(query.get("image_path", ""))
            predictions = []
            for prediction in item.get("predictions", [])[:10]:
                if not isinstance(prediction, dict):
                    continue
                value = dict(prediction)
                value["image_path"] = self._portable_path(value.get("image_path", ""))
                predictions.append(value)
            normalized.append({**item, "query": query, "predictions": predictions})
        return normalized

    def _legacy_preview(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        normalized = []
        for item in report.get("topk_preview", [])[:12]:
            query_pid = item.get("query_pid")
            query_match = re.search(r"_c(\d+)", str(item.get("query_path", "")))
            query_camera = int(query_match.group(1)) - 1 if query_match else None
            predictions = []
            for match in item.get("top_matches", [])[:10]:
                if (
                    match.get("gallery_pid") == query_pid
                    and match.get("gallery_camid") == query_camera
                ):
                    continue
                distance = float(match.get("distance", 1.0))
                predictions.append(
                    {
                        "rank": len(predictions) + 1,
                        "image_path": match.get("gallery_path"),
                        "person_id": match.get("gallery_pid"),
                        "camera_id": match.get("gallery_camid"),
                        "distance": distance,
                        "similarity": 1.0 - distance,
                        "correct": match.get("gallery_pid") == query_pid,
                    }
                )
            normalized.append(
                {
                    "query": {
                        "image_path": item.get("query_path"),
                        "person_id": query_pid,
                        "camera_id": query_camera,
                    },
                    "predictions": predictions,
                    "top1_correct": bool(predictions and predictions[0]["correct"]),
                }
            )
        return normalized
