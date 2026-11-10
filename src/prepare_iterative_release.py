from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


VERSION = re.compile(r"^.+:v(?P<version>\d+)$")


def update_yaml(path: Path, updates: dict) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, values in updates.items():
        if isinstance(values, dict):
            payload.setdefault(section, {}).update(values)
        else:
            payload[section] = values
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare(
    *, source_release: str, target_release: str, checkpoint: str, run_root: str,
    dataset_root: str | None = None, dataset_pointer: str | None = None,
) -> dict[str, str | int]:
    match = VERSION.fullmatch(target_release)
    if match is None:
        raise ValueError(f"Invalid target release: {target_release!r}")
    version = int(match.group("version"))
    slug = target_release.replace(":", "-").replace("_", "-")
    pipeline_root = f"artifacts/pipeline/{slug}"
    common = {
        "release": {"id": target_release, "base_release_id": source_release},
        "artifacts": {"run_root": run_root},
    }
    update_yaml(Path("configs/duke_v2_train.yaml"), {
        **common,
        "transfer": {
            "base_checkpoint": checkpoint,
            "expected_checkpoint_epoch": 48 if source_release.startswith("market1501-vit-bnneck:v1") else None,
        },
    })
    if dataset_root:
        for config_path in (Path("configs/duke_v2_train.yaml"), Path("configs/duke_v2_eval.yaml")):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload.setdefault("data", {}).setdefault("location", {})["root"] = dataset_root
            payload.setdefault("evaluation", {})["dataset_pointer"] = dataset_pointer
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    update_yaml(
        Path("configs/duke_v2_eval.yaml"),
        {
            "release": common["release"],
            "artifacts": {"run_root": f"{run_root}/evaluations/duke"},
        },
    )
    update_yaml(
        Path("configs/market1501_v2_eval.yaml"),
        {
            "release": common["release"],
            "artifacts": {"run_root": f"{run_root}/evaluations/market1501"},
        },
    )
    update_yaml(Path("configs/model_validation_v2.yaml"), {"release_id": target_release})
    update_yaml(Path("configs/catastrophic_forgetting.yaml"), {"release_id": target_release})
    params = Path("params.yaml")
    payload = yaml.safe_load(params.read_text(encoding="utf-8"))
    payload["v2_pipeline"].update(
        {
            "release_slug": slug,
            "release_id": target_release,
            "source_root": run_root,
            "model_version": version,
        }
    )
    if dataset_root:
        payload["v2_pipeline"]["duke_root"] = dataset_root
    if dataset_pointer:
        payload["v2_pipeline"]["duke_pointer"] = dataset_pointer
    params.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return {
        "source_release": source_release,
        "target_release": target_release,
        "target_version": version,
        "target_slug": slug,
        "run_root": run_root,
        "pipeline_root": pipeline_root,
        "base_checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an arbitrary vN to vN+1 release workspace.")
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-pointer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(
        source_release=args.source_release,
        target_release=args.target_release,
        checkpoint=args.checkpoint,
        run_root=args.run_root,
        dataset_root=args.dataset_root,
        dataset_pointer=args.dataset_pointer,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
