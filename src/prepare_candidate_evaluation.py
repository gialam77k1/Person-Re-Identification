from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def prepare_config(
    template: Path,
    output: Path,
    *,
    source_release: str,
    target_release: str,
    run_root: Path,
    data_config: Path | None = None,
) -> dict:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation template is not a mapping: {template}")
    result = copy.deepcopy(payload)
    if data_config is not None:
        data_payload = yaml.safe_load(data_config.read_text(encoding="utf-8"))
        if not isinstance(data_payload, dict) or not isinstance(data_payload.get("data"), dict):
            raise ValueError(f"Data config is invalid: {data_config}")
        result["data"] = copy.deepcopy(data_payload["data"])
        pointer = data_payload.get("evaluation", {}).get("dataset_pointer")
        if pointer:
            result.setdefault("evaluation", {})["dataset_pointer"] = pointer
    result.setdefault("release", {})["id"] = target_release
    result["release"]["base_release_id"] = source_release
    result["experiment_name"] = target_release.replace(":", "-")
    result.setdefault("artifacts", {})["run_root"] = str(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a version-neutral candidate evaluation config from a template."
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-config")
    args = parser.parse_args()
    payload = prepare_config(
        Path(args.template),
        Path(args.output),
        source_release=args.source_release,
        target_release=args.target_release,
        run_root=Path(args.run_root),
        data_config=Path(args.data_config) if args.data_config else None,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "release_id": payload["release"]["id"],
                "dataset": payload["data"]["dataset"]["name"],
                "run_root": payload["artifacts"]["run_root"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
