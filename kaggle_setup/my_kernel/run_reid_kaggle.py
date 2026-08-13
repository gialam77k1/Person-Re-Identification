from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = root / "job-config.json"
    job_config = json.loads(config_path.read_text(encoding="utf-8"))

    overrides = list(job_config.get("overrides", []))
    dataset_name = str(job_config["dataset_name"])
    dataset_root = str(job_config["kaggle_dataset_root"])
    if not any(item.startswith("data.dataset.name=") for item in overrides):
        overrides.append(f"data.dataset.name={dataset_name}")
    if not any(item.startswith("data.location.root=") for item in overrides):
        overrides.append(f"data.location.root={dataset_root}")

    command = [
        sys.executable,
        "src/train.py",
        "--config",
        str(job_config.get("config_path", "configs/dadnet.yaml")),
    ]
    for override in overrides:
        command.extend(["--set", override])

    print("Running Kaggle training command:")
    print(" ".join(command))
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
