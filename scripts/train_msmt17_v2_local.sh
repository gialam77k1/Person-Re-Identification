#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="configs/msmt17_v2_train.yaml"
DECISION_PATH="artifacts/cross-domain/market1501-v1-on-msmt17/retraining/decision.json"
CHECKPOINT_PATH="artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth"
DATASET_PATH="datasets/MSMT17_V1"
RUN_ROOT="artifacts/msmt17/market1501-msmt17-vit-bnneck-v2"
CONDA_ENV="${REID_CONDA_ENV:-reid}"
CHECK_ONLY=false

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
elif [[ -n "${1:-}" ]]; then
  echo "Cách dùng: $0 [--check]" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Không tìm thấy conda trong PATH." >&2
  exit 1
fi

conda run -n "${CONDA_ENV}" python - "${DECISION_PATH}" "${CHECKPOINT_PATH}" "${DATASET_PATH}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import torch

decision_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
dataset_path = Path(sys.argv[3])

assert torch.cuda.is_available(), "CUDA không khả dụng"
assert decision_path.is_file(), f"Thiếu retraining decision: {decision_path}"
decision = json.loads(decision_path.read_text(encoding="utf-8"))
assert decision.get("decision") == "RETRAIN_APPROVED", "Retraining gate chưa được phê duyệt"
assert checkpoint_path.is_file(), f"Thiếu checkpoint v1: {checkpoint_path}"
assert dataset_path.is_dir(), f"Thiếu MSMT17: {dataset_path}"

train_lines = [
    line for line in (dataset_path / "list_train.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(train_lines) == 30248, f"MSMT17 train phải có 30248 ảnh, nhận {len(train_lines)}"
train_ids = {int(line.split()[1]) for line in train_lines}
assert len(train_ids) == 1041, f"MSMT17 train phải có 1041 ID, nhận {len(train_ids)}"

free = shutil.disk_usage(".").free
assert free >= 12 * 1024**3, f"Cần ít nhất 12 GiB trống, hiện còn {free / 1024**3:.1f} GiB"
print("GPU:", torch.cuda.get_device_name(0))
print("MSMT17 train:", len(train_lines), "ảnh /", len(train_ids), "ID")
print("Retraining gate:", decision["decision"])
print("Disk trống:", f"{free / 1024**3:.1f} GiB")
PY

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "MSMT17 v2 training preflight: OK"
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run root đã tồn tại: ${RUN_ROOT}" >&2
  echo "Không ghi đè artifact v2; hãy lưu hoặc đổi tên run hiện tại trước." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/train.py --config "${CONFIG_PATH}"
