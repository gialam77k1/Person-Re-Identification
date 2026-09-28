#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
  RUN_SLUG="market1501-vit-bnneck-local"
else
  RUN_SLUG="${1:-market1501-vit-bnneck-local}"
fi
CONDA_ENV="${REID_CONDA_ENV:-reid}"
RUN_ROOT="artifacts/market1501/${RUN_SLUG}"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Không tìm thấy conda trong PATH." >&2
  exit 1
fi

conda run -n "${CONDA_ENV}" python -c \
  "import shutil, torch; from pathlib import Path; assert torch.cuda.is_available(), 'CUDA không khả dụng'; assert Path('datasets/Market-1501-v15.09.15/bounding_box_train').is_dir(), 'Thiếu Market-1501'; free=shutil.disk_usage('.').free; assert free >= 6 * 1024**3, f'Cần ít nhất 6 GiB trống, hiện còn {free / 1024**3:.1f} GiB'; print('GPU:', torch.cuda.get_device_name(0)); print('Disk trống:', f'{free / 1024**3:.1f} GiB')"

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "Local training preflight: OK"
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run root đã tồn tại: ${RUN_ROOT}" >&2
  echo "Hãy chọn RUN_SLUG khác để không ghi đè artifact." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python src/train.py \
  --config configs/dadnet.yaml \
  --set data.dataset.name=market1501 \
  --set data.location.root=datasets/Market-1501-v15.09.15 \
  --set evaluation.use_rerank=false \
  --set evaluation.flip_test=false \
  --set logging.enable_mlflow=true \
  --set runtime.run_slug="${RUN_SLUG}" \
  --set artifacts.run_root="${RUN_ROOT}"
