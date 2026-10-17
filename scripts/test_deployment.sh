#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-reid}"
API_URL="${DEPLOYMENT_API_URL:-http://localhost:8080}"
SAMPLE_IMAGE="${DEPLOYMENT_SAMPLE_IMAGE:-${PROJECT_ROOT}/datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg}"
PIPELINE_ROOT="${PROJECT_ROOT}/artifacts/pipeline/market1501-vit-bnneck-v1-dvc"

unit_tests() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m unittest discover -v tests
}

integration_tests() {
  RUN_DEPLOYMENT_INTEGRATION=1 DEPLOYMENT_API_URL="${API_URL}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m unittest -v tests.integration.test_deployment_api
}

evaluate_market1501() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python src/evaluate_deployment_retrieval.py \
    --server-url http://localhost:8000 \
    --workers "${DEPLOYMENT_EVAL_WORKERS:-8}" \
    --set runtime.run_slug=market1501-vit-bnneck-v1-dvc-deployment \
    --set artifacts.run_root="${PIPELINE_ROOT}/deployment"
}

quality_gate() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python src/validate_deployment_quality.py \
    --offline-metrics "${PIPELINE_ROOT}/evaluate/metrics/evaluation_latest.json" \
    --deployment-metrics "${PIPELINE_ROOT}/deployment/metrics/deployment_retrieval_latest.json" \
    --rank1-max-drop 0.005 \
    --embedding-dim 512 \
    --output "${PIPELINE_ROOT}/deployment/metrics/deployment_quality_gate.json"
}

benchmark() {
  [[ -f "${SAMPLE_IMAGE}" ]] || {
    echo "Thiếu ảnh benchmark: ${SAMPLE_IMAGE}" >&2
    exit 1
  }
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python src/benchmark_inference_api.py \
    --api-url "${API_URL}" \
    --endpoint search \
    --image-path "${SAMPLE_IMAGE}" \
    --iterations "${DEPLOYMENT_BENCHMARK_ITERATIONS:-100}" \
    --warmup "${DEPLOYMENT_BENCHMARK_WARMUP:-10}" \
    --workers "${DEPLOYMENT_BENCHMARK_WORKERS:-1}" \
    --top-k 5 \
    --output "${PIPELINE_ROOT}/deployment/metrics/api_latency_benchmark.json"
}

persistence() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/verify_qdrant_persistence.py \
    --output "${PIPELINE_ROOT}/deployment/metrics/qdrant_persistence.json"
}

usage() {
  echo "Usage: $0 {unit|integration|evaluate|quality|benchmark|persistence|all}"
}

cd "${PROJECT_ROOT}"
case "${1:-}" in
  unit) unit_tests ;;
  integration) integration_tests ;;
  evaluate) evaluate_market1501 ;;
  quality) quality_gate ;;
  benchmark) benchmark ;;
  persistence) persistence ;;
  all)
    unit_tests
    integration_tests
    evaluate_market1501
    quality_gate
    benchmark
    persistence
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
