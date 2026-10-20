# Kiểm thử deployment

Giai đoạn 7 chứng minh release `market1501-vit-bnneck:v1-dvc` giữ nguyên contract và chất lượng khi chạy qua FastAPI, Triton và Qdrant.

## Phạm vi kiểm thử

### Unit và mock test

- So sánh preprocessing theo contract với preprocessing đã dùng bởi Triton evaluator ở mức bit-exact.
- Kiểm tra JPG/PNG, MIME, file hỏng, dung lượng, số pixel và giới hạn `top_k`.
- Mock Triton/Qdrant để kiểm tra L2 normalization và mapping PID/camera/source image.
- Kiểm tra quality gate ở cả trường hợp PASS và FAIL.
- Kiểm tra API từ chối khởi động khi SHA256 của ONNX graph hoặc external weights không khớp manifest.

Chạy:

```bash
./scripts/test_deployment.sh unit
```

Integration test bị skip mặc định để unit test không phụ thuộc Docker/GPU.

### Integration API → Triton → Qdrant

```bash
./scripts/mlops_stack.sh up
./scripts/test_deployment.sh integration
```

Integration suite gọi HTTP thật và kiểm tra:

- `/health`, `/ready` và `/model-info`.
- Ba lần inference liên tiếp đều trả 512 số finite, norm bằng 1.
- Embedding của ảnh gradient deterministic đạt cosine tối thiểu `0,9999` và max absolute error tối đa `0,0001` so với fixture của release.
- `/search?top_k=5` trả đúng 5 result có PID, camera ID, similarity score và source image.

Fixture được lưu tại `tests/fixtures/deployment_embedding_v1.json` dưới dạng float32 base64; không cần commit ảnh binary.

### Full Market-1501 evaluation

```bash
./scripts/test_deployment.sh evaluate
./scripts/test_deployment.sh quality
```

Evaluation dùng toàn bộ 3.368 query và 15.913 gallery, 8 worker gọi Triton HTTP. Online protocol dùng letterbox, L2 normalization, cosine distance, không flip test và không re-ranking.

Quality gate dùng `rank1_base` thay vì metric re-ranking để tránh so hai retrieval protocol khác nhau:

```text
rank1_drop = offline_rank1_base - deployment_rank1
PASS khi rank1_drop <= 0.005
```

Gate cũng bắt buộc dataset/query count khớp, embedding 512 chiều, toàn bộ finite và đã L2 normalize.

### Benchmark latency

```bash
./scripts/test_deployment.sh benchmark
```

Mặc định đo 10 warmup + 100 request `/search?top_k=5`, concurrency 1. Có thể đổi bằng:

```bash
DEPLOYMENT_BENCHMARK_ITERATIONS=200 \
DEPLOYMENT_BENCHMARK_WARMUP=20 \
DEPLOYMENT_BENCHMARK_WORKERS=4 \
./scripts/test_deployment.sh benchmark
```

### Qdrant persistence

```bash
./scripts/test_deployment.sh persistence
```

Script chụp collection metadata và SHA256 của một point, restart riêng container Qdrant, đợi health endpoint rồi đối chiếu lại. Không xóa container, volume hoặc collection.

### Chạy toàn bộ

```bash
./scripts/test_deployment.sh all
```

Lệnh này mất khoảng 20 phút trên máy nghiệm thu vì luôn chạy full Market-1501 evaluation.

## Artifact report

Các report runtime nằm trong thư mục DVC pipeline đã được Git ignore:

```text
artifacts/pipeline/market1501-vit-bnneck-v1-dvc/deployment/metrics/
├── deployment_retrieval_latest.json
├── deployment_quality_gate.json
├── api_latency_benchmark.json
└── qdrant_persistence.json
```

## Kết quả nghiệm thu ngày 2026-10-17

### Test

- 17 unit/API/mock test PASS.
- 3 integration test thật PASS.
- Embedding regression chạy ba lần ổn định trong ngưỡng parity.
- Checksum tampering test PASS: model sai checksum bị từ chối ngay khi tạo inference service.

### Chất lượng deployment

| Metric | Offline base | Triton deployment |
|---|---:|---:|
| Rank-1 | 91,1817% | **91,3599%** |
| Rank-5 | 96,4964% | 96,3480% |
| mAP | 80,2097% | 80,0992% |
| mINP | 51,6134% | 51,2075% |

- Rank-1 drop: `-0,0017815`, tức deployment tăng khoảng `0,178 điểm %` so với offline base.
- Ngưỡng cho phép: giảm tối đa `0,005` tương đương `0,5 điểm %`.
- Quality gate: **PASS**, 6/6 check.
- 3.368 query và 15.913 gallery đều sinh embedding 512 chiều, finite và L2-normalized.

### Latency local

Đo `/search?top_k=5`, 10 warmup, 100 request tuần tự:

| Metric | Kết quả |
|---|---:|
| p50 | 86,97 ms |
| p95 | 92,10 ms |
| p99 | 97,11 ms |
| Mean | 87,55 ms |
| Throughput | 11,42 request/s |
| Lỗi | 0/100 |

Các số này là baseline trên máy local hiện tại, không phải SLA cho phần cứng khác.

### Persistence

- Qdrant trước/sau restart đều `green`.
- Point count giữ nguyên `12.936`.
- Vector contract giữ nguyên 512 chiều, cosine distance.
- Sample point SHA256 trước/sau giống nhau: `b058eb9566d2d9905109e9960fc793851e35d3aaa9acf3d99433aa0e600ff221`.
- Persistence test: **PASS**.
