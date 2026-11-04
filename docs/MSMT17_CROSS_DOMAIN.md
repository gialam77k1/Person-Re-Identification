# Đánh giá cross-domain trên MSMT17

Mục tiêu của bước này là đo model v1 đã train bằng Market-1501 khi gặp miền dữ liệu MSMT17,
không dùng nhãn MSMT17 để fine-tune trước khi đo. Đây là baseline deployment để quyết định có
cần retrain hay không, không phải metric thay thế metric Market-1501 của model release.

## Dataset đã version hóa

- Dataset: `MSMT17_V1`.
- DVC pointer: `datasets/MSMT17_V1.dvc`.
- DVC revision: `c8a4c6f18b86490a45f940ab9fb0eb70.dir`.
- Train/query/gallery: 30.248 / 11.659 / 82.161 ảnh.
- Ảnh thật nằm trong DVC/MinIO; Git chỉ chứa pointer và cấu hình.

## Chạy smoke test

Smoke test chỉ xác nhận luồng inference và metric hoạt động, không dùng làm baseline chính thức:

```bash
MAX_QUERIES=100 MAX_GALLERY=2000 ./scripts/evaluate_msmt17_v1.sh
```

## Chạy đầy đủ

```bash
./scripts/mlops_stack.sh up
./scripts/evaluate_msmt17_v1.sh
```

Kết quả mặc định được ghi tại
`artifacts/cross-domain/market1501-v1-on-msmt17/metrics/v1_on_msmt17.json` và log vào
experiment MLflow `market1501-v1-cross-domain-msmt17`. Query/gallery embeddings cùng metadata
được lưu bên cạnh để dùng cho drift report. Evaluator tính distance theo batch query nên không
cấp phát ma trận 11.659 × 82.161 trong RAM.

Report luôn chứa release ID, Git commit, DVC revision, số mẫu, embedding dimension, Rank-1/5/10/20,
mAP, mINP và query coverage. Chỉ report chạy đủ query/gallery mới được dùng làm baseline retraining.

## Phát hiện embedding drift

Luồng drift lấy mẫu query MSMT17 xác định bằng seed `42`, infer bằng đúng release đang phục vụ và
so sánh với reference embeddings Market-1501 của model v1:

```bash
SAMPLE_SIZE=1000 ./scripts/cross_dataset_drift.sh
```

Các kiểm tra gồm centroid cosine distance, standardized mean shift, norm mean delta và KS statistic.
Ngưỡng nằm trong `configs/drift_detection.yaml`. Kết quả `DRIFT` là tín hiệu domain shift cần kết hợp
với metric cross-domain; nó không tự động cho phép retrain hay promote model.

Có thể chạy cùng luồng trên giao diện GitHub tại
`Actions → MSMT17 cross-domain drift → Run workflow`. Workflow dùng self-hosted runner vì cần GPU,
Triton, MinIO/DVC và upload report JSON làm GitHub Actions artifact trong 30 ngày.

Output local mặc định:

```text
artifacts/cross-domain/market1501-v1-on-msmt17/drift/
├── drift_report.json
├── msmt17_query_embeddings.npy
├── msmt17_query_embeddings.jsonl
├── msmt17_query_metadata.json
└── reid_cross_dataset_drift.prom
```

## Decision gate trước khi retrain

Gate chỉ đề xuất retrain khi đồng thời thỏa mãn:

- Evaluation dùng đủ 11.659 query và 82.161 gallery, coverage tối thiểu 99%.
- Embedding 512 chiều, hữu hạn và report có DVC revision.
- Drift report của đúng model release trả về `drift_detected=true`.
- Rank-1 dưới 0,50 hoặc mAP dưới 0,30 theo policy hiện tại.
- Có tên người phê duyệt thủ công dài tối thiểu 3 ký tự.

Chạy lần đầu không truyền người duyệt để lấy đề xuất:

```bash
./scripts/retraining_decision.sh
```

Nếu output là `REVIEW_REQUIRED`, xem lại evaluation/drift report rồi mới ghi nhận phê duyệt:

```bash
REQUIRE_APPROVED=1 ./scripts/retraining_decision.sh gialam77k1
```

Các trạng thái có thể có là `INSUFFICIENT_EVIDENCE`, `NO_RETRAIN`, `REVIEW_REQUIRED` và
`RETRAIN_APPROVED`. Chỉ trạng thái cuối mới cho phép chạy cấu hình retrain MSMT17 10 epoch.
Tên người duyệt trong CLI là audit metadata; khi tự động hóa CI/CD cần kết hợp GitHub Environment
required reviewer để GitHub xác thực danh tính người duyệt.
