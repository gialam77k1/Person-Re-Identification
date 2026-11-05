# Các bước cuối để hoàn thành dự án Person Re-Identification

Tài liệu này là checklist thực thi từ trạng thái hiện tại đến khi model v2 được đánh giá, triển khai,
giám sát và có thể rollback. Dataset, checkpoint, ONNX, embedding và report lớn chỉ lưu bằng DVC/
MinIO hoặc MLflow; không commit trực tiếp vào Git.

## Trạng thái hiện tại

- [x] Model v1 đã được đóng băng, đóng gói và có manifest/checksum.
- [x] MLOps local gồm PostgreSQL, MinIO, MLflow, Qdrant, Triton, API, Prometheus và Grafana.
- [x] MSMT17 đã được version hóa bằng DVC và drift gate đã trả về `RETRAIN_APPROVED`.
- [x] Self-hosted runner chạy bằng systemd service và smoke test trên `main` đã PASS.
- [x] Pipeline train MSMT17 v2 đã có trên `main` với tối đa 20 epoch.
- [x] Preflight đã PASS trên RTX 3090: 30.248 ảnh, 1.041 ID, MLflow healthy và DVC đồng bộ.
- [ ] Chưa chạy training v2 chính thức.

## 1. Train model v2

Chạy từ `main` và không tạo trước thư mục output của v2:

```bash
git switch main
git pull --ff-only origin main
./scripts/train_msmt17_v2_local.sh --check
./scripts/train_msmt17_v2_local.sh
```

Cấu hình train:

- Base checkpoint: model v1 tại epoch 48.
- Epoch 1–2: freeze 8 block ViT thấp.
- Từ epoch 3: unfreeze toàn model với learning rate backbone bằng `0,1 ×` learning rate head.
- Tối đa 20 epoch, warmup 2 epoch, cosine scheduler và early stopping theo mAP, patience 3.
- Classifier Market-1501 cũ bị loại bỏ và tạo lại theo 1.041 ID của MSMT17.

Theo dõi run tại `http://localhost:5000`. Training hoàn thành khi có đủ:

```text
artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/
├── checkpoints/best_model.pth
├── checkpoints/last_model.pth
├── logs/effective_config.json
├── logs/mlflow_run.json
├── logs/transfer_report.json
└── metrics/metrics_v2.json
```

Gate: run MLflow ở trạng thái hoàn tất, checkpoint tốt nhất tồn tại, metrics hữu hạn và không có lỗi
CUDA/OOM. Early stopping trước epoch 20 là hợp lệ nếu được ghi rõ trong metrics và MLflow.

## 2. Đánh giá v2 trên hai dataset

Cần bổ sung hai entrypoint đánh giá tái lập trước khi chạy:

- `scripts/evaluate_msmt17_v2.sh`: đánh giá khả năng thích nghi trên MSMT17.
- `scripts/evaluate_market1501_v2.sh`: đánh giá lại trên Market-1501 để đo catastrophic forgetting.

Mỗi report phải có Rank-1, Rank-5, mAP, mINP, số query/gallery, Git commit, DVC revision và checksum
checkpoint. Tạo bảng so sánh tối thiểu:

| Model | Dataset | Rank-1 | Rank-5 | mAP | mINP |
|---|---|---:|---:|---:|---:|
| v1 | MSMT17 | baseline cross-domain | baseline | baseline | baseline |
| v2 | MSMT17 | kết quả mới | kết quả mới | kết quả mới | kết quả mới |
| v1 | Market-1501 | baseline release v1 | baseline | baseline | baseline |
| v2 | Market-1501 | kết quả kiểm tra quên | kết quả mới | kết quả mới | kết quả mới |

Gate: v2 phải cải thiện rõ so với baseline cross-domain v1 trên MSMT17. Ngưỡng suy giảm được chấp
nhận trên Market-1501 phải được ghi thành config trước khi promote, không quyết định thủ công sau khi
đã xem kết quả.

## 3. Tách pipeline release v2 trong DVC

Không chạy `dvc repro` cho v2 ngay ở trạng thái hiện tại. `dvc.yaml` đang dùng
`configs/vit_reid.yaml`, đường dẫn train v1 và `metrics_v1.json`. Cần sửa pipeline để nhận config và
release riêng cho v2, đồng thời không ghi đè artifact v1.

Pipeline v2 cần giữ thứ tự:

```text
validate_data
  → train hoặc import checkpoint v2 đã train local
  → evaluate_msmt17
  → evaluate_market1501
  → export_onnx
  → verify_onnx
  → extract_reference
  → validate_release
  → package_triton
```

Release slug đề xuất:

```text
market1501-msmt17-vit-bnneck-v2
```

Gate: `dvc repro` chỉ chạy stage bị ảnh hưởng; mọi output v1 vẫn còn nguyên; config v2, dataset
revision và Git commit xuất hiện trong manifest.

## 4. Export và validation model v2

Từ checkpoint tốt nhất, tạo:

- ONNX graph và external weights.
- PyTorch–ONNX parity report.
- Preprocessing contract `224×224`, preserve-aspect-ratio và padding.
- Reference embeddings 512 chiều cùng PID, camera ID và source path.
- Model manifest và SHA256 cho toàn bộ artifact bắt buộc.

Validation gate phải kiểm tra:

- Rank-1, mAP và mINP theo ngưỡng v2 đã chốt.
- Embedding đúng 512 chiều, không NaN/Inf và đã L2-normalize.
- Cosine parity tối thiểu `0.9999`.
- Max absolute error tối đa `0.0001`.
- Input/output/preprocessing contract khớp.
- Checksum, Git commit, DVC revision và MLflow run đầy đủ.

Model FAIL vẫn được lưu trong MLflow nhưng không được gán alias `candidate`, `champion` hoặc
`production`.

## 5. Version hóa artifact và đăng ký MLflow

- DVC-add checkpoint v2, ONNX, external weights, reference embeddings, metadata và release report.
- Chạy `./scripts/dvc_local.sh push` và xác nhận remote MinIO đồng bộ.
- Đăng ký model v2 trong MLflow Model Registry với alias ban đầu là `candidate`.
- Liên kết registry version tới Git commit, DVC revision, config, metrics và validation report.

Không đổi alias `champion` ở bước này.

## 6. Đóng gói và deploy staging

- Tạo Triton model repository riêng cho v2; không ghi đè repository v1.
- Tạo Qdrant collection mới, ví dụ `reid_reference_v2`; không xóa `reid_reference_v1`.
- Nạp reference embeddings v2 và kiểm tra số vector/metadata.
- Build API image với tag chứa Git commit và model release.
- Deploy vào environment `staging` bằng release workflow.

Gate staging:

- `/health`, `/ready` và `/model-info` PASS.
- `/model-info` trả đúng release v2, Git commit và preprocessing version.
- API → Triton → Qdrant trả đúng top-k.
- Restart service không làm mất Qdrant data.
- Model sai checksum bị từ chối khởi động.

## 7. Regression test và benchmark deployment

Chạy unit, contract, mock và integration test. Sau đó đo:

- Chất lượng retrieval qua API trên MSMT17 và Market-1501.
- Embedding regression so với offline.
- Latency p50, p95, p99 cho tổng API, Triton và Qdrant.
- Throughput, error rate, GPU, CPU và RAM.

Gate: deployment metrics phải nằm trong ngưỡng đã cấu hình và không được thấp hơn offline vượt mức
cho phép. Lưu report vào MLflow và release directory.

## 8. Promote production và kiểm tra rollback

Chỉ sau manual approval:

1. Chuyển alias v2 từ `candidate` sang `champion`.
2. Deploy đúng API image, Triton release và Qdrant collection v2 lên production.
3. Chạy smoke test production.
4. Thử rollback toàn bộ về image/model/collection v1.
5. Xác nhận v1 hoạt động, sau đó mới chuyển lại v2 nếu cần.

Không xóa artifact hoặc collection v1 sau khi promote v2.

## 9. Monitoring và drift loop

Theo dõi Grafana sau deploy:

- Request count, error rate, throughput và latency.
- Trạng thái API, Triton, Qdrant, GPU, CPU và RAM.
- Model release đang phục vụ, embedding norm và similarity distribution.
- No-match rate, data drift và embedding drift.

Chạy drift report định kỳ. Khi drift vượt ngưỡng, tạo retraining decision mới; không tự động train hoặc
promote nếu chưa có approval.

## 10. Hoàn thiện báo cáo và demo

- Cập nhật README và sơ đồ kiến trúc theo trạng thái cuối.
- Lưu bảng so sánh v1/v2 và giải thích trade-off cross-domain.
- Chụp màn hình MLflow, DVC lineage, GitHub Actions, Grafana và kết quả API.
- Demo luồng: drift → approval → train → validation → staging → promotion → rollback.
- Tạo Git tag/release cuối sau khi production và rollback đều PASS.

## Thứ tự thực hiện ngay

```text
1. Train v2 tối đa 20 epoch
2. Viết và chạy evaluation v2 trên MSMT17 + Market-1501
3. Chốt quality thresholds v2
4. Tách DVC release pipeline v2
5. Export ONNX + parity + reference embeddings
6. Validation gate + DVC push + MLflow candidate
7. Triton/Qdrant staging + deployment tests
8. Promote + rollback test
9. Monitoring + báo cáo + demo
```
