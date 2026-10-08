# Kế hoạch triển khai MLOps

Tài liệu này mô tả lộ trình đưa model `market1501-vit-bnneck:v1` từ trạng thái đã đóng băng sang một hệ thống MLOps có khả năng tái lập, triển khai, giám sát và rollback.

## Trạng thái hiện tại

- Model production: ViT-B/16 + BNNeck.
- Release ID: `market1501-vit-bnneck:v1`.
- Checkpoint tốt nhất: epoch 48.
- Đã export ONNX và kiểm tra PyTorch–ONNX parity.
- Đã trích xuất 12.936 reference embeddings.
- Đã đóng gói Triton model repository.
- Đã nạp embeddings vào Qdrant.
- Đã có model manifest, preprocessing contract và SHA256.

Không cần train lại model v1 ở thời điểm bắt đầu kế hoạch này.

## Giai đoạn 1 — Dựng MLOps infrastructure local ✅

Trạng thái: hoàn tất ngày 2026-10-08.

Phần triển khai được chia thành ba commit độc lập:

1. PostgreSQL, MinIO và MLflow core.
2. Tích hợp Qdrant và Triton vào stack chung.
3. Script vận hành, kiểm tra health và tài liệu local.

Kết quả nghiệm thu:

- PostgreSQL, MinIO, MLflow, Qdrant và Triton đều healthy.
- MLflow UI và health endpoint hoạt động.
- Hai bucket `mlflow-artifacts` và `dvc-storage` được tạo private và idempotent.
- Qdrant giữ nguyên 12.936 points của collection `reid_reference_v1` sau restart.
- PostgreSQL, MinIO và Qdrant đều giữ nguyên dữ liệu sau kiểm tra restart.

### Mục tiêu

Có hạ tầng local ổn định để lưu metadata, model artifact và theo dõi thí nghiệm.

### Công việc

- Tạo `docker-compose.mlops.yml`.
- Chạy PostgreSQL làm backend database cho MLflow.
- Chạy MinIO làm object storage.
- Chạy MLflow Tracking Server và Model Registry.
- Tích hợp Qdrant và Triton hiện tại vào stack chung.
- Pin version cho tất cả Docker image.
- Tạo healthcheck cho từng service.
- Tạo `.env.mlops.example`; không commit `.env.mlops` hoặc secret.
- Tạo hai MinIO bucket riêng:
  - `mlflow-artifacts`.
  - `dvc-storage`.

### Tiêu chí hoàn thành

```bash
docker compose -f docker-compose.mlops.yml up -d
docker compose -f docker-compose.mlops.yml ps
```

- PostgreSQL, MinIO, MLflow, Qdrant và Triton đều healthy.
- Truy cập được MLflow UI.
- Dữ liệu PostgreSQL, MinIO và Qdrant không mất sau khi restart container.

### Commit gợi ý

```text
Add local MLOps infrastructure stack
```

## Giai đoạn 2 — Version hóa dữ liệu và artifact bằng DVC

### Mục tiêu

Có thể lấy lại đúng dataset và model artifact ứng với từng Git commit.

### Công việc

- Cài DVC với S3 support.
- Khởi tạo DVC trong repository.
- Cấu hình MinIO làm DVC remote.
- Không commit access key hoặc secret key.
- Đưa các thành phần sau vào DVC:
  - Market-1501.
  - Checkpoint epoch 48.
  - ONNX graph và external weights.
  - Reference embeddings và metadata.
- Thêm DVC hash hoặc dataset version vào model manifest.
- Commit các file `.dvc` và cấu hình DVC không chứa secret.

### Tiêu chí hoàn thành

```bash
dvc status
dvc push
dvc pull
```

- `dvc status` không báo artifact chưa đồng bộ.
- `dvc pull` phục hồi được đầy đủ artifact sau khi xóa bản local thử nghiệm.
- Checksum của model v1 vẫn khớp manifest.

### Commit gợi ý

```text
Version datasets and model artifacts with DVC
```

## Giai đoạn 3 — Chuyển MLflow sang tracking server

### Mục tiêu

Thay MLflow SQLite local bằng tracking server dùng PostgreSQL và MinIO.

### Công việc

Mỗi training run cần log:

- Git commit.
- DVC dataset revision.
- Config đầy đủ.
- Rank-1, Rank-5, mAP và mINP.
- Checkpoint tốt nhất.
- ONNX và parity report.
- Preprocessing contract.
- Model manifest.
- Thời gian train và thông tin GPU.

Đăng ký model v1 vào Model Registry:

```text
Registered model: market1501-vit-bnneck
Version: 1
Alias: champion
Release: market1501-vit-bnneck:v1
```

### Tiêu chí hoàn thành

- Run xuất hiện trong MLflow UI.
- Artifact được lưu và tải từ MinIO.
- Model v1 xuất hiện trong Model Registry.
- Model Registry version truy ngược được về Git commit và DVC revision.

### Commit gợi ý

```text
Integrate MLflow tracking and model registry
```

## Giai đoạn 4 — Tạo pipeline tái lập bằng DVC

### Mục tiêu

Chạy toàn bộ pipeline model bằng một lệnh và chỉ chạy lại stage bị ảnh hưởng.

### Pipeline

```text
validate_data
  → train
  → evaluate
  → export_onnx
  → verify_onnx
  → extract_reference
  → package_triton
  → validate_release
```

### Công việc

- Tạo `dvc.yaml`.
- Khai báo dependencies, parameters, outputs và metrics cho từng stage.
- Tách output theo model release để không ghi đè release cũ.
- Cho phép bỏ qua training khi chỉ thay đổi bước export hoặc serving.

### Tiêu chí hoàn thành

```bash
dvc repro
```

- Pipeline chạy được từ đầu đến cuối.
- DVC không chạy lại các stage không bị ảnh hưởng.
- Output được gắn với đúng dataset version, Git commit và config.

### Commit gợi ý

```text
Add reproducible DVC model pipeline
```

## Giai đoạn 5 — Tạo model validation gate

### Mục tiêu

Ngăn model không đạt chất lượng được promote hoặc deploy.

### Ngưỡng ban đầu

```yaml
rank1_min: 0.91
map_min: 0.80
onnx_cosine_min: 0.9999
onnx_max_abs_error: 0.0001
embedding_dim: 512
```

### Kiểm tra bắt buộc

- Rank-1, mAP và mINP đạt ngưỡng.
- Không có NaN hoặc Inf trong embedding.
- PyTorch–ONNX parity PASS.
- Input/output contract đúng.
- Checksum và manifest đầy đủ.
- Deployment metrics không giảm quá ngưỡng cho phép.

Model không đạt vẫn được log vào MLflow nhưng không được gán alias production hoặc champion.

### Tiêu chí hoàn thành

- Validation gate trả về kết quả PASS/FAIL rõ ràng.
- Báo cáo validation được lưu trong MLflow và model release.
- Chỉ model PASS mới được chuyển sang bước đóng gói/deploy.

### Commit gợi ý

```text
Add model quality validation gate
```

## Giai đoạn 6 — Xây dựng FastAPI inference service

### Mục tiêu

Cung cấp API truy hồi người bằng Triton và Qdrant.

### Endpoint

```text
GET  /health
GET  /ready
GET  /model-info
POST /embed
POST /search
```

### Luồng `/search`

```text
validate image
  → preprocessing
  → Triton inference
  → L2 normalize
  → Qdrant search
  → top-k results
```

Response cần chứa:

- Model release.
- Person ID.
- Camera ID.
- Similarity score.
- Đường dẫn hoặc định danh ảnh nguồn.

### Tiêu chí hoàn thành

- Nhận được ảnh JPG/PNG hợp lệ.
- Từ chối file sai định dạng hoặc quá giới hạn.
- Trả đúng top-k từ Qdrant.
- `/ready` chỉ thành công khi Triton, model và Qdrant collection sẵn sàng.
- `/model-info` trả đúng release ID, Git commit và preprocessing version.

### Commit gợi ý

```text
Add FastAPI person retrieval service
```

## Giai đoạn 7 — Kiểm thử deployment

### Mục tiêu

Chứng minh pipeline triển khai không làm giảm chất lượng model.

### Công việc

- Unit test preprocessing.
- API contract test.
- Mock test Triton và Qdrant.
- Integration test API → Triton → Qdrant.
- Regression test embedding.
- Market-1501 deployment evaluation.
- Benchmark latency p50, p95 và p99.
- Kiểm tra restart service không làm mất Qdrant data.
- Kiểm tra model sai checksum bị từ chối khởi động.

### Tiêu chí hoàn thành

- Rank-1 deployment không giảm quá `0,5%` so với offline.
- Embedding luôn có 512 chiều.
- Không có NaN hoặc Inf.
- Integration test chạy ổn định và tái lập được.

### Commit gợi ý

```text
Add MLOps integration and regression tests
```

## Giai đoạn 8 — CI/CD

### Mục tiêu

Tự động kiểm tra code và kiểm soát việc phát hành model.

### CI cho pull request

- Lint và unit test.
- Validate config và manifest.
- Build API image.
- Smoke test không cần GPU.
- Kiểm tra repository không chứa model lớn hoặc secret.

### Release workflow

```text
model candidate
  → validation gate
  → build image
  → deploy staging
  → smoke test
  → manual approval
  → production
```

Docker image phải được tag bằng Git commit và model release, ví dụ:

```text
reid-api:1ca60d8-model-v1
```

### Tiêu chí hoàn thành

- Pull request lỗi test không được merge.
- Chỉ model qua validation gate mới được deploy staging.
- Production cần approval.
- Có thể rollback về image và model release trước đó.

### Commit gợi ý

```text
Add CI/CD workflows
```

## Giai đoạn 9 — Monitoring

### Mục tiêu

Theo dõi cả sức khỏe hệ thống và hành vi của model.

### System metrics

- Request count và error rate.
- Latency tổng, Triton và Qdrant.
- Throughput.
- GPU, CPU và RAM.
- Trạng thái service và collection size.

### Model metrics

- Model release đang phục vụ.
- Embedding norm.
- Phân bố similarity score.
- Tỷ lệ query không tìm được match.
- Data hoặc embedding drift.

### Công việc

- Tích hợp Prometheus.
- Tạo dashboard Grafana.
- Thêm alert khi service unhealthy, error rate cao hoặc latency vượt ngưỡng.
- Thiết kế drift report chạy định kỳ.

### Commit gợi ý

```text
Add monitoring and model observability stack
```

## Giai đoạn 10 — Prefect orchestration và Kafka

### Prefect

Chỉ triển khai sau khi DVC pipeline đã chạy ổn định.

Prefect chịu trách nhiệm:

- Schedule pipeline.
- Retry stage lỗi.
- Theo dõi lịch sử chạy.
- Gửi thông báo.
- Điều phối worker local hoặc remote.

### Kafka

Chỉ triển khai khi xuất hiện nhu cầu camera stream hoặc xử lý bất đồng bộ, ví dụ:

- Nhiều camera gửi sự kiện đồng thời.
- Cần queue inference.
- Cần replay sự kiện.
- Có nhiều downstream consumer.

Không thêm Kafka nếu hệ thống mới chỉ nhận ảnh qua HTTP và trả top-k trực tiếp.

### Commit gợi ý

```text
Add Prefect pipeline orchestration
```

Kafka nên có commit và giai đoạn thiết kế riêng nếu thực sự được sử dụng.

## Thứ tự thực hiện

- [x] Giai đoạn 1 — MLOps infrastructure local.
- [ ] Giai đoạn 2 — DVC data và artifact versioning.
- [ ] Giai đoạn 3 — MLflow Tracking Server và Model Registry.
- [ ] Giai đoạn 4 — DVC reproducible pipeline.
- [ ] Giai đoạn 5 — Model validation gate.
- [ ] Giai đoạn 6 — FastAPI inference service.
- [ ] Giai đoạn 7 — Deployment tests.
- [ ] Giai đoạn 8 — CI/CD.
- [ ] Giai đoạn 9 — Monitoring.
- [ ] Giai đoạn 10 — Prefect; chỉ thêm Kafka khi có nhu cầu streaming.

## Nguyên tắc triển khai

- Không ghi đè model release v1.
- Không commit dataset, model binary, embeddings hoặc secret vào Git.
- Mỗi release phải truy ngược được về Git commit, config, dataset version và metrics.
- Mỗi giai đoạn phải có test và tiêu chí hoàn thành trước khi chuyển sang giai đoạn sau.
- Ưu tiên hệ thống local chạy ổn định trước khi mở rộng lên cloud hoặc Kubernetes.
