# Person Re-Identification MLOps với ViT-B/16

Hệ thống Person Re-Identification chạy local, sử dụng **ViT-B/16 + BNNeck** và cung cấp
vòng đời MLOps khép kín qua một Control Center: phát hiện drift, phê duyệt retraining,
train model v2 trên self-hosted GPU runner, đánh giá, validation, quản lý artifact và triển khai.

Model production hiện tại là `market1501-vit-bnneck:v1`. Pipeline DukeMTMC-reID tạo model
`market1501-duke-vit-bnneck:v2` dưới alias `candidate`; hệ thống không tự động thay
`champion` hoặc deploy production nếu chưa có phê duyệt.

## Trạng thái model hiện tại

### Model v1 — production/champion

- Dataset: Market-1501.
- Backbone: ViT-B/16 pretrained ImageNet.
- Embedding: 512 chiều với BNNeck.
- Checkpoint tốt nhất: epoch 48.
- Input/output: `1×3×224×224` → `1×512`.
- Preprocessing: giữ tỉ lệ ảnh và padding ở giữa.

| Đánh giá | Rank-1 | Rank-5 | mAP | mINP |
|---|---:|---:|---:|---:|
| Flip test | 91,18% | 96,50% | 80,21% | 51,61% |
| Flip test + re-ranking | **92,07%** | 95,72% | **88,77%** | **74,94%** |

Re-ranking chỉ dùng khi đánh giá hoặc truy hồi offline, không nằm trong ONNX graph phục vụ online.

### Model v2 — candidate pipeline

Model v2 được transfer learning từ checkpoint v1 sang DukeMTMC-reID:

- 16.522 ảnh train, 702 identity.
- Tối đa 30 epoch; AMP, cosine scheduler và warmup 3 epoch.
- Freeze 8 ViT block thấp đến hết epoch 5; full unfreeze từ epoch 6.
- Learning rate backbone bằng `0,05 ×` learning rate head.
- Early stopping theo mAP, patience 5.
- Batch train/eval 16 và distance query chunk 64 để giới hạn bộ nhớ.
- Đánh giá trên cả DukeMTMC-reID và Market-1501 để kiểm tra catastrophic forgetting.

Cấu hình chuẩn nằm tại [`configs/duke_v2_train.yaml`](configs/duke_v2_train.yaml).

## Kiến trúc

```text
Control Center (Next.js, :3001)
  → Control API (FastAPI, :8090)
  → GitHub Actions trên self-hosted GPU runner
  → drift → approval → retrain → evaluate → validate → publish

Inference API (:8080)
  → Triton (:8000) tạo embedding 512 chiều
  → Qdrant (:6333) tìm kiếm cosine top-k

MLflow + PostgreSQL + MinIO   tracking, registry và artifact
Prometheus + Grafana          system/model metrics và drift
DVC + MinIO                   version hóa dataset và model artifact
```

## Yêu cầu

- Linux, Docker Engine và Docker Compose v2.
- NVIDIA GPU, driver và NVIDIA Container Toolkit cho Triton/training.
- Conda với môi trường Python 3.10.
- Self-hosted GitHub Actions runner online trên cùng máy.
- Dataset Market-1501, DukeMTMC-reID và model artifact được quản lý bằng DVC/MinIO.

Tạo môi trường local:

```bash
conda create -n reid python=3.10 -y
conda activate reid
pip install -r requirements.txt
pip install -r requirements-ci.txt
```

## Khởi động nhanh

### 1. Chuẩn bị cấu hình

```bash
cp .env.mlops.example .env.mlops
```

Thay tất cả giá trị `change-me` trong `.env.mlops`. Tối thiểu cần cấu hình:

- credential PostgreSQL, MinIO và Grafana;
- `CONTROL_GITHUB_TOKEN` có quyền Actions read/write cho repository;
- `CONTROL_CALLBACK_TOKEN` là chuỗi ngẫu nhiên dài;
- `CONTROL_WORKFLOW_REF=main`;
- `DOCKER_CONTEXT=default`.

Không commit `.env.mlops`, token hoặc secret. GitHub Actions cần có:

- secret `MLOPS_ENV_FILE`: toàn bộ nội dung `.env.mlops`;
- secret `CONTROL_CALLBACK_TOKEN`: giống giá trị local;
- variable `CONTROL_API_URL=http://localhost:8090`.

### 2. Khởi động stack

```bash
./scripts/mlops_stack.sh validate
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
```

Kiểm tra container:

```bash
docker compose --env-file .env.mlops -f docker-compose.mlops.yml ps
```

### 3. Mở Control Center

Mở **http://localhost:3001**, chọn `DukeMTMC-reID` và bấm **Start MLOps lifecycle**.
Luồng vận hành bình thường không yêu cầu chạy từng script bằng terminal.

```text
Validate dataset
  → Extract current embeddings
  → Detect drift
  → Evaluate model v1 on DukeMTMC-reID
  → WAITING_APPROVAL
  → Approve Retraining
  → Train model v2 (tối đa 30 epoch)
  → Evaluate DukeMTMC-reID + Market-1501
  → Validation/forgetting gate
  → Export ONNX + parity
  → DVC push + MLflow candidate
  → Release workflow: staging → manual production approval
```

Cycle tiếp tục chạy trên self-hosted runner khi đóng trình duyệt. Khi mở lại, UI phục hồi
timeline từ Control API. Approval, promotion và rollback đều được ghi audit log.

## Giao diện và endpoint local

| Thành phần | URL | Ghi chú |
|---|---|---|
| Control Center | http://localhost:3001 | Vận hành toàn bộ lifecycle |
| Control API | http://localhost:8090/docs | API điều phối và audit |
| Inference API | http://localhost:8080/docs | Swagger cho `/embed` và `/search` |
| MLflow | http://localhost:5000 | Runs, artifacts và Model Registry |
| Grafana | http://localhost:3000 | Monitoring và drift dashboards |
| Prometheus | http://localhost:9090 | Metrics và alert rules |
| MinIO Console | http://localhost:9001 | MLflow/DVC object storage |
| Qdrant Dashboard | http://localhost:6333/dashboard | Collection và vector |
| Triton metadata | http://localhost:8000/v2 | Server name, version và extensions |

`http://localhost:8000/v2/health/ready` là health endpoint của Triton. Khi service khỏe,
endpoint này trả HTTP `200` với body rỗng; trang trắng trong trình duyệt là hành vi bình thường.

Kiểm tra nhanh:

```bash
curl --fail http://localhost:8090/health
curl --fail http://localhost:8080/ready
curl --fail http://localhost:8000/v2/health/ready
curl --silent http://localhost:8000/v2 | jq .
```

## Inference

API nhận JPG/PNG, preprocess giống training, gọi Triton, L2-normalize embedding và tìm top-k
trên Qdrant.

```bash
curl --fail http://localhost:8080/model-info

curl --fail -X POST 'http://localhost:8080/search?top_k=5' \
  -F 'image=@datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg;type=image/jpeg'
```

Response `/search` chứa model release, person ID, camera ID, similarity score và định danh ảnh nguồn.

## Train và đánh giá thủ công

Control Center là cách vận hành khuyến nghị. Các lệnh dưới đây dành cho phát triển hoặc chẩn đoán.

Train Market-1501:

```bash
./scripts/train_local.sh --check
./scripts/train_local.sh market1501-vit-bnneck-local
```

Preflight và train DukeMTMC-reID v2:

```bash
./scripts/train_duke_v2_local.sh --check
./scripts/train_duke_v2_local.sh
```

Output v2 được tách khỏi v1 tại:

```text
artifacts/duke/market1501-duke-vit-bnneck-v2/
├── checkpoints/best_model.pth
├── checkpoints/last_model.pth
├── logs/effective_config.json
├── logs/mlflow_run.json
├── logs/transfer_report.json
└── metrics/metrics_v2.json
```

Không tạo trước hoặc ghi đè release directory đang được một run khác sử dụng.

## DVC, release và validation

Chạy pipeline tái lập:

```bash
./scripts/dvc_local.sh status
./scripts/dvc_local.sh repro
./scripts/dvc_local.sh push
```

Release gate kiểm tra quality metrics, catastrophic forgetting, embedding 512 chiều hữu hạn,
L2 normalization, PyTorch–ONNX cosine parity tối thiểu `0,9999`, max absolute error tối đa
`0,0001`, preprocessing contract, checksum và lineage Git/DVC/MLflow.

Model FAIL vẫn được giữ trong MLflow để điều tra nhưng không được gán alias `candidate`,
`champion` hoặc `production`. V1 không bị ghi đè hoặc xóa khi tạo v2.

## Kiểm thử

Chạy cùng bộ kiểm tra CI:

```bash
conda activate reid
./scripts/ci.sh
```

Kiểm thử deployment:

```bash
./scripts/test_deployment.sh unit
./scripts/test_deployment.sh integration
./scripts/test_deployment.sh evaluate
./scripts/test_deployment.sh quality
./scripts/test_deployment.sh benchmark
./scripts/test_deployment.sh persistence
```

Pull request phải qua lint, unit tests, repository validation và API image smoke test. Production
luôn cần manual approval và có thể rollback đồng bộ API image, Triton model và Qdrant collection.

## Cấu trúc repository

```text
configs/                 Training, evaluation và validation policy
src/                     Model, training, evaluation, API và Control API
control-ui/              Next.js MLOps Control Center
scripts/                 Entry point local, CI, DVC và deployment
.github/workflows/       CI, drift, retraining, release và rollback
model_releases/          Manifest, contract và checksum nhỏ theo release
monitoring/              Prometheus, Grafana và Alertmanager configuration
tests/                   Unit, contract và deployment tests
docker-compose.mlops.yml Stack MLOps local
dvc.yaml                 Pipeline tái lập v1
dvc.yaml (stage v2_*)    Release pipeline v2 với output tách biệt
```

Checkpoint, ONNX external weights, embedding, dataset, MLflow artifact và Triton model repository
không được commit vào Git. Git chỉ lưu code, config, DVC pointer, manifest, contract và checksum.

## Tài liệu chi tiết

- [MLOps Control Center](docs/MLOPS_CONTROL_CENTER.md)
- [Hạ tầng MLOps local](docs/MLOPS_LOCAL.md)
- [DukeMTMC-reID v2 training](docs/DUKE_V2_TRAINING.md)
- [Cross-domain drift](docs/DUKE_CROSS_DOMAIN.md)
- [DVC v2 release pipeline](docs/DVC_V2_PIPELINE.md)
- [MLflow tracking và registry](docs/MLFLOW_TRACKING.md)
- [Model validation gate](docs/MODEL_VALIDATION_GATE.md)
- [FastAPI inference](docs/FASTAPI_INFERENCE.md)
- [Deployment testing](docs/DEPLOYMENT_TESTING.md)
- [CI/CD](docs/CI_CD.md)
- [Monitoring](docs/MONITORING.md)

## Model v1 release

- Release ID: `market1501-vit-bnneck:v1`
- Git tag: `model-v1.0.0`
- Source commit trong manifest: `3d4d97e`
- MLflow registered model: `market1501-vit-bnneck`, version `1`, alias `champion`

Xác minh release:

```bash
conda run -n reid python src/verify_model_release.py
```

Lineage và checksum nằm tại
[`model_releases/market1501-vit-bnneck/v1/model_manifest.json`](model_releases/market1501-vit-bnneck/v1/model_manifest.json).
