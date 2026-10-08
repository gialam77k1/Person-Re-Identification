# Vận hành MLOps infrastructure local

Stack local sử dụng một Docker Compose chung cho PostgreSQL, MinIO, MLflow, Qdrant và Triton.

## Thành phần và version

| Thành phần | Image | Vai trò |
|---|---|---|
| PostgreSQL | `postgres:16.4-bookworm` | Backend store cho MLflow |
| MinIO | `bitnamilegacy/minio:2025.7.23-debian-12-r5` | Object storage local |
| MLflow | `ghcr.io/mlflow/mlflow:v3.10.0` | Tracking Server và Model Registry |
| Qdrant | `qdrant/qdrant:v1.15.4` | Vector database |
| Triton | `nvcr.io/nvidia/tritonserver:24.08-py3` | ONNX inference server |

Image MLflow nội bộ cài thêm `boto3` và `psycopg2-binary` để kết nối MinIO và PostgreSQL. Job `minio-init` dùng chính image này để tạo bucket, không phụ thuộc MinIO Client image. MinIO Community Edition đã ngừng phát hành image mới từ upstream, vì vậy stack local pin bản legacy công khai của Bitnami thay vì dùng tag `latest` hoặc một registry yêu cầu đăng nhập.

## Chuẩn bị

Yêu cầu:

- Docker Engine và Docker Compose.
- NVIDIA driver và NVIDIA Container Toolkit.
- Triton model repository v1 đã được tạo.
- Các port `5000`, `5432`, `6333`, `6334`, `8000`, `8001`, `8002`, `9000`, `9001` chưa bị chiếm dụng.

Tạo file cấu hình local:

```bash
cp .env.mlops.example .env.mlops
```

Đổi tất cả giá trị `change-me` trong `.env.mlops`. Giữ `AWS_ACCESS_KEY_ID` và `AWS_SECRET_ACCESS_KEY` đồng bộ với tài khoản MinIO tương ứng. File `.env.mlops` đã được Git ignore.

Kiểm tra cấu hình và model artifact:

```bash
./scripts/mlops_stack.sh validate
```

## Khởi động

```bash
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
```

Hoặc chạy Docker Compose trực tiếp:

```bash
docker compose --env-file .env.mlops -f docker-compose.mlops.yml up -d --build
docker compose --env-file .env.mlops -f docker-compose.mlops.yml ps
```

Các địa chỉ local:

- MLflow UI: `http://localhost:5000`.
- MinIO API: `http://localhost:9000`.
- MinIO Console: `http://localhost:9001`.
- Qdrant HTTP: `http://localhost:6333`.
- Triton HTTP: `http://localhost:8000`.
- Triton metrics: `http://localhost:8002/metrics`.

Job `minio-init` tự tạo hai bucket private và có thể chạy lại an toàn:

- `mlflow-artifacts`.
- `dvc-storage`.

## Kiểm tra và xem log

```bash
./scripts/mlops_stack.sh status
./scripts/mlops_stack.sh verify
./scripts/mlops_stack.sh logs
./scripts/mlops_stack.sh logs mlflow
```

Lệnh `verify` kiểm tra PostgreSQL, bốn HTTP health endpoint và bảo đảm hai MinIO bucket tồn tại.

## Dừng stack

```bash
./scripts/mlops_stack.sh down
```

Lệnh này không xóa volume. Không thêm `--volumes` nếu chưa chủ động sao lưu dữ liệu.

Dữ liệu được giữ ở:

- Named volume `reid-postgres-data`.
- Named volume `reid-minio-data`.
- Bind mount `artifacts/qdrant/storage` của Qdrant.
- Triton đọc model repository bằng read-only bind mount.

## Kết quả nghiệm thu ngày 2026-10-08

- PostgreSQL, MinIO, MLflow, Qdrant và Triton đều báo `healthy`.
- MLflow khởi tạo 44 bảng metadata trong PostgreSQL.
- Hai bucket private `mlflow-artifacts` và `dvc-storage` tồn tại; chạy lại job khởi tạo không tạo trùng bucket.
- Qdrant collection `reid_reference_v1` giữ nguyên trạng thái `green`, vector size 512, cosine distance và 12.936 points trước/sau restart.
- Triton readiness endpoint trả thành công với model repository v1.
- Named volume PostgreSQL/MinIO và Qdrant bind mount vẫn còn nguyên sau restart.

## Xử lý lỗi thường gặp

### Triton không khởi động

Kiểm tra GPU và NVIDIA Container Toolkit:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

Kiểm tra model repository có đủ:

```text
reid_embedding/config.pbtxt
reid_embedding/1/model.onnx
reid_embedding/1/model_embedding.onnx.data
```

### MLflow không kết nối PostgreSQL hoặc MinIO

```bash
./scripts/mlops_stack.sh logs postgres
./scripts/mlops_stack.sh logs minio
./scripts/mlops_stack.sh logs mlflow
```

Xác nhận credential AWS trong `.env.mlops` khớp credential MinIO.

### Qdrant không thấy dữ liệu cũ

Xác nhận `QDRANT_STORAGE_PATH` vẫn trỏ đến `./artifacts/qdrant/storage`. Không tạo collection mới nếu collection `reid_reference_v1` đã tồn tại.
