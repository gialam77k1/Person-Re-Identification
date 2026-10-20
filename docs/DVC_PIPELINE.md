# Pipeline tái lập bằng DVC

## Mục tiêu

Pipeline biến dataset Market-1501 và cấu hình ViT-B/16 + BNNeck thành một model release có thể kiểm tra nguồn gốc. Toàn bộ luồng được chạy bằng một lệnh:

```bash
./scripts/dvc_local.sh repro
```

Đồ thị hiện tại:

```text
validate_data
  -> train
     -> evaluate
     -> export_onnx -> verify_onnx
     -> extract_reference
  -> validate_release -> package_triton
```

`validate_release` là model validation gate. Chỉ report `PASS` mới cho phép `package_triton` chạy; script đóng gói còn đối chiếu SHA256 để chắc chắn đúng ONNX đã được gate kiểm tra.

## Các stage

| Stage | Vai trò | Output chính |
|---|---|---|
| `validate_data` | Kiểm tra cấu trúc, tên file, PID/camera và số lượng ảnh | `validate/data_manifest.json` |
| `train` | Train ViT, ghi checkpoint, metrics, config và MLflow run | `train/` |
| `evaluate` | Đánh giá flip test + re-ranking | `evaluate/metrics/evaluation_latest.json` |
| `export_onnx` | Export ONNX graph và external weights | `onnx/exports/` |
| `verify_onnx` | So sánh embedding PyTorch và ONNX Runtime | `parity/metrics/onnx_parity.json` |
| `extract_reference` | Trích xuất 12.936 reference embeddings | `reference/embeddings/` |
| `validate_release` | Chạy quality/contract/checksum/deployment gate và tạo manifest | `release/` |
| `package_triton` | Chỉ đóng gói candidate có gate PASS | `triton/model_repository/` |

Mọi output nằm dưới thư mục riêng của release:

```text
artifacts/pipeline/<pipeline.release_slug>/
```

Muốn tạo release mới, đổi đồng thời `pipeline.release_slug` và `pipeline.release_id` trong `params.yaml`. Không dùng lại slug cũ cho một model khác.

## Parameters và phạm vi chạy lại

Các tham số điều phối nằm trong `params.yaml`; cấu hình model/training chi tiết vẫn nằm trong `configs/vit_reid.yaml`.

- Đổi dataset, model hoặc tham số train sẽ chạy lại `train` và các stage phụ thuộc.
- Đổi `evaluate.*` sẽ chạy lại `train` vì các giá trị này cũng tham gia validation trong lúc train, sau đó chạy lại các stage downstream cần thiết.
- Đổi `export.opset` chỉ làm bẩn `export_onnx` và release validation; `train` và `evaluate` không chạy lại.
- Đổi `parity.*` chỉ chạy lại `verify_onnx` và release validation.
- Đổi `validation.*` chạy lại gate và bước đóng gói phía sau, không chạy lại training.
- Đổi `triton.*` chỉ chạy lại `package_triton`.

Có thể xem đồ thị và trạng thái trước khi chạy:

```bash
./scripts/dvc_local.sh dag
./scripts/dvc_local.sh status
./scripts/dvc_local.sh repro --dry
```

## Vận hành

Khởi động MLflow, MinIO và các service local trước nếu stage train cần ghi một run mới:

```bash
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
./scripts/dvc_local.sh repro
./scripts/dvc_local.sh metrics show
./scripts/dvc_local.sh push
```

Khôi phục output trên máy khác:

```bash
cp .env.mlops.example .env.mlops
# Điền credential MinIO local vào .env.mlops.
./scripts/dvc_local.sh pull
./scripts/dvc_local.sh repro
```

`.env.mlops` không được commit. Git chỉ lưu `dvc.yaml`, `dvc.lock`, parameters, code và tài liệu; các output lớn được lưu trong DVC cache/MinIO.

## Bootstrap checkpoint v1

Release đầu tiên của pipeline dùng đúng checkpoint epoch 48 đã được đóng băng trước đó, SHA256 `a54da0c9a9eb2652c4ca192c6a6e9b9fb0c59513900d5401a8ae4edfbcdcfd5d`. Checkpoint, training metrics và effective config được đưa vào DVC cache cho stage `train` thay vì train lại 50 epoch chỉ để chuyển sang cơ chế pipeline.

Đây chỉ là bước bootstrap cho model v1. Khi code train, dataset hoặc tham số liên quan thay đổi, DVC sẽ thực thi thật `src/train.py` và tạo release mới trong thư mục được chỉ định bởi `pipeline.release_slug`.

## Kết quả nghiệm thu ngày 2026-10-17

- `dvc repro` chạy thành công từ `evaluate` đến `validate_release`; `validate_data` và checkpoint v1 bootstrap được lấy từ cache.
- Chạy `dvc repro` lần hai bỏ qua toàn bộ 8 stage và báo pipeline đã cập nhật.
- Kiểm thử tạm đổi riêng `export.opset` từ 18 sang 17 cho thấy `train`, `evaluate` và `extract_reference` đều được bỏ qua; tham số đã được hoàn nguyên về 18 sau kiểm thử.
- Dataset revision: `0d61c823b34177c53e9e2161db8257b3.dir`.
- Release validation hiện dùng model validation gate: 18 check PASS và 8 nhóm artifact có SHA256 đầy đủ.
- Metrics: Rank-1 `0,92072`, Rank-5 `0,95724`, mAP `0,88768`, mINP `0,74940`.
- ONNX parity: max absolute error `1,335144e-5`, minimum cosine similarity `0,99999988`.
