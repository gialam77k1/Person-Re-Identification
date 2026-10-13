# MLflow Tracking Server và Model Registry

## Những thay đổi đã thực hiện

- Bỏ cấu hình SQLite local, chuyển training client sang `http://localhost:5000`.
- Metadata MLflow được lưu trong PostgreSQL; artifact được lưu trong bucket MinIO `mlflow-artifacts` thông qua MLflow artifact proxy.
- Mỗi training run mới ghi Git commit, DVC dataset revision, effective config, metrics theo epoch, best metrics, checkpoint tốt nhất, thời gian train và thông tin GPU.
- Thêm workflow publish release để gắn ONNX, parity report, preprocessing contract và model manifest vào một release run hoàn chỉnh.
- Đăng ký `market1501-vit-bnneck` version 1 và gắn alias `champion`.
- Thêm verifier kiểm tra metrics, artifact, Registry lineage và tải thử artifact từ MinIO.

## Khởi động và kiểm tra hạ tầng

```bash
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
```

MLflow UI: `http://localhost:5000`

## Training run mới

```bash
./scripts/train_local.sh <run-slug-moi>
```

Training sẽ dừng ngay nếu không kết nối được tracking server, thay vì âm thầm ghi vào SQLite local. Run lưu `mlflow_run.json` trong thư mục log để các bước hậu xử lý có thể truy lại run ID.

Các thông tin được ghi tự động:

- Tags: `git.commit`, `mlflow.source.git.commit`, `dvc.dataset_revision`, loại run và dataset.
- Metrics: Rank-1, Rank-5, mAP, mINP theo epoch; best metrics và tổng thời gian train.
- Artifact: config gốc, effective config, metadata phần cứng, metrics JSON và checkpoint tốt nhất.

Sau khi export ONNX, kiểm tra parity và tạo manifest, gắn các artifact hậu xử lý vào chính training run bằng metadata đã lưu:

```bash
conda run --no-capture-output -n reid python src/publish_model_release.py \
  --manifest model_releases/<model>/<version>/model_manifest.json \
  --run-metadata artifacts/<training-run>/logs/mlflow_run.json
```

Chế độ này mở lại đúng run ID, bổ sung ONNX, parity report, preprocessing contract và model manifest, sau đó mới tạo Model Registry version. Vì vậy lineage của Registry vẫn trỏ về training run gốc.

## Publish model v1

Lệnh này idempotent theo `release.id`: chạy lại không tạo thêm release run hoặc model version cho cùng release.

```bash
conda run --no-capture-output -n reid \
  python src/publish_model_release.py
```

Kiểm tra Registry và tải thử artifact:

```bash
conda run --no-capture-output -n reid \
  python src/verify_mlflow_release.py
```

## Kết quả nghiệm thu ngày 2026-10-13

```text
Run ID:           724f910da5684749bcef271f7f7e8d18
Run status:       FINISHED
Registered model: market1501-vit-bnneck
Version:          1
Alias:            champion
Release:          market1501-vit-bnneck:v1
Git commit:       3d4d97e240ad11b98f222774dc352f88697a9647
DVC revision:     0d61c823b34177c53e9e2161db8257b3.dir
```

Kết quả verifier:

- Có đủ Rank-1, Rank-5, mAP, mINP và thời gian train.
- Có đủ config, checkpoint, ONNX graph/external weights, parity report, preprocessing contract và model manifest.
- Registry version và run có cùng Git commit, DVC revision và release ID.
- Parity report được tải lại thành công qua MLflow từ MinIO.

Run v1 là bản import của lần train cũ. Thời gian train `5.208,42` giây được ước lượng từ timestamp của effective config và train log. GPU của lần train cũ chưa được ghi, vì vậy release run đánh dấu rõ thông tin phần cứng hiện tại là từ máy publish. Các training run mới ghi trực tiếp thời gian và GPU thực tế.
