# DVC release pipeline cho model v2

Pipeline v2 nằm chung trong `dvc.yaml` nhưng dùng prefix stage `v2_` và output root riêng:

```text
artifacts/pipeline/market1501-duke-vit-bnneck-v2/
```

Pipeline v1 không bị đổi tên hoặc ghi đè. Chạy toàn bộ release v2 bằng:

```bash
./scripts/dvc_local.sh repro v2_package_triton
```

Graph thực thi:

```text
v2_validate_data
  → v2_checkpoint
      ├── v2_evaluate_duke ──────────┐
      ├── v2_evaluate_market1501 ────┤
      ├── v2_export_onnx             │
      │     └── v2_verify_onnx ──────┤
      └── v2_extract_reference ──────┤
                                      ↓
                            v2_validate_release
                                      ↓
                              v2_package_triton
```

`v2_checkpoint` import checkpoint tốt nhất đã được workflow retraining tạo ra. Stage này từ chối
artifact nếu MLflow run chưa `FINISHED`, release ID sai, checkpoint không bắt nguồn từ epoch 48 hoặc
classifier Market-1501 chưa được reset. DVC không tự train lại 50 epoch khi chỉ thay đổi export,
parity, reference hoặc Triton packaging.

Manifest cuối tại `release/model_manifest.json` chứa Git commit của pipeline và training run, DVC
revision riêng cho DukeMTMC-reID và Market-1501, MLflow run ID, SHA256 effective training config, epoch tốt
nhất và checksum của checkpoint, ONNX, preprocessing, embeddings cùng metadata.

Release gate sử dụng policy bất biến `configs/model_validation_v2.yaml`. Ngoài quality threshold của
DukeMTMC-reID và Market-1501, gate đọc trực tiếp file NumPy để xác nhận embedding có shape `N×512`, metadata
thẳng hàng, không có NaN/Inf và norm L2 nằm trong sai số `0.0001`. Báo cáo ONNX phải đạt cosine
`>=0.9999`, max absolute error `<=0.0001`, đồng thời I/O runtime phải khớp preprocessing contract.
Report PASS/FAIL, manifest và policy đều được log vào training run trên MLflow. Gate FAIL vẫn giữ đủ
evidence điều tra nhưng trả exit code khác 0 nên DVC không package Triton và workflow không đăng ký
alias `candidate`.

Kiểm tra graph mà không chạy GPU stage:

```bash
DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache ./scripts/dvc_local.sh dag
```

Sau khi pipeline hoàn thành, workflow gọi script publication duy nhất. Script push toàn bộ dependency
của stage cuối, kiểm tra trạng thái MinIO và tạo evidence có checksum:

```bash
./scripts/publish_v2_artifacts.sh
```

Kết quả `release/dvc_publication.json` chỉ PASS khi `dvc status --cloud --with-deps` sạch, mọi file
trong manifest tồn tại và SHA256 khớp. Publication report cũng được DVC-add/push; các file `.pth`,
`.onnx` và `.npy` không được upload làm GitHub Actions artifact hoặc commit trực tiếp vào Git.
