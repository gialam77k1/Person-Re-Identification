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
