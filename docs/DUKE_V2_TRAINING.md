# Retrain model v2 trên DukeMTMC-reID

Tài liệu này ghi lại thay đổi so với model v1 và cách chạy retraining local sau khi drift gate
được phê duyệt. Dataset, checkpoint và output lớn không được commit lên Git.

## Những gì đã thay đổi

- Release mới: `market1501-duke-vit-bnneck:v2`.
- Điểm khởi tạo: checkpoint model v1 tại epoch 48, không tải lại ImageNet weights.
- Dataset train: DukeMTMC-reID gồm 16.522 ảnh, 702 person ID và 8 camera.
- Backbone, embedding projection và BNNeck được kế thừa từ v1.
- Classifier chính và ba auxiliary classifier của Market-1501 bị loại khỏi checkpoint; code tạo
  head mới có 702 outputs theo nhãn DukeMTMC-reID.
- Epoch 1–5: freeze patch projection, positional/class token và 8 block ViT thấp; 4 block trên
  cùng ReID heads vẫn được train.
- Từ epoch 6: unfreeze toàn bộ model. Learning rate backbone bằng `0,05 ×` learning rate head.
- Tối đa 50 epoch, warmup 3 epoch, cosine scheduler và early stopping theo mAP với patience 5.
- Batch train vật lý giữ ở 16; eval batch 16 và distance query batch 64 để giới hạn cả CUDA lẫn host RAM.
- Tám block ViT thấp được giữ freeze đến hết epoch 5; từ epoch 6 mới full-unfreeze với backbone LR bằng `0,05 ×` head LR để giảm catastrophic forgetting trên Market-1501.
- Evaluation trong vòng train không dùng re-ranking để tránh cấp phát ma trận rất lớn của DukeMTMC-reID.

## Gate bắt buộc

Ghi nhận phê duyệt retraining sau khi đã xem full evaluation và drift report:

```bash
REQUIRE_APPROVED=1 ./scripts/retraining_decision.sh gialam77k1
```

Output phải là:

```text
Retraining decision: RETRAIN_APPROVED
```

Decision report được giữ trong artifact local, không đưa lên Git:

```text
artifacts/cross-domain/market1501-v1-on-duke/retraining/decision.json
```

## Kiểm tra trước khi train

```bash
./scripts/train_duke_v2_local.sh --check
```

Preflight từ chối chạy nếu thiếu GPU CUDA, checkpoint v1, dataset, dung lượng đĩa, approval hoặc
nếu số ảnh/ID không đúng contract. Số epoch trong decision phải khớp cấu hình 50 epoch; MLflow
tracking server phải healthy và DVC remote phải có dataset cùng checkpoint. Checkpoint được nạp thử
trong training và bắt buộc phải là epoch 48; classifier cũ luôn bị reset.

## Chạy training local

```bash
./scripts/train_duke_v2_local.sh
```

Output được tách riêng và script không ghi đè run đã tồn tại:

```text
artifacts/duke/market1501-duke-vit-bnneck-v2/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
├── logs/
│   ├── effective_config.json
│   ├── mlflow_run.json
│   ├── train.log
│   └── transfer_report.json
└── metrics/
    └── metrics_v2.json
```

Theo dõi run tại MLflow UI `http://localhost:5000`. Run log release ID, base release, Git commit,
DVC revision, checkpoint nguồn, trạng thái freeze từng epoch, learning rate, Rank-1/5, mAP, mINP,
thời gian train và thông tin GPU.

## Sau khi training xong

Không tự động promote v2. Cần tiếp tục evaluate đầy đủ, so sánh v1/v2, export ONNX, kiểm tra parity,
chạy validation gate, đóng gói Triton và deploy staging trước khi đổi alias `champion`.
