# Person Re-Identification với ViT-B/16

Repository này chỉ duy trì pipeline tốt nhất hiện tại cho bài toán Person Re-Identification trên Market-1501: **ViT-B/16 + BNNeck**, phát hành với tên `market1501-vit-bnneck:v1`.

## Kết quả hiện tại

Checkpoint tốt nhất được chọn tại epoch 48 trên 3.368 query hợp lệ.

| Chế độ đánh giá | Rank-1 | Rank-5 | mAP | mINP |
|---|---:|---:|---:|---:|
| Flip test | 91,18% | 96,50% | 80,21% | 51,61% |
| Flip test + re-ranking | **92,07%** | 95,72% | **88,77%** | **74,94%** |

Re-ranking chỉ dùng khi đánh giá hoặc truy hồi, không nằm trong graph ONNX phục vụ online.

## Mô hình và cách train

- Backbone: ViT-B/16 pretrained trên ImageNet.
- Embedding: 512 chiều, có BNNeck.
- Nhánh đặc trưng: CLS/global và local branch 4 vùng.
- Loss: ID loss + triplet loss + auxiliary branch loss.
- Fine-tuning theo giai đoạn: freeze backbone trong 5 epoch đầu, sau đó unfreeze và fine-tune toàn bộ mô hình đến tối đa 50 epoch.
- Cosine learning-rate scheduler, warmup 5 epoch, gradient clipping và early stopping.
- Ảnh đầu vào `1×3×224×224`, giữ tỉ lệ ảnh và padding ở giữa.

Cấu hình duy nhất được hỗ trợ là [`configs/vit_reid.yaml`](configs/vit_reid.yaml).

## Cấu trúc chính

```text
configs/vit_reid.yaml                         Cấu hình train/evaluate hiện tại
scripts/train_local.sh                        Kiểm tra môi trường và train local
src/train.py                                  Train ViT ReID
src/evaluate.py                               Đánh giá checkpoint
src/export_onnx.py                            Export ONNX
src/verify_onnx.py                            Kiểm tra PyTorch–ONNX parity
src/extract_reference.py                      Trích xuất reference embeddings
src/prepare_triton_model.py                    Đóng gói Triton model repository
src/qdrant_local.py                            Tạo collection và nạp Qdrant
src/evaluate_deployment_retrieval.py           Đánh giá pipeline đã deploy
model_releases/market1501-vit-bnneck/v1/       Manifest và preprocessing contract
```

## Chuẩn bị môi trường local

```bash
conda create -n reid python=3.10 -y
conda activate reid
pip install -r requirements.txt
```

Đặt Market-1501 theo cấu trúc:

```text
datasets/Market-1501-v15.09.15/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

Kiểm tra CUDA, dataset và dung lượng đĩa trước khi train:

```bash
./scripts/train_local.sh --check
```

## Train

Tạo một run mới bằng tên không trùng với thư mục artifact đã có:

```bash
./scripts/train_local.sh market1501-vit-bnneck-local
```

Hoặc chạy trực tiếp:

```bash
conda run --no-capture-output -n reid python src/train.py \
  --config configs/vit_reid.yaml \
  --set runtime.run_slug=market1501-vit-bnneck-local \
  --set artifacts.run_root=artifacts/market1501/market1501-vit-bnneck-local
```

## Đánh giá checkpoint tốt nhất

```bash
conda run -n reid python src/evaluate.py \
  --config configs/vit_reid.yaml \
  --checkpoint artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth \
  --set runtime.run_slug=market1501-vit-bnneck-v1-eval \
  --set artifacts.run_root=artifacts/market1501/market1501-vit-bnneck-v1-eval
```

`evaluation.flip_test=true` và `evaluation.use_rerank=true` đã được bật trong config để tái tạo bộ metrics cao nhất.

## Chuẩn bị model cho MLOps

### 1. Export ONNX

```bash
conda run -n reid python src/export_onnx.py \
  --config configs/vit_reid.yaml \
  --checkpoint artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth \
  --set runtime.run_slug=market1501-vit-bnneck-v1-onnx \
  --set artifacts.run_root=artifacts/market1501/market1501-vit-bnneck-v1-onnx
```

### 2. Kiểm tra PyTorch–ONNX parity

```bash
conda run -n reid python src/verify_onnx.py \
  --config configs/vit_reid.yaml \
  --checkpoint artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth \
  --onnx-path artifacts/market1501/market1501-vit-bnneck-v1-onnx/exports/model_embedding.onnx \
  --num-samples 8 \
  --set runtime.run_slug=market1501-vit-bnneck-v1-onnx-parity \
  --set artifacts.run_root=artifacts/market1501/market1501-vit-bnneck-v1-onnx-parity
```

Release v1 đã đạt parity: max absolute error `1,335144e-5`, cosine similarity nhỏ nhất `0,99999988`.

### 3. Trích xuất reference embeddings

```bash
conda run -n reid python src/extract_reference.py \
  --config configs/vit_reid.yaml \
  --checkpoint artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth \
  --set runtime.run_slug=market1501-vit-bnneck-v1-reference \
  --set artifacts.run_root=artifacts/market1501/market1501-vit-bnneck-v1-reference
```

### 4. Đóng gói Triton

```bash
conda run -n reid python src/prepare_triton_model.py \
  --onnx-path artifacts/market1501/market1501-vit-bnneck-v1-onnx/exports/model_embedding.onnx \
  --output-root artifacts/triton/market1501-vit-bnneck-v1/model_repository \
  --model-name reid_embedding \
  --model-version 1 \
  --instance-kind KIND_GPU

docker compose -f docker-compose.triton.yml up -d
```

### 5. Nạp embeddings vào Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d

conda run -n reid python src/qdrant_local.py create-collection \
  --collection-name reid_reference_v1 \
  --vector-size 512 \
  --distance Cosine

conda run -n reid python src/qdrant_local.py upsert-reference \
  --collection-name reid_reference_v1 \
  --embeddings-path artifacts/market1501/market1501-vit-bnneck-v1-reference/embeddings/reference_embeddings.npy \
  --pids-path artifacts/market1501/market1501-vit-bnneck-v1-reference/embeddings/reference_pids.npy \
  --camids-path artifacts/market1501/market1501-vit-bnneck-v1-reference/embeddings/reference_camids.npy
```

Reference set hiện tại gồm 12.936 vector.

## Model release v1

- Release ID: `market1501-vit-bnneck:v1`
- Git tag: `model-v1.0.0`
- Commit ghi trong manifest: `3d4d97e`
- Checkpoint epoch: 48
- Contract đầu vào/đầu ra: `1×3×224×224` → `1×512`

Kiểm tra checksum của checkpoint, ONNX, external weights và preprocessing contract:

```bash
conda run -n reid python src/verify_model_release.py
```

Thông tin truy vết đầy đủ nằm trong [`model_releases/market1501-vit-bnneck/v1/model_manifest.json`](model_releases/market1501-vit-bnneck/v1/model_manifest.json).

## Quy ước artifact

Checkpoint, ONNX weights, embeddings, MLflow runs và model repository của Triton không được commit vào Git. Git chỉ lưu code, config, manifest, preprocessing contract và checksum để tái tạo hoặc xác minh release.
