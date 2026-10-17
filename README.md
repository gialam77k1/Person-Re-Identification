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
src/api/                                       FastAPI embedding và person search
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

Hạ tầng local dùng PostgreSQL, MinIO, MLflow, Qdrant, Triton và FastAPI trong một stack chung. Hướng dẫn vận hành nằm tại [`docs/MLOPS_LOCAL.md`](docs/MLOPS_LOCAL.md); API inference nằm tại [`docs/FASTAPI_INFERENCE.md`](docs/FASTAPI_INFERENCE.md); kiểm thử deployment nằm tại [`docs/DEPLOYMENT_TESTING.md`](docs/DEPLOYMENT_TESTING.md); pipeline tái lập nằm tại [`docs/DVC_PIPELINE.md`](docs/DVC_PIPELINE.md); model validation gate nằm tại [`docs/MODEL_VALIDATION_GATE.md`](docs/MODEL_VALIDATION_GATE.md); kế hoạch đầy đủ nằm tại [`KE_HOACH_MLOPS.md`](KE_HOACH_MLOPS.md).

```bash
./scripts/mlops_stack.sh validate
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
```

Chạy toàn bộ pipeline theo dependency graph và chỉ tái chạy stage bị ảnh hưởng:

```bash
./scripts/dvc_local.sh repro
./scripts/dvc_local.sh metrics show
```

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
  --onnx-path artifacts/pipeline/market1501-vit-bnneck-v1-dvc/onnx/exports/model_embedding.onnx \
  --output-root artifacts/pipeline/market1501-vit-bnneck-v1-dvc/triton/model_repository \
  --model-name reid_embedding \
  --model-version 1 \
  --validation-report artifacts/pipeline/market1501-vit-bnneck-v1-dvc/release/validation_report.json \
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

### 6. Chạy FastAPI inference

```bash
./scripts/mlops_stack.sh up
curl --fail http://localhost:8080/ready
curl --fail http://localhost:8080/model-info
```

Search bằng ảnh JPG/PNG:

```bash
curl --fail -X POST 'http://localhost:8080/search?top_k=5' \
  -F 'image=@datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg;type=image/jpeg'
```

Chi tiết endpoint, response và giới hạn upload nằm trong [`docs/FASTAPI_INFERENCE.md`](docs/FASTAPI_INFERENCE.md).

### 7. Kiểm thử deployment

```bash
./scripts/test_deployment.sh unit
./scripts/test_deployment.sh integration
./scripts/test_deployment.sh evaluate
./scripts/test_deployment.sh quality
./scripts/test_deployment.sh benchmark
./scripts/test_deployment.sh persistence
```

Full deployment đạt Rank-1 `91,36%`, quality gate PASS và `/search` local có p50/p95/p99 lần lượt `86,97/92,10/97,11 ms`. Quy trình và report nằm tại [`docs/DEPLOYMENT_TESTING.md`](docs/DEPLOYMENT_TESTING.md).

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

Model v1 đã được đăng ký trên MLflow Model Registry với tên `market1501-vit-bnneck`, version `1` và alias `champion`. Xem cách publish và kiểm tra tại [`docs/MLFLOW_TRACKING.md`](docs/MLFLOW_TRACKING.md).

## Quy ước artifact

Checkpoint, ONNX weights, embeddings, MLflow runs và model repository của Triton không được commit vào Git. Git chỉ lưu code, config, manifest, preprocessing contract và checksum để tái tạo hoặc xác minh release.
