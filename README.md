# Person Re-Identification với ResNet50 và DADNet-inspired

Dự án này xây dựng pipeline `Person Re-Identification` bằng `PyTorch` trên bộ dữ liệu `Market-1501`. Mục tiêu là huấn luyện một mô hình nhận diện lại người giữa các camera khác nhau, đồng thời hỗ trợ theo dõi thí nghiệm, lưu checkpoint và đánh giá theo các chỉ số phổ biến như `Rank-1` và `mAP`.

Hiện tại repo có 2 hướng chính:

- `Baseline`: `ResNet50 -> Embedding -> Classifier`
- `DADNet-inspired`: thêm attention và head tăng khả năng phân biệt đặc trưng

## 1. Tổng quan kiến trúc

### Baseline

```text
Input
  -> ResNet50 Backbone
  -> Global Average Pooling
  -> Linear(2048 -> 512)
  -> BatchNorm
  -> ReLU
  -> Classifier
```

### DADNet-inspired

```text
Input
  -> ResNet50 Backbone
  -> CFT Attention Module
  -> Position-Aware Attention
  -> Global Average Pooling
  -> DEM (Distinguishability Enhancement Module)
  -> Classifier
```

Ghi chú:

- Đây là phiên bản `inspired by DADNet`, không phải bản tái hiện nguyên gốc 100% từ paper.
- Backbone hiện tại vẫn là `ResNet50`.
- Các thử nghiệm gần đây tập trung vào `loss`, `batch strategy`, `scheduler`, `re-ranking` và tinh chỉnh attention nhẹ thay vì thay backbone.

## 2. Cấu trúc dự án

```text
Person-Re-Identification/
├─ configs/
│  ├─ dadnet.yaml
│  ├─ dadnet_smoke.yaml
│  ├─ baseline.yaml
│  └─ baseline_smoke.yaml
├─ datasets/
│  └─ Market-1501-v15.09.15/
├─ artifacts/
├─ mlruns/
├─ src/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ extract_reference.py
│  ├─ common/
│  ├─ data/
│  ├─ models/
│  └─ reid/
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 3. Yêu cầu môi trường

- Windows 10/11
- Python `3.10` khuyến nghị
- GPU NVIDIA là tùy chọn nhưng rất nên có nếu train full

Môi trường đã được xác nhận chạy trong máy hiện tại:

```bash
conda activate C:\tmp\reid-mlops
```

Nếu muốn tạo môi trường mới từ đầu:

```bash
conda create -n reid-mlops python=3.10 -y
conda activate reid-mlops
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra nhanh sau khi cài:

```bash
python -c "import torch, torchvision, mlflow, yaml, numpy, PIL, tqdm; print('torch =', torch.__version__); print('torchvision =', torchvision.__version__); print('cuda =', torch.cuda.is_available())"
```

Nếu chạy CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Tham khảo cài đặt GPU:

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)

## 4. Dataset

Repo hiện chạy local, vì vậy bạn chỉ cần có dataset trên máy và truyền đúng `DatasetRoot`.

### 4.1. Dataset mặc định

Thư mục dữ liệu mặc định trong config:

```text
datasets/Market-1501-v15.09.15/
```

Với `Market-1501` hoặc `DukeMTMC-reID`, dataset root cần có đủ:

- `bounding_box_train`
- `query`
- `bounding_box_test`

Ví dụ:

```text
datasets/dukemtmc/
├─ bounding_box_train/
├─ query/
└─ bounding_box_test/
```

Với `MSMT17`, root cần giữ nguyên protocol gốc:

- `list_train.txt`
- `list_query.txt`
- `list_gallery.txt`
- thư mục ảnh tương ứng của `MSMT17`

Các đường dẫn mẫu đang được khai báo trong:

- [baseline.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\baseline.yaml)
- [dadnet.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\dadnet.yaml)

### 4.2. Schema config dataset

Cấu trúc config dataset hiện tại đã được chuẩn hóa theo hướng:

```yaml
data:
  source_type: image_folder
  dataset:
    name: market1501
  location:
    root: datasets/Market-1501-v15.09.15
    splits:
      train: bounding_box_train
      query: query
      gallery: bounding_box_test
```

Code vẫn tương thích ngược với config cũ dùng `train_dir`, `query_dir`, `gallery_dir`, nhưng nên ưu tiên format mới để chuẩn bị cho bước manifest/version sau này.

Ý nghĩa của schema mới:

- `source_type`: kiểu nguồn dữ liệu, hiện tại hỗ trợ `image_folder`
- `dataset.name`: tên logic của dataset để log/so sánh thí nghiệm
- `location.root`: nơi chứa dữ liệu
- `location.splits`: mô tả split train/query/gallery

Sau này nếu chuyển sang MLOps đầy đủ hơn, chỗ này có thể mở rộng tiếp sang:

- `source_type: manifest`
- `location.manifest_path`
- `dataset.version`
- `dataset.uri`

Với `MSMT17`, pipeline hiện tại giữ nguyên protocol gốc bằng cách đọc trực tiếp:

- `list_train.txt`
- `list_query.txt`
- `list_gallery.txt`

thay vì ép phải đổi dữ liệu vật lý sang `bounding_box_train/query/bounding_box_test`.

## 5. Cách chạy

### 5.1. Kích hoạt môi trường

```bash
conda activate C:\tmp\reid-mlops
```

Hoặc nếu dùng môi trường Python khác, chỉ cần đảm bảo đã cài đủ dependency trong `requirements.txt`.

### 5.2. Train model

Train với `Market-1501`:

```bash
python src/train.py --config configs/dadnet.yaml --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-train --set artifacts.run_root=artifacts/market1501/market1501-dadnet-train
```

Ví dụ với `DukeMTMC-reID`:

```bash
python src/train.py --config configs/dadnet.yaml --set data.dataset.name=dukemtmc-reid --set data.location.root=datasets/dukemtmc --set runtime.run_slug=dukemtmc-dadnet-train --set artifacts.run_root=artifacts/dukemtmc-reid/dukemtmc-dadnet-train
```

Ví dụ với `MSMT17`:

```bash
python src/train.py --config configs/dadnet.yaml --set data.dataset.name=msmt17 --set data.location.root=datasets/MSMT17_V1 --set runtime.run_slug=msmt17-dadnet-train --set artifacts.run_root=artifacts/msmt17/msmt17-dadnet-train
```

### 5.3. Evaluate checkpoint

Nếu bạn đã có `best_model.pth`, có thể evaluate riêng:

```bash
python src/evaluate.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-eval --set artifacts.run_root=artifacts/market1501/market1501-dadnet-eval
```

### 5.4. Smoke test

```bash
python src/train.py --config configs/dadnet_smoke.yaml
python src/train.py --config configs/baseline_smoke.yaml
```

### 5.5. Override nhanh từ CLI

```bash
python src/train.py --config configs/dadnet.yaml --set data.batch_size=16 --set train.learning_rate=0.00003 --set logging.enable_mlflow=false
```

Giá trị sau dấu `=` được parse theo YAML, nên có thể dùng được với:

- số như `32`, `0.0001`
- boolean như `true`, `false`
- list như `"[1, 2, 3]"` nếu cần mở rộng sau này

Một số ví dụ hay dùng:

```bash
python src/train.py --config configs/dadnet.yaml --set train.learning_rate=0.00003 --set train.scheduler_type=cosine
python src/train.py --config configs/dadnet.yaml --set data.batch_size=16 --set train.triplet_margin=0.4
python src/train.py --config configs/dadnet.yaml --set augmentation.random_erasing=false --set augmentation.color_jitter=false
```

### 5.6. Trích xuất embedding tham chiếu

```bash
python src/extract_reference.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-eval --set artifacts.run_root=artifacts/market1501/market1501-dadnet-eval
```

Ví dụ với dataset drift:

```bash
python src/extract_reference.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=dukemtmc-reid --set data.location.root=datasets/dukemtmc --set runtime.run_slug=dukemtmc-dadnet-eval --set artifacts.run_root=artifacts/dukemtmc-reid/dukemtmc-dadnet-eval
python src/extract_reference.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=msmt17 --set data.location.root=datasets/MSMT17_V1 --set runtime.run_slug=msmt17-dadnet-eval --set artifacts.run_root=artifacts/msmt17/msmt17-dadnet-eval
```

### 5.7. Dùng lại model đã train sẵn từ Kaggle hoặc nguồn ngoài

Nếu bạn đã có một thư mục model như:

```text
model/
├─ checkpoints/
│  └─ best_model.pth
├─ logs/
└─ metrics/
```

thì có thể chạy lại `evaluate` và `extract reference embeddings` trên máy local bằng:

```bash
python src/evaluate.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-imported --set artifacts.run_root=artifacts/market1501/market1501-dadnet-imported
python src/extract_reference.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-imported --set artifacts.run_root=artifacts/market1501/market1501-dadnet-imported
```

Các lệnh này sẽ:

- dùng checkpoint từ `model/checkpoints/best_model.pth`
- không ghi đè artifact gốc trong thư mục `model/`
- tạo một run local mới trong `artifacts/<dataset>/<run-slug>/`
- lưu lại `evaluate.log`, `extract.log`, `evaluation_latest.json` và bộ `reference_embeddings`

### 5.8. Export checkpoint sang ONNX

Cài thêm dependency export nếu máy chưa có:

```bash
pip install onnx onnxscript
```

Export một checkpoint local sang ONNX embedding model:

```bash
python src/export_onnx.py --config configs/dadnet.yaml --checkpoint model/checkpoints/best_model.pth --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-dadnet-export-onnx --set artifacts.run_root=artifacts/market1501/market1501-dadnet-export-onnx
```

Kết quả sẽ nằm trong:

- `artifacts/<dataset>/<run-slug>/exports/model_embedding.onnx`
- `artifacts/<dataset>/<run-slug>/exports/onnx_export_manifest.json`
- `artifacts/<dataset>/<run-slug>/logs/export_onnx.log`

Model ONNX này trả về trực tiếp `embeddings`, phù hợp cho bước so khớp đặc trưng trong hệ thống ReID và là đầu vào tự nhiên cho giai đoạn serving sau này.

Ghi chú: với stack `PyTorch 2.11` hiện tại trong dự án, nên dùng `opset 18` để tránh lỗi convert version khi exporter tự sinh graph ONNX mới.

### 5.9. Dong goi ONNX thanh Triton model repository

Sau khi da co file ONNX, co the tao cau truc model repository cho Triton bang:

```bash
python src/prepare_triton_model.py --onnx-path artifacts/market1501/market1501-dadnet-export-onnx/exports/model_embedding.onnx --output-root artifacts/triton/local-cpu-model-repository/model_repository --model-name reid_embedding --model-version 1 --max-batch-size 0 --input-height 224 --input-width 224 --embedding-dim 512 --instance-kind KIND_CPU
```

Ket qua se duoc tao theo cau truc:

```text
artifacts/triton/local-cpu-model-repository/model_repository/
└─ reid_embedding/
   ├─ config.pbtxt
   └─ 1/
      └─ model.onnx
```

Lenh nay se:

- copy file ONNX vao dung cau truc Triton
- sinh `config.pbtxt` cho `images -> embeddings`
- bat `dynamic_batching`
- luu `triton_model_manifest.json` de sau nay noi tiep sang serving
- mac dinh dung `KIND_CPU` de de test local; co the doi sang `KIND_GPU` khi dong goi cho may co CUDA/Triton GPU on dinh

Gia tri mac dinh hien tai phu hop voi model ReID cua do an:

- input: `3 x 224 x 224`
- output: `512-dim embeddings`
- model name: `reid_embedding`
- max batch size local mac dinh: `0`
- preferred batch size `4, 8, 16` chi nen bat khi ban export duoc ONNX dang dynamic-batch that su

Neu muon dong goi ban cho GPU, co the goi them:

```bash
python src/prepare_triton_model.py --onnx-path artifacts/market1501/market1501-dadnet-export-onnx/exports/model_embedding.onnx --output-root artifacts/triton/local-gpu-model-repository/model_repository --instance-kind KIND_GPU
```

### 5.10. Chay Triton local bang Docker Compose

Tao file env rieng cho Triton:

```bash
cp .env.triton.example .env.triton
```

Sau do sua gia tri `TRITON_MODEL_REPOSITORY` trong `.env.triton` tro toi model repository vua tao.

Kiem tra nhanh file env can co:

```text
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3
TRITON_MODEL_REPOSITORY=E:/.../artifacts/triton/local-cpu-model-repository/model_repository
TRITON_NVIDIA_VISIBLE_DEVICES=void
```

Chay Triton local:

```bash
docker compose --env-file .env.triton -f docker-compose.triton.yml up -d
```

Dung server:

```bash
docker compose --env-file .env.triton -f docker-compose.triton.yml down
```

Compose hien tai map 3 cong mac dinh cua Triton:

- HTTP: `8000`
- gRPC: `8001`
- Metrics: `8002`

Ban local mac dinh dang chay theo huong CPU-safe:

- `TRITON_NVIDIA_VISIBLE_DEVICES=void`
- khong ep Docker Compose phai dat reservation GPU

Neu sau nay ban deploy tren may CUDA on dinh va muon dung GPU, co the doi:

```text
TRITON_NVIDIA_VISIBLE_DEVICES=0
```

Sau khi server len, co the kiem tra health qua:

- [http://localhost:8000/v2/health/live](http://localhost:8000/v2/health/live)
- [http://localhost:8000/v2/health/ready](http://localhost:8000/v2/health/ready)

### 5.11. Goi infer local de lay embedding tu Triton

Sau khi Triton da chay, co the gui 1 anh vao model `reid_embedding` bang:

```bash
python src/triton_infer.py --image-path datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg --server-url http://localhost:8000 --model-name reid_embedding
```

Client nay:

- preprocess anh dung voi pipeline test cua repo
- resize ve `224 x 224`
- normalize theo `ImageNet mean/std`
- goi HTTP infer toi Triton
- luu `embedding` ra file `.npy`
- luu manifest JSON de phuc vu buoc vector search sau nay

Ket qua mac dinh duoc luu trong:

- `artifacts/inference/local-triton/<image-stem>_embedding.npy`
- `artifacts/inference/local-triton/<image-stem>_infer_manifest.json`
- `artifacts/inference/local-triton/triton_infer.log`

### 5.12. Qdrant local cho vector search

Tao file env:

```bash
cp .env.qdrant.example .env.qdrant
```

Chay Qdrant local:

```bash
docker compose --env-file .env.qdrant -f docker-compose.qdrant.yml up -d
```

Dung Qdrant:

```bash
docker compose --env-file .env.qdrant -f docker-compose.qdrant.yml down
```

Tao collection `reid_reference`:

```bash
python src/qdrant_local.py --qdrant-url http://localhost:6333 create-collection --collection-name reid_reference --vector-size 512
```

Neu da co bo `reference_embeddings.npy`, `reference_pids.npy`, `reference_camids.npy` thi upsert vao Qdrant:

```bash
python src/qdrant_local.py --qdrant-url http://localhost:6333 upsert-reference --collection-name reid_reference --embeddings-path artifacts/market1501/market1501-dadnet-eval/embeddings/reference_embeddings.npy --pids-path artifacts/market1501/market1501-dadnet-eval/embeddings/reference_pids.npy --camids-path artifacts/market1501/market1501-dadnet-eval/embeddings/reference_camids.npy
```

Sau khi Triton da sinh `embedding.npy`, co the query top-k nhu sau:

```bash
python src/qdrant_local.py --qdrant-url http://localhost:6333 query-embedding --collection-name reid_reference --embedding-path artifacts/inference/local-triton/0001_c1s1_001051_00_embedding.npy
```

Phan nay la cau noi dau tien cho retrieval:

- Triton sinh `embedding`
- Qdrant luu `reference embeddings`
- query embedding di tim top-k match gan nhat

### 5.13. Evaluate deployment quality qua Triton

Sau khi Triton da chay, co the tinh `Rank-1`, `Rank-5`, `Rank-10`, `Rank-20`, `mAP`, `mINP` tren query set bang command sau:

```bash
python src/evaluate_deployment_retrieval.py --config configs/dadnet.yaml --server-url http://localhost:8000 --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-deployment-eval --set artifacts.run_root=artifacts/market1501/market1501-deployment-eval
```

Neu muon smoke test nhanh tren mot phan query set:

```bash
python src/evaluate_deployment_retrieval.py --config configs/dadnet.yaml --server-url http://localhost:8000 --max-queries 100 --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-deployment-smoke --set artifacts.run_root=artifacts/market1501/market1501-deployment-smoke
```

Neu muon smoke test nhanh nhung van dam bao gallery co dung identity de soat pipeline:

```bash
python src/evaluate_deployment_retrieval.py --config configs/dadnet.yaml --server-url http://localhost:8000 --max-queries 20 --gallery-match-query-pids-only --set data.dataset.name=market1501 --set data.location.root=datasets/Market-1501-v15.09.15 --set runtime.run_slug=market1501-deployment-smoke-pid-gallery --set artifacts.run_root=artifacts/market1501/market1501-deployment-smoke-pid-gallery
```

Lenh nay se:

- dua tung anh query qua Triton de sinh embedding
- dua tung anh gallery qua Triton de sinh embedding
- tinh metric retrieval theo chuan Market1501
- luu `deployment_retrieval_evaluate.log` va `deployment_retrieval_latest.json`

Luu y:

- command nay dung cho benchmark deployment quality
- collection Qdrant `reid_reference` hien tai dang phu hop cho database retrieval local, khong phai gallery benchmark cua Market-1501
- vi vay, de danh gia `Rank-1` va `mAP` dung nghia, can dung gallery split chuan

## 6. Những gì đang có trong bản hiện tại

Pipeline hiện đã hỗ trợ:

- `RandomIdentitySampler`
- `AMP` khi có CUDA
- `Early stopping`
- `ReduceLROnPlateau` hoặc `Cosine scheduler + warmup`
- `Label smoothing`
- `Triplet loss`
- `Center loss` tùy chọn
- `Color jitter` và `Random erasing`
- `Re-ranking` khi evaluate
- lưu `best_model.pth` và `last_model.pth`
- theo dõi thí nghiệm bằng `MLflow`
- tách artifact theo `dataset + config + command + timestamp` để tránh ghi đè giữa các run

## 7. Các file đầu ra

Sau khi train hoặc evaluate, kết quả thường được lưu ở:

- `artifacts/<dataset>/<run-slug>/checkpoints/last_model.pth`
- `artifacts/<dataset>/<run-slug>/checkpoints/best_model.pth`
- `artifacts/<dataset>/<run-slug>/metrics/metrics_v1.json`
- `artifacts/<dataset>/<run-slug>/metrics/evaluation_latest.json`
- `artifacts/<dataset>/<run-slug>/embeddings/reference_embeddings.npy`
- `artifacts/<dataset>/<run-slug>/embeddings/reference_pids.npy`
- `artifacts/<dataset>/<run-slug>/embeddings/reference_camids.npy`
- `artifacts/<dataset>/<run-slug>/embeddings/reference_manifest.json`
- `artifacts/<dataset>/<run-slug>/logs/effective_config.json`

Khi chạy train/evaluate bằng các lệnh Python ở trên, bạn sẽ quan tâm nhất tới:

- `artifacts/<dataset>/<run-slug>/checkpoints/best_model.pth`
- `artifacts/<dataset>/<run-slug>/metrics/metrics_v1.json`
- `artifacts/<dataset>/<run-slug>/metrics/evaluation_latest.json`
- `artifacts/<dataset>/<run-slug>/logs/train.log`
- `artifacts/<dataset>/<run-slug>/logs/evaluate.log`

## 8. MLflow

Mở giao diện MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Sau đó truy cập:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 9. Local Commands

Repo hiện dùng trực tiếp Python scripts và Docker Compose để chạy local. Các bước quan trọng:

- `python src/train.py`: train model
- `python src/evaluate.py`: evaluate checkpoint
- `python src/extract_reference.py`: trích xuất reference embeddings
- `python src/export_onnx.py`: export checkpoint sang ONNX embedding model
- `python src/prepare_triton_model.py`: đóng gói ONNX thành Triton model repository
- `python src/triton_infer.py`: gọi Triton và lấy embedding
- `python src/qdrant_local.py`: tạo collection, upsert reference embeddings, query top-k trong Qdrant
- `python src/evaluate_deployment_retrieval.py`: đánh giá chất lượng deployment qua Triton
- `docker compose --env-file .env.triton -f docker-compose.triton.yml up -d`: chạy Triton local
- `docker compose --env-file .env.qdrant -f docker-compose.qdrant.yml up -d`: chạy Qdrant local

Các local wrapper cũ đã được bỏ để repo tập trung vào Python/Docker, dễ chạy hơn trên nhiều môi trường.

## 10. Git và push code

Repo đã có [`.gitignore`](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\.gitignore) để tránh đẩy lên:

- `datasets/`
- `artifacts/`
- `mlruns/`
- file mô hình như `*.pth`, `*.pt`, `*.npy`

Nếu `git` báo lỗi `dubious ownership`, chạy:

```bash
git config --global --add safe.directory "C:/Users/Gia Lam/Desktop/IUH Data/Năm 5 - Kỳ 1/Person-Re-Identification"
```

Quy trình cơ bản:

```bash
git status
git add .
git commit -m "Your commit message"
git push origin <ten-branch>
```

## 11. Hướng phát triển tiếp

- So sánh lại `dadnet.yaml` với `baseline.yaml`
- So riêng `before rerank` và `after rerank`
- Tối ưu thêm `sampler`, `triplet margin`, `scheduler step`
- Nếu cần, tách riêng mô hình DADNet sang file chuyên biệt thay vì để chung trong [reid_model.py](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\src\models\reid_model.py)

## 12. Tài liệu tham khảo

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [MLflow Quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [MLflow Self Hosting Overview](https://mlflow.org/docs/latest/self-hosting/index.html)
