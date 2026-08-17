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

```powershell
conda activate C:\tmp\reid-mlops
```

Nếu muốn tạo môi trường mới từ đầu:

```powershell
conda create -n reid-mlops python=3.10 -y
conda activate reid-mlops
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra nhanh sau khi cài:

```powershell
python -c "import torch, torchvision, mlflow, yaml, numpy, PIL, tqdm; print('torch =', torch.__version__); print('torchvision =', torchvision.__version__); print('cuda =', torch.cuda.is_available())"
```

Nếu chạy CPU:

```powershell
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

### 5.1. Chạy nhanh nhất để train local

Kích hoạt môi trường:

```powershell
conda activate C:\tmp\reid-mlops
```

Train một run mới với `Market-1501`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_train.ps1 -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15"
```

Train rồi evaluate luôn trong cùng một run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_pipeline.ps1 -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15"
```

Ví dụ với `DukeMTMC-reID`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_pipeline.ps1 -DatasetName dukemtmc-reid -DatasetRoot "datasets/dukemtmc"
```

Ví dụ với `MSMT17`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_pipeline.ps1 -DatasetName msmt17 -DatasetRoot "datasets/MSMT17_V1"
```

### 5.2. Nếu muốn chạy trực tiếp bằng Python

Train trực tiếp:

```powershell
python .\src\train.py --config .\configs\dadnet.yaml --set data.dataset.name=market1501 --set data.location.root="datasets/Market-1501-v15.09.15"
```

Train trực tiếp với Duke:

```powershell
python .\src\train.py --config .\configs\dadnet.yaml --set data.dataset.name=dukemtmc-reid --set data.location.root="datasets/dukemtmc"
```

Train trực tiếp với MSMT17:

```powershell
python .\src\train.py --config .\configs\dadnet.yaml --set data.dataset.name=msmt17 --set data.location.root="datasets/MSMT17_V1"
```

### 5.3. Smoke test

```powershell
conda activate C:\tmp\reid-mlops
python .\src\train.py --config .\configs\dadnet_smoke.yaml
python .\src\train.py --config .\configs\baseline_smoke.yaml
```

### 5.4. Override nhanh từ CLI

```powershell
python .\src\train.py --config .\configs\dadnet.yaml --set data.batch_size=16 --set train.learning_rate=0.00003 --set logging.enable_mlflow=false
```

Giá trị sau dấu `=` được parse theo YAML, nên có thể dùng được với:

- số như `32`, `0.0001`
- boolean như `true`, `false`
- list như `"[1, 2, 3]"` nếu cần mở rộng sau này

Một số ví dụ hay dùng:

```powershell
python .\src\train.py --config .\configs\dadnet.yaml --set train.learning_rate=0.00003 --set train.scheduler_type=cosine
python .\src\train.py --config .\configs\dadnet.yaml --set data.batch_size=16 --set train.triplet_margin=0.4
python .\src\train.py --config .\configs\dadnet.yaml --set augmentation.random_erasing=false --set augmentation.color_jitter=false
```

### 5.5. Evaluate checkpoint

Nếu bạn đã có `best_model.pth`, có thể evaluate riêng:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_evaluate.ps1 -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15" -CheckpointPath ".\artifacts\market1501\...\checkpoints\best_model.pth"
```

Hoặc gọi Python trực tiếp:

```powershell
python .\src\evaluate.py --config .\configs\dadnet.yaml --checkpoint ".\artifacts\market1501\...\checkpoints\best_model.pth" --set data.dataset.name=market1501 --set data.location.root="datasets/Market-1501-v15.09.15"
```

### 5.6. Trích xuất embedding tham chiếu

```powershell
python .\src\extract_reference.py --config .\configs\dadnet.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
python .\src\extract_reference.py --config .\configs\dadnet.yaml --checkpoint .\artifacts\checkpoints\best_model.pth --set data.dataset.name=dukemtmc-reid --set data.location.root="datasets/dukemtmc"
python .\src\extract_reference.py --config .\configs\dadnet.yaml --checkpoint .\artifacts\checkpoints\best_model.pth --set data.dataset.name=msmt17 --set data.location.root="datasets/MSMT17_V1"
```

Hoặc dùng script local:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_extract.ps1 -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15" -CheckpointPath ".\artifacts\market1501\...\checkpoints\best_model.pth"
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

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_imported_model_pipeline.ps1 -ModelRoot ".\model" -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15"
```

Script này sẽ:

- dùng checkpoint từ `model/checkpoints/best_model.pth`
- không ghi đè artifact gốc trong thư mục `model/`
- tạo một run local mới trong `artifacts/<dataset>/<run-slug>/`
- lưu lại `evaluate.log`, `extract.log`, `evaluation_latest.json` và bộ `reference_embeddings`

### 5.8. Export checkpoint sang ONNX

Cài thêm dependency export nếu máy chưa có:

```powershell
conda activate C:\tmp\reid-mlops
pip install onnx
pip install onnxscript
```

Export một checkpoint local sang ONNX embedding model:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_export_onnx.ps1 -DatasetName market1501 -DatasetRoot "datasets/Market-1501-v15.09.15" -CheckpointPath ".\model\checkpoints\best_model.pth"
```

Kết quả sẽ nằm trong:

- `artifacts/<dataset>/<run-slug>/exports/model_embedding.onnx`
- `artifacts/<dataset>/<run-slug>/exports/onnx_export_manifest.json`
- `artifacts/<dataset>/<run-slug>/logs/export_onnx.log`

Model ONNX này trả về trực tiếp `embeddings`, phù hợp cho bước so khớp đặc trưng trong hệ thống ReID và là đầu vào tự nhiên cho giai đoạn serving sau này.

Ghi chú: với stack `PyTorch 2.11` hiện tại trong dự án, nên dùng `opset 18` để tránh lỗi convert version khi exporter tự sinh graph ONNX mới.

### 5.9. Dong goi ONNX thanh Triton model repository

Sau khi da co file ONNX, co the tao cau truc model repository cho Triton bang:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_prepare_triton_model.ps1 -OnnxPath ".\artifacts\market1501\market1501-dadnet-export-onnx-20260817-162230\exports\model_embedding.onnx"
```

Ket qua se duoc tao theo cau truc:

```text
artifacts/triton/triton-repository-<timestamp>/model_repository/
└─ reid_embedding/
   ├─ config.pbtxt
   └─ 1/
      └─ model.onnx
```

Script nay se:

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

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_prepare_triton_model.ps1 -OnnxPath "<duong-dan-onnx>" -InstanceKind KIND_GPU
```

### 5.10. Chay Triton local bang Docker Compose

Tao file env rieng cho Triton:

```powershell
Copy-Item .\.env.triton.example .\.env.triton
```

Sau do sua gia tri `TRITON_MODEL_REPOSITORY` trong `.env.triton` tro toi model repository vua tao.

Kiem tra nhanh file env can co:

```text
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3
TRITON_MODEL_REPOSITORY=E:/.../artifacts/triton/triton-repository-<timestamp>/model_repository
TRITON_NVIDIA_VISIBLE_DEVICES=void
```

Chay Triton local:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_triton_local.ps1 -Detach
```

Dung server:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_triton_local.ps1
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

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_triton_infer.ps1 -ImagePath ".\datasets\Market-1501-v15.09.15\query\0001_c1s1_001051_00.jpg"
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

Nếu chạy qua `run_local_pipeline.ps1` thì thường bạn sẽ quan tâm nhất tới:

- `artifacts/<dataset>/<run-slug>/checkpoints/best_model.pth`
- `artifacts/<dataset>/<run-slug>/metrics/metrics_v1.json`
- `artifacts/<dataset>/<run-slug>/metrics/evaluation_latest.json`
- `artifacts/<dataset>/<run-slug>/logs/train.log`
- `artifacts/<dataset>/<run-slug>/logs/evaluate.log`

## 8. MLflow

Mở giao diện MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Sau đó truy cập:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 9. Local Automation

Repo hiện dùng local runner để train và evaluate ngay trên máy.

Ba script chính:

- `scripts/run_local_train.ps1`: chỉ train
- `scripts/run_local_evaluate.ps1`: chỉ evaluate
- `scripts/run_local_extract.ps1`: chỉ extract reference embeddings
- `scripts/run_local_pipeline.ps1`: train rồi evaluate trong cùng một run
- `scripts/run_imported_model_pipeline.ps1`: dùng checkpoint đã train sẵn để evaluate + extract trên local
- `scripts/run_local_export_onnx.ps1`: export checkpoint sang ONNX embedding model
- `scripts/run_prepare_triton_model.ps1`: dong goi ONNX thanh Triton model repository
- `scripts/run_triton_local.ps1`: chay Triton local bang Docker Compose
- `scripts/stop_triton_local.ps1`: dung Triton local
- `scripts/run_triton_infer.ps1`: gui anh qua Triton va lay embedding

`run_local_pipeline.ps1` là lựa chọn nên dùng hằng ngày vì:

- tự tạo `run_slug`
- train và evaluate dùng chung một `run_root`
- log và artifact nằm chung một chỗ
- phù hợp để sau này gọi từ pipeline local/MLOps

## 10. Git và push code

Repo đã có [`.gitignore`](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\.gitignore) để tránh đẩy lên:

- `datasets/`
- `artifacts/`
- `mlruns/`
- file mô hình như `*.pth`, `*.pt`, `*.npy`

Nếu `git` báo lỗi `dubious ownership`, chạy:

```powershell
git config --global --add safe.directory "C:/Users/Gia Lam/Desktop/IUH Data/Năm 5 - Kỳ 1/Person-Re-Identification"
```

Quy trình cơ bản:

```powershell
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
