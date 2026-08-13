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
│  ├─ baseline.yaml
│  ├─ baseline_smoke.yaml
│  ├─ dadnet.yaml
│  ├─ dadnet_smoke.yaml
│  ├─ dadnet_loss_tuned.yaml
│  ├─ dadnet_batch_tuned.yaml
│  ├─ dadnet_scheduler_tuned.yaml
│  └─ dadnet_eval_final.yaml
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

Kiểm tra nhanh:

```powershell
python -c "import torch, torchvision, mlflow, yaml, numpy, PIL, tqdm; print('torch =', torch.__version__); print('torchvision =', torchvision.__version__); print('cuda =', torch.cuda.is_available())"
```

Nếu muốn tạo môi trường mới:

```powershell
conda create -n reid-mlops python=3.10 -y
conda activate reid-mlops
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chạy CPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Tham khảo cài đặt GPU:

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)

## 4. Dataset

Thư mục dữ liệu mặc định:

```text
datasets/Market-1501-v15.09.15/
```

Cần có đủ các thư mục:

- `bounding_box_train`
- `query`
- `bounding_box_test`

Các đường dẫn đang được khai báo trong:

- [baseline.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\baseline.yaml)
- [dadnet.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\dadnet.yaml)

Cấu trúc config dataset hiện tại đã được chuẩn hóa theo hướng:

```yaml
data:
  dataset:
    name: market1501
    root: datasets/Market-1501-v15.09.15
  splits:
    train: bounding_box_train
    query: query
    gallery: bounding_box_test
```

Code vẫn tương thích ngược với config cũ dùng `train_dir`, `query_dir`, `gallery_dir`, nhưng nên ưu tiên format mới để chuẩn bị cho bước multi-dataset.

Hiện tại code đã dùng dataset factory theo `data.dataset.name`, nên các bước tiếp theo có thể mở rộng chung cho:

- `market1501`
- `dukemtmc-reid`
- `msmt17`

## 5. Cách chạy

### Smoke test

Baseline:

```powershell
conda activate C:\tmp\reid-mlops
python .\src\train.py --config .\configs\baseline_smoke.yaml
```

DADNet-inspired:

```powershell
conda activate C:\tmp\reid-mlops
python .\src\train.py --config .\configs\dadnet_smoke.yaml
```

### Train chính

Baseline:

```powershell
conda activate C:\tmp\reid-mlops
python .\src\train.py --config .\configs\baseline.yaml
```

DADNet-inspired:

```powershell
conda activate C:\tmp\reid-mlops
python .\src\train.py --config .\configs\dadnet.yaml
```

Tại thời điểm hiện tại, `dadnet.yaml` là file cấu hình chính đang được tinh chỉnh trực tiếp để thử phá plateau của metric.

Có thể override nhanh cấu hình từ CLI mà không cần sửa YAML:

```powershell
python .\src\train.py --config .\configs\baseline.yaml --set data.batch_size=32 --set train.learning_rate=0.0001 --set logging.enable_mlflow=false
python .\src\evaluate.py --config .\configs\dadnet_eval_final.yaml --checkpoint .\artifacts\checkpoints\best_model.pth --set evaluation.use_rerank=true
python .\src\extract_reference.py --config .\configs\baseline.yaml --checkpoint .\artifacts\checkpoints\best_model.pth --set data.eval_batch_size=64
```

Giá trị sau dấu `=` được parse theo YAML, nên có thể dùng được với:

- số như `32`, `0.0001`
- boolean như `true`, `false`
- list như `"[1, 2, 3]"` nếu cần mở rộng sau này

### Các config thử nghiệm riêng

Loss-focused:

```powershell
python .\src\train.py --config .\configs\dadnet_loss_tuned.yaml
```

Batch-focused:

```powershell
python .\src\train.py --config .\configs\dadnet_batch_tuned.yaml
```

Scheduler-focused:

```powershell
python .\src\train.py --config .\configs\dadnet_scheduler_tuned.yaml
```

Config đánh giá cuối:

```powershell
python .\src\evaluate.py --config .\configs\dadnet_eval_final.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
```

### Đánh giá checkpoint

Baseline:

```powershell
python .\src\evaluate.py --config .\configs\baseline.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
```

DADNet-inspired:

```powershell
python .\src\evaluate.py --config .\configs\dadnet.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
```

### Trích xuất embedding tham chiếu

Baseline:

```powershell
python .\src\extract_reference.py --config .\configs\baseline.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
```

DADNet-inspired:

```powershell
python .\src\extract_reference.py --config .\configs\dadnet.yaml --checkpoint .\artifacts\checkpoints\best_model.pth
```

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

## 7. Các file đầu ra

Sau khi train hoặc evaluate, kết quả thường được lưu ở:

- `artifacts/checkpoints/last_model.pth`
- `artifacts/checkpoints/best_model.pth`
- `artifacts/metrics/metrics_v1.json`
- `artifacts/metrics/evaluation_latest.json`
- `artifacts/embeddings/reference_embeddings.npy`
- `artifacts/embeddings/reference_pids.npy`
- `artifacts/embeddings/reference_camids.npy`
- `artifacts/embeddings/reference_manifest.json`

## 8. MLflow

Mở giao diện MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Sau đó truy cập:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 9. Git và push code

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

## 10. Hướng phát triển tiếp

- So sánh lại `dadnet.yaml` với `baseline.yaml`
- Thử ablation sạch giữa:
  `dadnet.yaml`
  `dadnet_loss_tuned.yaml`
  `dadnet_batch_tuned.yaml`
  `dadnet_scheduler_tuned.yaml`
- So riêng `before rerank` và `after rerank`
- Tối ưu thêm `sampler`, `triplet margin`, `scheduler step`
- Nếu cần, tách riêng mô hình DADNet sang file chuyên biệt thay vì để chung trong [reid_model.py](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\src\models\reid_model.py)

## 11. Tài liệu tham khảo

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [MLflow Quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [MLflow Self Hosting Overview](https://mlflow.org/docs/latest/self-hosting/index.html)
