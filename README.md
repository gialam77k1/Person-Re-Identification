# Person Re-Identification MLOps Baseline

Capstone này triển khai baseline cho bài toán person re-identification trên bộ dữ liệu `Market-1501` bằng `PyTorch`, có hỗ trợ:

- Huấn luyện mô hình ReID với `ResNet50`
- Đánh giá theo `Rank-1`, `Rank-5`, `Rank-10`, `mAP`
- Lưu checkpoint và metric
- Theo dõi thí nghiệm bằng `MLflow`
- Trích xuất embedding tham chiếu để phục vụ các bước nhận dạng tiếp theo

## 1. Yêu cầu hệ thống

Cập nhật theo trạng thái hiện tại ngày **2026-08-02**.

- Hệ điều hành: Windows 10/11
- Python: **3.10 - 3.12**
- Khuyến nghị: **Python 3.10** để tương thích tốt với cả `PyTorch` và `MLflow`
- `pip` đã được cập nhật

Luu y:

- `PyTorch` tren Windows hien tai chi ho tro Python `3.9 - 3.12`.
- `MLflow` hien tai yeu cau Python `3.10+`.
- Giao cua an toan nhat cho project nay la **Python 3.10 hoac 3.11**.

## 2. Cau truc du an

```text
KLTN/
|- configs/
|  |- baseline.yaml
|  |- baseline_smoke.yaml
|- datasets/
|  |- Market-1501-v15.09.15/
|- artifacts/
|- mlruns/
|- src/
|  |- train.py
|  |- evaluate.py
|  |- extract_reference.py
|- requirements.txt
|- README.md
```

## 3. Setup moi truong

### Cach 1: Dung `venv` (khuyen nghi, don gian)

Mo PowerShell tai thu muc project:

```powershell
cd "C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\KLTN"
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Neu PowerShell chan script, chay tam:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Cach 2: Dung `conda`

```powershell
conda create -n reid-mlops python=3.10 -y
conda activate reid-mlops
python -m pip install --upgrade pip
```

## 4. Cai thu vien

### Buoc 1: Cai PyTorch

#### Neu chi chay CPU

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Neu chay NVIDIA GPU

Chon lenh dung voi phien ban CUDA cua may tai trang chinh thuc cua PyTorch:

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)

Vi du voi CUDA 12.4, lenh thuong co dang:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Neu ban khong chac may dang dung CUDA nao, uu tien cai ban `CPU` truoc de project chay on dinh.

### Buoc 2: Cai cac thu vien con lai

```powershell
pip install -r requirements.txt
```

## 5. Kiem tra cai dat

```powershell
python -c "import torch, torchvision, mlflow, yaml, numpy, PIL, tqdm; print('torch =', torch.__version__); print('cuda =', torch.cuda.is_available())"
```

Neu khong bao loi la moi truong da san sang.

## 6. Dataset

Project dang duoc cau hinh de chay voi `Market-1501` tai:

```text
datasets/Market-1501-v15.09.15/
```

Trong file [configs/baseline.yaml](C:/Users/Gia%20Lam/Desktop/IUH%20Data/Năm%205%20-%20Kỳ%201/KLTN/configs/baseline.yaml) duong dan mac dinh la:

```yaml
data:
  root: datasets/Market-1501-v15.09.15
  train_dir: datasets/Market-1501-v15.09.15/bounding_box_train
  query_dir: datasets/Market-1501-v15.09.15/query
  gallery_dir: datasets/Market-1501-v15.09.15/bounding_box_test
```

Ban can dam bao trong thu muc dataset co du 3 folder:

- `bounding_box_train`
- `query`
- `bounding_box_test`

## 7. Cach chay project

### 7.1. Chay smoke test 1 epoch

Dung de kiem tra pipeline truoc khi train full:

```powershell
python src\train.py --config configs\baseline_smoke.yaml
```

### 7.2. Train day du

```powershell
python src\train.py --config configs\baseline.yaml
```

Ket qua se duoc luu tai:

- `artifacts/checkpoints/last_model.pth`
- `artifacts/checkpoints/best_model.pth`
- `artifacts/metrics/metrics_v1.json`

### 7.3. Danh gia checkpoint

```powershell
python src\evaluate.py --config configs\baseline.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

File ket qua danh gia se duoc luu tai:

- `artifacts/metrics/evaluation_latest.json`

### 7.4. Trich xuat reference embeddings

```powershell
python src\extract_reference.py --config configs\baseline.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

Ket qua se duoc luu tai:

- `artifacts/embeddings/reference_embeddings.npy`
- `artifacts/embeddings/reference_pids.npy`
- `artifacts/embeddings/reference_camids.npy`
- `artifacts/embeddings/reference_manifest.json`

## 8. Theo doi MLflow

File config hien tai dang bat MLflow trong [configs/baseline.yaml](C:/Users/Gia%20Lam/Desktop/IUH%20Data/Năm%205%20-%20Kỳ%201/KLTN/configs/baseline.yaml) voi:

```yaml
logging:
  enable_mlflow: true
  tracking_uri: sqlite:///mlruns/mlflow.db
```

De mo giao dien MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Sau do mo trinh duyet tai:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 9. Su dung moi truong co san trong repo

Neu ban da co moi truong noi bo tai `.conda/reid-mlops`, co the chay truc tiep:

```powershell
.\.conda\reid-mlops\python.exe src\train.py --config configs\baseline.yaml
```

Hoac:

```powershell
.\.conda\reid-mlops\python.exe src\evaluate.py --config configs\baseline.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

## 10. Ghi chu quan trong

- Nen chay `baseline_smoke.yaml` truoc khi train full de phat hien loi som.
- Lan dau train voi `pretrained: true`, `torchvision` co the tai trong so `ResNet50` ve `artifacts/cache/torch`.
- Neu khong co GPU, code van chay tren CPU nhung se cham hon rat nhieu.
- Neu muon doi dataset hoac batch size, sua trong file config YAML.

## Nguon tham khao cap nhat

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [MLflow Quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [MLflow Self Hosting Overview](https://mlflow.org/docs/latest/self-hosting/index.html)
