# Person Re-Identification with Baseline and DADNet-style Upgrade

Project nay trien khai pipeline person re-identification bang `PyTorch` tren bo du lieu `Market-1501`, bao gom:

- Baseline `ResNet50 -> Embedding -> Classifier`
- Bien the nang cap theo huong `DADNet-inspired`
- Train, evaluate, extract reference embeddings
- Luu checkpoint, metric, artifact
- Theo doi thi nghiem bang `MLflow`

Cập nhật theo trạng thái local ngày **2026-08-02**.

## 1. Tong quan kien truc

### Baseline

```text
Input
  -> ResNet50 Backbone
  -> Global Pooling
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
  -> Optional Local-Part Branch (3 stripes)
  -> Global Average Pooling
  -> DEM (Distinguishability Enhancement Module)
  -> Global-Local Fusion
  -> BatchNorm
  -> Classifier
```

Luu y:

- Ban `DADNet` trong repo nay la phien ban `inspired by` so do kien truc, khong phai ban tai hien full paper 100%.
- Muc tieu la giu pipeline gon, de train, de so sanh voi baseline trong do an/capstone.
- Ban hien tai uu tien giam nham lan giua quan ao, tui xach, ba lo bang cach bo sung `local-part branch` o head thay vi nhan them attention qua sau.

## 2. Cau truc du an

```text
Person-Re-Identification/
|- configs/
|  |- baseline.yaml
|  |- baseline_smoke.yaml
|  |- dadnet.yaml
|  |- dadnet_smoke.yaml
|- datasets/
|  |- Market-1501-v15.09.15/
|- artifacts/
|- mlruns/
|- src/
|  |- train.py
|  |- evaluate.py
|  |- extract_reference.py
|  |- models/reid_model.py
|- requirements.txt
|- .gitignore
|- README.md
```

## 3. Yeu cau he thong

- Windows 10/11
- Python: `3.10` khuyen nghi nhat
- GPU NVIDIA la tuy chon, khong bat buoc

Ghi chu:

- `PyTorch` tren Windows nen di voi Python `3.10 - 3.12`.
- `MLflow` yeu cau Python `3.10+`.
- Moi truong da xac nhan chay duoc tren may nay la `C:\tmp\reid-mlops`.

## 4. Cai dat moi truong

### Cach 1: Dung moi truong da co san

```powershell
conda activate C:\tmp\reid-mlops
```

Kiem tra nhanh:

```powershell
python -c "import torch, torchvision, mlflow, yaml, numpy, PIL, tqdm; print('torch =', torch.__version__); print('torchvision =', torchvision.__version__); print('cuda =', torch.cuda.is_available())"
```

### Cach 2: Tao moi bang conda

```powershell
conda create -n reid-mlops python=3.10 -y
conda activate reid-mlops
python -m pip install --upgrade pip
```

Neu chay CPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Neu chay GPU NVIDIA, chon lenh phu hop tu trang PyTorch:

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)

Sau do cai cac thu vien con lai:

```powershell
pip install -r requirements.txt
```

## 5. Dataset

Config mac dinh hien tai dung:

```text
datasets/Market-1501-v15.09.15/
```

Can dam bao co du:

- `bounding_box_train`
- `query`
- `bounding_box_test`

Duong dan dang duoc khai bao trong [baseline.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\baseline.yaml) va [dadnet.yaml](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\configs\dadnet.yaml).

## 6. Cach chay

### Smoke test baseline

```powershell
conda activate C:\tmp\reid-mlops
python src\train.py --config configs\baseline_smoke.yaml
```

### Smoke test DADNet-inspired

```powershell
conda activate C:\tmp\reid-mlops
python src\train.py --config configs\dadnet_smoke.yaml
```

### Train full baseline

```powershell
conda activate C:\tmp\reid-mlops
python src\train.py --config configs\baseline.yaml
```

### Train full DADNet-inspired

```powershell
conda activate C:\tmp\reid-mlops
python src\train.py --config configs\dadnet.yaml
```

### Evaluate checkpoint

Baseline:

```powershell
python src\evaluate.py --config configs\baseline.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

DADNet-inspired:

```powershell
python src\evaluate.py --config configs\dadnet.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

### Extract reference embeddings

Baseline:

```powershell
python src\extract_reference.py --config configs\baseline.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

DADNet-inspired:

```powershell
python src\extract_reference.py --config configs\dadnet.yaml --checkpoint artifacts\checkpoints\best_model.pth
```

## 7. Ket qua smoke test da kiem tra

Tren may local hien tai:

- Baseline smoke: `rank1=41.86%`, `mAP=23.03%`
- DADNet ban dau: `rank1=37.44%`, `mAP=20.78%`
- DADNet sau tinh chinh: `rank1=45.01%`, `mAP=25.65%`

Dieu nay cho thay ban `DADNet-inspired` sau khi tinh chinh da vuot baseline trong smoke test 1 epoch.

Luu y:

- Smoke test chi la moc kiem tra nhanh.
- Ket luan cuoi cung nen dua tren train full va evaluate cung dieu kien.

## 8. File output

Sau khi train/evaluate, artifact duoc luu trong:

- `artifacts/checkpoints/last_model.pth`
- `artifacts/checkpoints/best_model.pth`
- `artifacts/metrics/metrics_v1.json`
- `artifacts/metrics/evaluation_latest.json`
- `artifacts/embeddings/reference_embeddings.npy`
- `artifacts/embeddings/reference_pids.npy`
- `artifacts/embeddings/reference_camids.npy`
- `artifacts/embeddings/reference_manifest.json`

## 9. MLflow

De mo giao dien MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Sau do mo trinh duyet tai:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 10. Git va push code

Repo da co [`.gitignore`](C:\Users\Gia Lam\Desktop\IUH Data\Năm 5 - Kỳ 1\Person-Re-Identification\.gitignore) de tranh day len:

- `datasets/`
- `artifacts/`
- `mlruns/`
- `.conda/`
- cac file model nhu `*.pth`, `*.npy`, `*.pt`

Neu ban muon tao repo moi:

```powershell
git init
git add .
git status
```

Hay kiem tra `git status` truoc khi commit de chac rang dataset va checkpoint khong bi add.

## 11. Huong phat trien tiep

- Train full `dadnet.yaml` va so sanh voi `baseline.yaml`
- Them ablation:
  `baseline`
  `baseline + CFT`
  `baseline + CFT + DEM`
- Ghi bang so sanh `Rank-1`, `Rank-5`, `Rank-10`, `mAP`
- Neu can, tach rieng model moi thanh `dadnet_reid.py`

## Tai lieu tham khao

- [PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)
- [MLflow Quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/)
- [MLflow Self Hosting Overview](https://mlflow.org/docs/latest/self-hosting/index.html)
