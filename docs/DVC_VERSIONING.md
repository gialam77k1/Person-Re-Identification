# Version hóa dữ liệu và model artifact bằng DVC

## Những thay đổi đã thực hiện

- Cài `dvc[s3]` trong môi trường `reid` và thêm dependency vào `requirements.txt`.
- Khởi tạo DVC, dùng remote `minio` tại `s3://dvc-storage/person-reid`.
- Credential chỉ được đọc từ `.env.mlops`; Git không lưu access key hoặc secret key.
- Version hóa Market-1501, checkpoint epoch 48, ONNX graph, external weights và reference embeddings kèm metadata.
- Bổ sung DVC hash vào manifest của `market1501-vit-bnneck:v1`.
- Mở rộng `src/verify_model_release.py` để kiểm tra cả SHA256 release lẫn DVC pointer.

## Artifact đang được quản lý

| Thành phần | DVC hash |
|---|---|
| Market-1501 | `0d61c823b34177c53e9e2161db8257b3.dir` |
| Checkpoint epoch 48 | `7a230c525a7c62deabeeceab31238be6` |
| ONNX graph | `1bc31348d93fa9abe4527a6dabe07fb3` |
| ONNX external weights | `06456b87da26789ec21374b0626219e1` |
| Reference embeddings và metadata | `33604888da92aa91c26805b9304a54ca.dir` |

## Cách sử dụng

Khởi động MinIO trước, sau đó dùng wrapper để tự động nạp credential local:

```bash
./scripts/mlops_stack.sh up
./scripts/dvc_local.sh status
./scripts/dvc_local.sh push
./scripts/dvc_local.sh pull
```

Kiểm tra SHA256 và DVC pointer của model v1:

```bash
conda run -n reid python src/verify_model_release.py
```

## Quy trình phục hồi an toàn

Không xóa artifact thật để thử nghiệm. Dùng một thư mục tạm và một bản clone cục bộ của repository, sau đó chạy `dvc pull` trong bản clone. Nếu muốn kiểm tra trực tiếp trong working tree, chỉ xóa đúng output đã được DVC quản lý và chạy ngay:

```bash
./scripts/dvc_local.sh pull
conda run -n reid python src/verify_model_release.py --repository-only
```

Không chạy `dvc gc` nếu chưa kiểm tra các Git revision và tag vẫn cần giữ.

## Kết quả nghiệm thu ngày 2026-10-11

- `dvc push` đã đẩy 66.215 object lên MinIO.
- `dvc status --cloud` xác nhận cache và remote `minio` đồng bộ.
- Một clone sạch trong thư mục tạm đã chạy `dvc pull`, tải 66.215 object và phục hồi 68.044 file/output.
- Market-1501 phục hồi đủ 68.036 file; reference bundle phục hồi đủ 5 file.
- SHA256 của checkpoint, ONNX graph và external weights sau phục hồi khớp model manifest v1.
