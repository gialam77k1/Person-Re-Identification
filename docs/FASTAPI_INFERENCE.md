# FastAPI inference service

Service cung cấp embedding và truy hồi người online qua Triton và Qdrant. Model contract không được hard-code trong API mà được đọc từ model manifest đã qua validation gate.

## Luồng xử lý

```text
JPG/PNG multipart upload
  -> kiểm tra MIME, định dạng, dung lượng và số pixel
  -> RGB + letterbox + ImageNet normalization theo preprocessing contract
  -> Triton HTTP inference
  -> kiểm tra shape/NaN/Inf và L2 normalize
  -> Qdrant cosine query
  -> top-k kèm person ID, camera ID, score và ảnh nguồn
```

Re-ranking không nằm trong online serving path.

## Khởi động

Đảm bảo `.env.mlops` dùng model repository và manifest của release hiện tại:

```dotenv
TRITON_MODEL_REPOSITORY=./artifacts/pipeline/market1501-vit-bnneck-v1-dvc/triton/model_repository
API_MODEL_MANIFEST_PATH=artifacts/pipeline/market1501-vit-bnneck-v1-dvc/release/model_manifest.json
QDRANT_COLLECTION=reid_reference_v1
```

Nếu artifact chưa có ở local:

```bash
./scripts/dvc_local.sh pull
```

Khởi động toàn bộ stack:

```bash
./scripts/mlops_stack.sh up
./scripts/mlops_stack.sh verify
```

API mặc định ở `http://localhost:8080`; OpenAPI UI ở `http://localhost:8080/docs`.

Có thể chạy API trực tiếp trong môi trường Conda khi Triton và Qdrant đã chạy:

```bash
conda run --no-capture-output -n reid \
  uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## Endpoint

### Liveness

```bash
curl --fail http://localhost:8080/health
```

`/health` chỉ xác nhận process API đang hoạt động.

### Readiness

```bash
curl --fail http://localhost:8080/ready
```

Endpoint trả HTTP `200` chỉ khi manifest đã nạp, Triton server sẵn sàng, model Triton sẵn sàng và collection Qdrant ở trạng thái `green`, đúng 512 chiều, có ít nhất một point. Nếu một điều kiện không đạt, endpoint trả HTTP `503` và chi tiết từng check.

### Model info

```bash
curl --fail http://localhost:8080/model-info
```

Response gồm release ID, Git commit, model name, embedding dimension, input shape, preprocessing schema version/SHA256 và collection đang dùng.

### Embedding

```bash
curl --fail -X POST http://localhost:8080/embed \
  -F 'image=@datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg;type=image/jpeg'
```

Embedding trả về đã được L2 normalize và có đúng 512 chiều.

### Search

```bash
curl --fail -X POST 'http://localhost:8080/search?top_k=5' \
  -F 'image=@datasets/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg;type=image/jpeg'
```

Ví dụ response:

```json
{
  "model_release": "market1501-vit-bnneck:v1-dvc",
  "top_k": 5,
  "results": [
    {
      "person_id": 1,
      "camera_id": 2,
      "similarity_score": 0.94,
      "source_image": "bounding_box_test/0001_c2s1_000301_00.jpg"
    }
  ]
}
```

`top_k` mặc định là 5 và không được vượt quá 100. Các giới hạn có thể đổi bằng biến môi trường trong `.env.mlops`.

## Image validation

- Chỉ nhận multipart field tên `image` với MIME `image/jpeg` hoặc `image/png`.
- Kiểm tra format thực bên trong file phải khớp MIME khai báo.
- Mặc định từ chối file lớn hơn 10 MiB.
- Mặc định từ chối ảnh lớn hơn 25 triệu pixel.
- File rỗng, file hỏng, embedding sai shape, NaN/Inf hoặc norm bằng 0 đều bị từ chối.
- Khi khởi động, service đối chiếu SHA256 của ONNX graph, external weights và preprocessing contract với model manifest; artifact bị sửa sẽ làm service fail-fast.

## Kiểm thử

```bash
conda run --no-capture-output -n reid \
  python -m unittest -v tests.test_inference_api
```

Bộ test dùng fake Triton/Qdrant, không cần GPU hoặc container đang chạy. Integration test với dependency thật thuộc Giai đoạn 7.

## Kết quả nghiệm thu Giai đoạn 6

Nghiệm thu local ngày 2026-10-17 với Triton và Qdrant thật:

- Docker image `person-reid/inference-api:1.0.0` build thành công.
- Container API healthy bằng chính endpoint `/ready`.
- `/ready` PASS đủ manifest, Triton server, Triton model và Qdrant collection.
- `/model-info` trả release `market1501-vit-bnneck:v1-dvc`, Git commit `3bc5cd9e68d4b186900ba52e89976e9c94be28cc` và preprocessing schema version `1`.
- `/embed` trả vector 512 chiều, norm đo được `1,00000004`.
- `/search?top_k=5` trả đúng 5 point từ collection `reid_reference_v1`, có đủ PID, camera ID, similarity score và source image.
- 6 API tests và 6 model-gate regression tests đều PASS.

Benchmark latency, deployment quality regression và restart test đầy đủ được ghi tại [`DEPLOYMENT_TESTING.md`](DEPLOYMENT_TESTING.md).
