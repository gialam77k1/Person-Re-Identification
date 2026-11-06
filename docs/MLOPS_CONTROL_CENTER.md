# MLOps Control Center

Control Center là giao diện vận hành local cho toàn bộ vòng đời model v1 → v2. Người vận hành chọn
dataset, bấm `Start MLOps lifecycle`, theo dõi timeline/metrics/log và thực hiện approval, promotion
hoặc rollback trên web; không phải chạy script bằng terminal trong luồng vận hành bình thường.

## Thành phần

- `control-ui` (`http://localhost:3001`): Next.js/Tailwind dashboard.
- `control-api` (`http://localhost:8090`): FastAPI control plane.
- SQLite volume `reid-control-data`: cycle, stage, event và audit log bền vững.
- GitHub Actions self-hosted runner: thực thi pre-approval và retraining workload.
- MLflow, Prometheus, Qdrant, Triton và inference API: nguồn trạng thái cho dashboard.

Một cycle gồm hai workflow phía sau:

```text
Start
  → cross-domain-drift.yml
  → validate / embeddings / drift / evaluate-v1
  → WAITING_APPROVAL
  → Approve Retraining
  → retrain-v2.yml
  → train-v2 với callback epoch/metrics
```

Các stage evaluation v2, validation, ONNX, registry và staging sẽ tiếp tục được nối vào workflow
post-approval ở các giai đoạn tiếp theo; Control Center đã định nghĩa đầy đủ timeline và callback API.

## Cấu hình secret

Sao chép `.env.mlops.example` thành `.env.mlops` và thay mọi giá trị `change-me`. Đặc biệt:

- `CONTROL_GITHUB_TOKEN`: fine-grained PAT chỉ cấp quyền Actions read/write cho repository này.
- `CONTROL_CALLBACK_TOKEN`: chuỗi ngẫu nhiên dài, dùng xác thực callback từ runner.
- Không dùng token GitHub ở frontend và không commit `.env.mlops`.

Tạo GitHub Actions secret `CONTROL_CALLBACK_TOKEN` có cùng giá trị với `.env.mlops`. Tạo repository
variable `CONTROL_API_URL=http://localhost:8090` nếu runner và stack chạy trên cùng máy.

## Khởi động

```bash
docker compose --env-file .env.mlops -f docker-compose.mlops.yml up -d --build
docker compose --env-file .env.mlops -f docker-compose.mlops.yml ps
curl --fail http://localhost:8090/health
```

Mở `http://localhost:3001`, chọn MSMT17 và bấm `Start MLOps lifecycle`.

## API

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/control/overview` | Tổng hợp health, model, collection, drift và cycle gần nhất |
| `GET` | `/control/services` | Trạng thái và public link của các service |
| `GET` | `/control/cycles` | Danh sách cycle |
| `GET` | `/control/cycles/{id}` | Timeline bền vững của một cycle |
| `GET` | `/control/cycles/{id}/events` | Event và metrics phục vụ live polling |
| `GET` | `/control/cycles/{id}/audit` | Approval/promotion/rollback audit log |
| `POST` | `/control/cycles` | Tạo cycle idempotent và dispatch pre-approval |
| `POST` | `/control/cycles/{id}/approve-retraining` | Phê duyệt và dispatch retraining |
| `POST` | `/control/cycles/{id}/promote` | Promote khi staging PASS |
| `POST` | `/control/cycles/{id}/rollback` | Dispatch rollback production |
| `POST` | `/control/cycles/{id}/stages/{stage}` | Callback xác thực từ runner |

## Quy tắc vận hành

- Đóng trình duyệt không dừng cycle; state nằm ở backend và runner.
- Start dùng idempotency key nên không dispatch trùng cycle cùng dataset/ngày/release.
- Approval chỉ bật ở `WAITING_APPROVAL`.
- Promote chỉ bật sau `staging_test=PASS`.
- Callback runner phải gửi header `X-Control-Callback-Token` hợp lệ.
- Production vẫn yêu cầu xác nhận; drift không tự động promote model.
