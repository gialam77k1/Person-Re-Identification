# MLOps Control Center

Control Center là giao diện vận hành local cho toàn bộ vòng đời lặp model vN → vN+1. Người vận hành import
batch trong Data Inbox, theo dõi timeline/metrics/log và thực hiện approval, promotion
hoặc rollback trên web; không phải chạy script bằng terminal trong luồng vận hành bình thường.

## Thành phần

- `control-ui` (`http://localhost:3001`): Next.js/Tailwind dashboard.
- `control-api` (`http://localhost:8090`): FastAPI control plane.
- SQLite volume `reid-control-data`: cycle, stage, event và audit log bền vững.
- GitHub Actions self-hosted runner: thực thi pre-approval và retraining workload.
- MLflow, Prometheus, Qdrant, Triton và inference API: nguồn trạng thái cho dashboard.

Một cycle gồm hai workflow phía sau:

```text
Import immutable batch
  → cross-domain-drift.yml
  → validate / embeddings / drift / evaluate-v1
  → WAITING_APPROVAL
  → Approve Retraining
  → retrain-v2.yml
  → verify evidence / approved decision / preflight
  → train-v2 với callback epoch/metrics
  → evaluate-v2 / validation gate / DVC push
  → MLflow alias candidate
```

Workflow post-approval dừng sau alias `candidate`. Nó không được gán `champion` và không deploy
production; export ONNX, parity, staging và production tiếp tục ở các gate sau của cycle.

Pre-approval tạo drift report từ Market-1501 reference embeddings và mẫu query DukeMTMC-reID. Report chứa
release/checksum model, checksum hai tập embedding, Git commit, DVC revision, seed, sample count và
toàn bộ threshold. File Prometheus được ghi vào `${MONITORING_STORAGE_PATH}/textfile` để
node-exporter expose cho Prometheus; dashboard **Person ReID - Model Behavior** hiển thị trạng thái,
giá trị drift và threshold. Một bản report cũng được log vào experiment `person-reid-drift` để audit,
nhưng quyết định DRIFT luôn do `generate_drift_report.py` tạo trước khi gọi MLflow.

Evidence pre-approval được upload dưới tên `mlops-evidence-{cycle_id}`. Control API lưu chính xác
GitHub run ID và artifact name. Khi người vận hành bấm Approve, Control API tải đúng artifact này,
kiểm tra checksum, chạy decision policy, lưu decision `RETRAIN_APPROVED` cùng approver/thời gian/SHA256
trong SQLite rồi mới dispatch retraining. Runner chỉ tải decision qua endpoint có callback token và
không được tự tạo approval. Runner tiếp tục kiểm tra Git commit, DVC revision, dataset và checkpoint
v1 trước preflight/training. Report JSON nhỏ được lưu bằng GitHub Actions artifact; checkpoint v2 được
lưu trong MLflow và DVC/MinIO.

## Cấu hình secret

Sao chép `.env.mlops.example` thành `.env.mlops` và thay mọi giá trị `change-me`. Đặc biệt:

- `CONTROL_GITHUB_TOKEN`: fine-grained PAT chỉ cấp quyền Actions read/write cho repository này.
- `CONTROL_CALLBACK_TOKEN`: chuỗi ngẫu nhiên dài, dùng xác thực callback từ runner.
- `CONTROL_HOST_DATASET_ROOT`: đường dẫn tuyệt đối tới `datasets/` trên máy self-hosted runner.
- Không dùng token GitHub ở frontend và không commit `.env.mlops`.

Tạo GitHub Actions secret `CONTROL_CALLBACK_TOKEN` có cùng giá trị với `.env.mlops`. Tạo repository
variable `CONTROL_API_URL=http://localhost:8090` nếu runner và stack chạy trên cùng máy.

## Khởi động

```bash
docker compose --env-file .env.mlops -f docker-compose.mlops.yml up -d --build
docker compose --env-file .env.mlops -f docker-compose.mlops.yml ps
curl --fail http://localhost:8090/health
```

Mở `http://localhost:3001/inbox`, chọn batch đã phát hiện và bấm `Import selected batch`.

## API

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/control/overview` | Tổng hợp health, model, collection, drift và cycle gần nhất |
| `GET` | `/control/services` | Trạng thái và public link của các service |
| `GET` | `/control/inbox` | Batch history, source phát hiện và policy validation |
| `POST` | `/control/inbox/import` | Validate batch, chống trùng và tự tạo cycle |
| `GET` | `/control/cycles` | Danh sách cycle |
| `GET` | `/control/cycles/{id}` | Timeline bền vững của một cycle |
| `GET` | `/control/cycles/{id}/events` | Event và metrics phục vụ live polling |
| `GET` | `/control/cycles/{id}/audit` | Approval/promotion/rollback audit log |
| `GET` | `/control/cycles/{id}/retraining-decision` | Runner lấy decision đã duyệt; yêu cầu callback token |
| `POST` | `/control/cycles` | Tạo cycle idempotent và dispatch pre-approval |
| `POST` | `/control/cycles/{id}/approve-retraining` | Phê duyệt và dispatch retraining |
| `POST` | `/control/cycles/{id}/promote` | Promote khi staging PASS |
| `POST` | `/control/cycles/{id}/rollback` | Dispatch rollback production |
| `POST` | `/control/cycles/{id}/stages/{stage}` | Callback xác thực từ runner |

## Quy tắc vận hành

- Đóng trình duyệt không dừng cycle; state nằm ở backend và runner.
- Fingerprint batch là idempotency key nên cùng dữ liệu/manifest không thể dispatch hai lần.
- Approval chỉ bật ở `WAITING_APPROVAL`.
- Approval bị từ chối nếu cycle không có workflow run ID và evidence artifact bất biến.
- UI hiển thị drift/evaluation evidence trước khi bật nút approval và decision SHA256 sau khi duyệt.
- Retraining có thể chạy lại an toàn: artifact v1 không bị ghi đè và run v2 hoàn chỉnh được tái sử dụng.
- Chỉ validation `PASS` mới được gán alias `candidate`; workflow không có quyền tự gán `champion`.
- Promote chỉ bật sau `staging_test=PASS`.
- Callback runner phải gửi header `X-Control-Callback-Token` hợp lệ.
- Production vẫn yêu cầu xác nhận; drift không tự động promote model.
