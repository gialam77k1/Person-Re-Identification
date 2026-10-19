# Monitoring cho Person Re-Identification

Giai đoạn 9 bổ sung quan sát hệ thống, hành vi model, cảnh báo và báo cáo embedding drift định kỳ cho stack local.

## Những thay đổi đã thực hiện

- FastAPI xuất Prometheus metrics tại `GET /metrics`.
- Prometheus thu thập API, Triton, Qdrant, host, container, GPU và Alertmanager.
- Grafana tự provision datasource cùng hai dashboard `Person ReID - System` và `Person ReID - Model Behavior`.
- Alertmanager nhận 9 alert rules về availability, error rate, latency, collection, no-match và drift.
- API lấy mẫu embedding thành JSONL có giới hạn; không lưu ảnh.
- GitHub Actions chạy drift report mỗi ngày, đồng thời có thể chạy thủ công.
- Drift report được ghi thành JSON và Prometheus textfile để dashboard/alert sử dụng.

## Khởi động monitoring

Cập nhật `.env.mlops`, đặc biệt đổi mật khẩu Grafana:

```bash
cp .env.mlops.example .env.mlops
```

Sau đó chạy stack:

```bash
docker compose --env-file .env.mlops -f docker-compose.mlops.yml up -d
docker compose --env-file .env.mlops -f docker-compose.mlops.yml ps
```

Các giao diện local:

- Grafana: `http://localhost:3000`.
- Prometheus: `http://localhost:9090`.
- Alertmanager: `http://localhost:9093`.
- API metrics: `http://localhost:8080/metrics`.
- Triton metrics: `http://localhost:8002/metrics`.

Prometheus và Grafana dùng named volume nên dữ liệu không mất khi restart container.

## System metrics

Dashboard `Person ReID - System` hiển thị:

- Request throughput và tỷ lệ HTTP 5xx.
- Latency API p50, p95, p99.
- Latency p95 riêng cho Triton và Qdrant.
- Readiness của manifest, Triton server/model và Qdrant collection.
- Số point trong Qdrant.
- CPU/RAM host và CPU container.
- GPU utilization và GPU memory từ DCGM exporter.

Throughput được tính từ `rate(reid_api_requests_total[5m])`, không cần một counter riêng dễ sai lệch.

## Model metrics

Dashboard `Person ReID - Model Behavior` hiển thị:

- Release ID và Git commit đang phục vụ.
- Raw embedding norm trước L2 normalization.
- Similarity score quantiles và histogram.
- Query rate theo `matched`/`no_match`.
- Tỷ lệ no-match dựa trên `API_MATCH_THRESHOLD`.
- Kết quả và từng thống kê của drift report gần nhất.

Giá trị khởi đầu:

```text
API_MATCH_THRESHOLD=0.5
API_DRIFT_SAMPLE_RATE=0.05
API_DRIFT_MAX_SAMPLES=10000
```

Các giá trị này cần được hiệu chỉnh bằng dữ liệu production có nhãn; `0.5` hiện là ngưỡng quan sát, không thay đổi kết quả top-k trả về.

## Alert ban đầu

- Target Prometheus không scrape được trong 2 phút.
- Thành phần serving unhealthy trong 2 phút.
- HTTP 5xx vượt 5% trong 5 phút.
- API p95 vượt 500 ms trong 5 phút.
- Triton/Qdrant p95 vượt 250 ms trong 5 phút.
- Qdrant collection rỗng.
- No-match vượt 40% trong 15 phút.
- Drift policy thất bại.
- Drift report cũ hơn 48 giờ.

Alert xuất hiện trong Prometheus và Alertmanager. Receiver `local-ui` chỉ lưu/hiển thị local; có thể bổ sung email, Slack hoặc webhook vào `monitoring/alertmanager/alertmanager.yml` khi có credential phù hợp. Không commit credential.

## Embedding drift

API lấy mẫu embedding đã normalize vào:

```text
${MONITORING_STORAGE_PATH}/samples/current_embeddings.jsonl
```

Thư mục mặc định `../person-reid-monitoring` nằm ngoài Git checkout để không bị `actions/checkout` xóa. File có permission `0600`, xoay vòng khi đạt số mẫu tối đa và không chứa ảnh hay person ID. Embedding vẫn là dữ liệu nhạy cảm; cần giới hạn quyền truy cập và chính sách retention phù hợp.

Chạy report thủ công:

```bash
./scripts/drift_report_local.sh \
  market1501-vit-bnneck:v1-dvc \
  market1501-vit-bnneck-v1-dvc
```

Policy nằm tại `configs/drift_detection.yaml`. Report so sánh reference và current theo:

- Khoảng cách cosine giữa hai centroid.
- Mean shift đã chuẩn hóa theo độ lệch chuẩn reference.
- Chênh lệch trung bình embedding norm.
- Kolmogorov–Smirnov statistic trên phân bố cosine tới reference centroid.

Output:

```text
../person-reid-monitoring/reports/drift_latest.json
../person-reid-monitoring/textfile/reid_drift.prom
```

Node exporter đọc textfile thứ hai để Prometheus scrape. Workflow `.github/workflows/drift.yml` chạy hằng ngày lúc 02:15 UTC; self-hosted runner cần secret `MLOPS_ENV_FILE` như release workflow.

Không nên promote hoặc rollback model chỉ dựa vào drift tự động. Drift là tín hiệu cần điều tra cùng traffic mix, camera mới, ánh sáng và dữ liệu có nhãn.

## Kiểm tra cấu hình

```bash
./scripts/ci.sh

docker run --rm \
  -v "$PWD/monitoring/prometheus:/etc/prometheus:ro" \
  --entrypoint promtool prom/prometheus:v3.5.0 \
  check config /etc/prometheus/prometheus.yml
```

