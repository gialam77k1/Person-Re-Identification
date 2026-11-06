# Các bước cuối để hoàn thành dự án Person Re-Identification

Tài liệu này mô tả vòng MLOps hoàn chỉnh từ dữ liệu mới, phát hiện drift, phê duyệt retraining,
train model v2, kiểm định, triển khai đến giám sát và rollback. Dataset, checkpoint, ONNX, embedding
và report lớn chỉ được lưu bằng DVC/MinIO hoặc MLflow, không commit trực tiếp vào Git.

## Nguyên tắc bắt buộc

- MLflow không tự phát hiện drift và không tự khởi chạy training. MLflow dùng để lưu lineage, config,
  metrics, artifact, checkpoint và Model Registry.
- Prometheus/Grafana hiển thị drift và cảnh báo.
- GitHub Actions trên self-hosted runner hoặc một orchestrator tương đương điều khiển retraining.
- Drift không được phép tự động đưa model lên production.
- Retraining chỉ chạy khi evidence hợp lệ và decision gate trả về `RETRAIN_APPROVED`.
- Model chỉ được deploy khi validation gate trả về `PASS`.
- Production luôn cần manual approval và phải rollback được về model v1.
- Sau bước bootstrap ban đầu, người vận hành không chạy script bằng terminal. Mọi thao tác bình thường
  từ v1 → v2 được bắt đầu, theo dõi, phê duyệt và retry trên Control Center.

## Trạng thái hiện tại

- [x] Model v1 đã được đóng băng, có manifest và checksum.
- [x] Stack local gồm PostgreSQL, MinIO, MLflow, Qdrant, Triton, API, Prometheus và Grafana.
- [x] Dataset Market-1501 và MSMT17 được quản lý bằng DVC/MinIO.
- [x] Có drift detector, cross-domain evaluation và retraining decision gate.
- [x] Drift MSMT17 đã được xác nhận và decision gate đã trả về `RETRAIN_APPROVED`.
- [x] Self-hosted runner có thể chạy smoke test và workload GPU.
- [x] Script retraining v2 đã có cấu hình tối đa 20 epoch và preflight đã PASS.
- [x] MLOps Control Center đã có UI, Control API, cycle/event/audit store và workflow dispatch.
- [ ] Chưa có workflow nối tự động approval → retraining → validation → registry.
- [ ] Chưa hoàn tất toàn bộ release model v2 và deployment staging/production.

## Luồng MLOps cần đạt

```text
Người dùng chọn dataset và bấm Start MLOps lifecycle
  → dữ liệu mới hoặc embedding từ API
  → drift report
  → Prometheus/Grafana alert
  → evaluation evidence
  → decision gate: REVIEW_REQUIRED
  → UI dừng tại WAITING_APPROVAL
  → người dùng bấm Approve Retraining: RETRAIN_APPROVED
  → self-hosted runner retrain model v2
  → MLflow tracking
  → evaluation + validation gate
  → MLflow Model Registry: candidate
  → deploy staging + integration test
  → người dùng bấm Promote to Production
  → champion hoặc rollback
```

## 1. Xây dựng MLOps Control Center

Xây một UI đơn giản làm cổng điều khiển chung, không thay thế MLflow, Grafana, MinIO hoặc GitHub
Actions. Người vận hành không phải mở terminal: UI đọc trạng thái từ các hệ thống này, gọi Control
API để khởi động workflow và hiển thị toàn bộ tiến trình từ model v1 đến model v2.

Kiến trúc:

```text
Next.js + Tailwind Control Center
  → FastAPI Control API
  → GitHub Actions trên self-hosted runner
  → drift / approval / retrain / validation / deploy

Control Center
  ├── MLflow API: run, metrics, artifact và Model Registry
  ├── Prometheus API: health, latency, drift và resource metrics
  ├── GitHub Actions API: workflow run và trạng thái từng job
  ├── Qdrant API: collection và vector count
  └── Triton API: model readiness và model version
```

Các màn hình tối thiểu:

- **Overview:** service health, model đang phục vụ, drift gần nhất và pipeline gần nhất.
- **Pipeline:** timeline từng stage, log tóm tắt và trạng thái `PENDING/RUNNING/PASS/FAIL`.
- **Models:** v1/v2, metrics, MLflow run, DVC revision và alias `candidate/champion`.
- **Monitoring:** các biểu đồ chính lấy từ Prometheus; có link mở dashboard Grafana đầy đủ.
- **Services:** link mở MLflow, Grafana, MinIO, Qdrant và GitHub Actions trong tab riêng.

Ba thao tác điều khiển:

1. `Start MLOps lifecycle`: chọn dataset rồi khởi động cycle từ model v1; workflow tự chạy cho đến
   gate cần phê duyệt, sau đó tự tiếp tục tới model v2 và staging.
2. `Promote to production`: chỉ bật khi staging PASS và phải xác nhận thủ công.
3. `Rollback to v1`: khôi phục đồng bộ API image, Triton model và Qdrant collection v1.

Trải nghiệm pipeline trên UI:

```text
✓ Validate dataset
✓ Extract current embeddings
✓ Detect drift
✓ Evaluate model v1 on MSMT17
! Waiting for retraining approval       [Approve Retraining]
● Train model v2 — Epoch 7/20           [View live logs]
○ Evaluate model v2
○ Validation gate
○ Export ONNX and verify parity
○ Register MLflow candidate
○ Deploy and test staging
```

Trong lúc chạy, UI phải hiển thị progress bar, epoch hiện tại, train/validation loss, Rank-1, mAP,
GPU utilization, thời gian đã chạy, log gần nhất và link tới MLflow run. Trình duyệt có thể đóng mà
cycle vẫn tiếp tục trên runner; khi mở lại UI phải phục hồi đúng trạng thái từ backend.

Một `cycle_id` có thể điều phối nhiều runner workflow phía sau: workflow pre-approval chạy validate,
drift và evaluation; workflow post-approval chạy retraining, validation và staging. Đây vẫn là một
cycle liên tục trên UI, người dùng không phải biết hoặc khởi chạy từng workflow riêng.

Quy tắc an toàn:

- Frontend không được chạy shell trực tiếp và không chứa GitHub token hoặc secret.
- Control API giữ credential phía server và chỉ được gọi các workflow đã cho phép.
- Mỗi lần bấm tạo một `cycle_id` duy nhất để theo dõi, audit và chống chạy trùng.
- Nút Start phải idempotent; không tạo hai retraining run cho cùng một decision artifact.
- Stage lỗi phải hiện nguyên nhân và nút `Retry`, nhưng chỉ retry đúng stage hoặc các stage phụ thuộc.
- Không nhúng UI khác bằng iframe nếu hệ thống chặn `X-Frame-Options/CSP`; dùng API tổng hợp và link
  mở UI gốc.
- Không được dùng một nút để bỏ qua approval hoặc promote thẳng lên production.

API điều khiển tối thiểu:

```text
GET  /control/overview
GET  /control/services
GET  /control/cycles
GET  /control/cycles/{cycle_id}
GET  /control/cycles/{cycle_id}/events
POST /control/cycles
POST /control/cycles/{cycle_id}/approve-retraining
POST /control/cycles/{cycle_id}/promote
POST /control/cycles/{cycle_id}/rollback
```

Thêm `control-ui` và `control-api` vào Docker Compose, dùng port không xung đột với Grafana `3000`,
MLflow `5000` và inference API `8080`.

Gate hoàn thành:

- Mở một URL duy nhất có thể thấy trạng thái các service và model đang phục vụ.
- Bấm `Start MLOps lifecycle` tạo đúng một cycle; mỗi stage chỉ tạo workflow run cần thiết một lần,
  không dispatch trùng.
- UI cập nhật được trạng thái từng stage mà không cần mở terminal.
- UI hiển thị live progress/log/metrics từ v1 đến v2 và phục hồi được cycle sau khi reload trang.
- Approval, promotion và rollback có xác nhận, audit log và thông báo PASS/FAIL rõ ràng.
- UI vẫn cung cấp link mở MLflow/Grafana/GitHub Actions để điều tra chi tiết.

## 2. Hoàn thiện workflow retraining khép kín

**Trạng thái implementation: hoàn thành.** Workflow đã ràng buộc approval với evidence artifact theo
`cycle_id`/GitHub run ID, xác minh checksum + Git/DVC lineage, chạy preflight → train → evaluate →
validation, lưu checkpoint qua MLflow/DVC và chỉ gán alias `candidate` khi gate `PASS`. Cần chạy một
cycle thật trên self-hosted GPU runner để nghiệm thu runtime.

Tạo workflow retraining chạy trên self-hosted GPU runner. Sau khi workflow pre-approval hoàn thành,
Control API giữ cycle ở `WAITING_APPROVAL`; nút `Approve Retraining` sẽ tạo decision artifact và tự
kích hoạt workflow retraining. Các script bên dưới chỉ là implementation nội bộ và người vận hành
không phải chạy trực tiếp. Workflow retraining phải:

1. Nhận release ID, đường dẫn drift report, evaluation report và retraining decision.
2. Kiểm tra đúng Git commit, DVC revision, dataset và base checkpoint.
3. Từ chối chạy nếu decision không phải `RETRAIN_APPROVED`.
4. Backend chạy `./scripts/train_msmt17_v2_local.sh --check` trước khi train.
5. Backend chạy `./scripts/train_msmt17_v2_local.sh` và truyền cấu hình release v2.
6. Chạy evaluation và validation gate sau training.
7. Upload report nhỏ làm GitHub Actions artifact; model lớn lưu bằng MLflow/DVC/MinIO.
8. Chỉ đăng ký alias `candidate` khi validation `PASS`.

Workflow không được tự gán `champion` và không được tự deploy production.

Gate hoàn thành:

- Một workflow run thể hiện đủ drift evidence, approval, training, validation và MLflow run ID.
- Tất cả stage và log tóm tắt được phản ánh lại trên Control Center.
- Workflow bị chặn rõ ràng khi thiếu approval hoặc checksum/DVC revision không khớp.
- Workflow có thể chạy lại an toàn và không ghi đè release v1.

## 3. Xác nhận drift bằng dữ liệu MSMT17

**Trạng thái implementation: hoàn thành.** Drift detector nhận metadata của mẫu MSMT17 và manifest
model v1, ghi trực tiếp Git commit, DVC revision, seed, số mẫu, threshold, checksum model/reference/
current embeddings vào report. Metrics được xuất vào textfile directory của node-exporter để
Prometheus scrape và dashboard Grafana hiển thị cả observed value lẫn threshold. Workflow đồng thời
lưu report vào experiment MLflow `person-reid-drift`; MLflow chỉ lưu dấu vết, không tham gia quyết
định `drift_detected`.

Kết quả kiểm tra lại bằng 1.000 ảnh query MSMT17 và 12.936 reference embeddings Market-1501:

- `drift_detected=true`.
- Centroid cosine distance `0,5532`, vượt ngưỡng `0,05`.
- Standardized mean shift `0,3698`, vượt ngưỡng `0,20`.
- KS statistic `0,2034`, vượt ngưỡng `0,15`.
- ONNX v1 có SHA256
  `e8425d8f645dc419225d02f2e586ceb3a244955ee5e292141951a5c7aaf1f4be`.
- Dataset revision `c8a4c6f18b86490a45f940ab9fb0eb70.dir`, seed `42`.

Luồng detection phải chạy trước retraining:

```text
Market-1501 reference embeddings
  + MSMT17 current embeddings
  → generate_drift_report.py
  → drift_report.json
  → reid_cross_dataset_drift.prom
```

Kết quả cần có:

- `drift_detected=true`.
- Release ID và checksum của model v1 đang được kiểm tra.
- Số lượng sample, seed và ngưỡng drift.
- Git commit, DVC revision và thời gian tạo report.
- Drift metrics xuất hiện trên Prometheus/Grafana.

MLflow có thể lưu thêm drift report trong một experiment riêng để truy vết, nhưng MLflow không thay
thế drift detector.

## 4. Gate phê duyệt retraining

**Trạng thái implementation: hoàn thành.** Control API tải đúng GitHub evidence artifact theo run ID,
xác minh checksum rồi chạy decision policy ở backend. Decision `RETRAIN_APPROVED` được lưu bền vững
với approver, thời gian, model nguồn, dataset đích, giới hạn 20 epoch và checksum evidence. Runner
chỉ được tải decision qua endpoint có callback token và phải xác minh SHA256 trước preflight; workflow
không còn gọi `retraining_decision.sh` để tự phê duyệt.

Decision gate kết hợp drift report và cross-domain evaluation. Trạng thái đầu tiên phải là
`REVIEW_REQUIRED`; Control Center chuyển cycle sang `WAITING_APPROVAL` và hiển thị evidence cùng
nút `Approve Retraining`. Khi người phụ trách bấm nút và xác nhận, Control API gọi decision gate ở
backend, tạo decision artifact có output `RETRAIN_APPROVED` rồi tự tiếp tục workflow.

Decision artifact phải ghi người phê duyệt, thời gian, model nguồn, dataset đích, số epoch tối đa và
checksum của evidence. Workflow retraining chỉ được đọc artifact này, không được bỏ qua gate bằng
một biến boolean tùy ý. Lệnh `scripts/retraining_decision.sh` chỉ được backend sử dụng, không phải
thao tác vận hành hằng ngày.

## 5. Retrain model v2 tự động bằng self-hosted runner

**Trạng thái implementation: hoàn thành.** Sau approval, workflow self-hosted tự preflight và train
không cần lệnh terminal. Preflight khóa đúng training contract; mỗi epoch gửi loss, backbone/head LR,
mAP, GPU, elapsed time, log gần nhất và link MLflow về Control Center. Completion gate chỉ cho stage
`train_v2` PASS khi MLflow `FINISHED`, đủ sáu artifact, transfer đúng epoch 48/reset classifier,
metrics hữu hạn, early-stop metadata hợp lệ và log không có CUDA/OOM. Việc sinh checkpoint v2 thật
vẫn cần chạy một lifecycle trên GPU runner.

Sau khi người dùng phê duyệt trên UI, Control API phải tiếp tục cycle và runner tự bắt đầu training;
không yêu cầu chạy thêm lệnh trong terminal. Control Center chuyển sang màn hình training và cập
nhật epoch, loss, learning rate, mAP, GPU và log gần nhất.

Cấu hình retraining:

- Base checkpoint: model v1 tại epoch 48.
- Dataset: MSMT17 train, 30.248 ảnh và 1.041 ID.
- Classifier Market-1501 cũ bị loại bỏ và khởi tạo lại cho 1.041 ID.
- Epoch 1–2: freeze 8 block ViT thấp.
- Từ epoch 3: unfreeze toàn model; learning rate backbone bằng `0,1 ×` learning rate head.
- Tối đa 20 epoch, warmup 2 epoch, cosine scheduler.
- Early stopping theo mAP với patience 3.
- Release: `market1501-msmt17-vit-bnneck:v2`.

Training hoàn thành khi MLflow run ở trạng thái `FINISHED`, không có lỗi CUDA/OOM, metrics hữu hạn
và có đủ:

```text
artifacts/msmt17/market1501-msmt17-vit-bnneck-v2/
├── checkpoints/best_model.pth
├── checkpoints/last_model.pth
├── logs/effective_config.json
├── logs/mlflow_run.json
├── logs/transfer_report.json
└── metrics/metrics_v2.json
```

Early stopping trước epoch 20 là hợp lệ nếu epoch tốt nhất và nguyên nhân dừng được ghi trong MLflow
và `metrics_v2.json`.

## 6. Đánh giá v2 và kiểm tra catastrophic forgetting

**Trạng thái implementation: hoàn thành.** Workflow đánh giá cùng SHA256 checkpoint v2 trên MSMT17
và Market-1501, ghi Rank-1/Rank-5/mAP/mINP, query/gallery count, Git commit và DVC revision. Policy
`configs/catastrophic_forgetting.yaml` được version hóa trước khi chạy: v2 phải tăng tối thiểu Rank-1
`0,05` và mAP `0,03` trên MSMT17; mức giảm tuyệt đối tối đa trên Market-1501 lần lượt là Rank-1
`0,05`, Rank-5 `0,03`, mAP `0,08`, mINP `0,10`. Cả hai report, policy và gate report được log
vào đúng MLflow training run kể cả khi gate FAIL; chỉ PASS mới được đi tiếp tới validation/registry.

Đánh giá cùng checkpoint tốt nhất trên:

- MSMT17 để đo mức cải thiện so với model v1 cross-domain.
- Market-1501 để đo suy giảm sau domain adaptation.

Mỗi report cần Rank-1, Rank-5, mAP, mINP, query/gallery count, checkpoint checksum, Git commit và
DVC revision. Ngưỡng chấp nhận phải được ghi trong config trước khi xem kết quả, không điều chỉnh
ngưỡng sau để làm model PASS.

Gate:

- v2 cải thiện rõ trên MSMT17 so với v1 cross-domain.
- Mức suy giảm trên Market-1501 không vượt ngưỡng catastrophic-forgetting đã cấu hình.
- Tất cả metrics hữu hạn và report được log vào cùng MLflow run/release.

## 7. Tạo release pipeline v2 bằng DVC

**Trạng thái implementation: hoàn thành.** `dvc.yaml` có graph v2 độc lập từ validate hai dataset,
import checkpoint đã train, đánh giá MSMT17/Market-1501, export/parity ONNX, extract reference,
release gate đến package Triton version 2. Mọi output nằm dưới
`artifacts/pipeline/market1501-msmt17-vit-bnneck-v2`; pipeline v1 không bị sửa hoặc ghi đè. Các nhánh
evaluation/export/reference chỉ hội tụ ở validation nên thay đổi serving không kích hoạt retraining.
Manifest v2 chứa Git commit, DVC revision của cả hai dataset, MLflow run ID và SHA256 effective config.
Chi tiết chạy và graph nằm tại `docs/DVC_V2_PIPELINE.md`.

Pipeline phải tách output v2 khỏi v1:

```text
validate_data
  → import hoặc train checkpoint_v2
  → evaluate_msmt17
  → evaluate_market1501
  → export_onnx
  → verify_onnx
  → extract_reference
  → validate_release
  → package_triton
```

Gate: `dvc repro` chỉ chạy lại stage bị ảnh hưởng; artifact v1 không bị ghi đè; manifest v2 chứa
dataset revision, Git commit, MLflow run ID và config hash.

## 8. Export, parity và validation model v2

Tạo ONNX graph, external weights, preprocessing contract, reference embeddings 512 chiều, metadata,
model manifest và SHA256. Validation gate bắt buộc kiểm tra:

- Quality metrics đạt ngưỡng v2.
- Embedding đúng 512 chiều, không NaN/Inf và đã L2-normalize.
- PyTorch–ONNX cosine parity tối thiểu `0.9999`.
- Max absolute error tối đa `0.0001`.
- Input/output và preprocessing contract khớp.
- Checksum, Git commit, DVC revision và MLflow run ID đầy đủ.

Model `FAIL` vẫn được giữ trong MLflow để điều tra nhưng không được đăng ký alias `candidate`,
`champion` hoặc `production`.

## 9. Version hóa artifact và đăng ký Model Registry

- DVC-add checkpoint, ONNX, external weights, embeddings, metadata và release report của v2.
- Push artifact lên MinIO và xác nhận `dvc status` sạch.
- Đăng ký v2 trong MLflow Model Registry với alias ban đầu `candidate`.
- Liên kết registry version tới Git commit, DVC revision, config, metrics và validation report.
- Chưa đổi alias `champion` tại bước này.

## 10. Deploy staging và kiểm thử deployment

- Tạo Triton model repository v2, không ghi đè model v1.
- Tạo Qdrant collection v2, không xóa collection v1.
- Build API image có tag Git commit và model release.
- Deploy vào GitHub environment `staging`.
- Chạy health, readiness, model-info, API contract, integration và regression test.
- Benchmark latency p50/p95/p99, throughput, error rate, GPU, CPU và RAM.
- Xác nhận restart không mất Qdrant data và checksum sai bị từ chối khởi động.

Deployment metrics không được giảm quá ngưỡng so với offline evaluation.

## 11. Promote production và rollback

Sau khi staging PASS và có manual approval:

1. Chuyển alias v2 từ `candidate` sang `champion`.
2. Deploy đúng API image, Triton release và Qdrant collection v2.
3. Chạy production smoke test.
4. Thử rollback toàn bộ về image/model/collection v1.
5. Xác nhận v1 hoạt động rồi mới chuyển lại v2 nếu cần.

Không xóa artifact, image hoặc collection v1 sau khi promote v2.

## 12. Monitoring và vòng lặp tiếp theo

Sau deploy, API tiếp tục lấy mẫu embedding. Workflow drift định kỳ tạo report mới; Prometheus và
Grafana hiển thị service health, latency, error rate, embedding norm, similarity distribution,
no-match rate và drift metrics.

Khi drift vượt ngưỡng, hệ thống quay lại bước decision gate. Đây mới là vòng MLOps khép kín:

```text
monitor → detect drift → approve → retrain → validate → deploy → monitor
```

## 13. Tiêu chí hoàn thành đồ án

- [ ] Có một lần chạy end-to-end từ drift MSMT17 đến model v2 candidate.
- [ ] Control Center hiển thị toàn bộ cycle và khởi động được workflow bằng một nút.
- [ ] Toàn bộ v1 → drift → approval → retrain → v2 → staging thao tác được trên web, không cần
  terminal.
- [ ] Không có secret trong frontend; mọi thao tác điều khiển đi qua Control API và có audit log.
- [x] Retraining đã được triển khai bằng workflow self-hosted runner sau approval; còn cần chạy cycle
  thật để nghiệm thu GPU/runtime.
- [ ] MLflow truy ngược được model v2 về Git, DVC, config, metrics và artifact.
- [ ] Validation gate ngăn model FAIL đi tiếp.
- [ ] Staging test và benchmark PASS.
- [ ] Production cần approval và rollback v1 đã được thử nghiệm.
- [ ] Grafana hiển thị system metrics, model metrics và drift.
- [ ] README, sơ đồ kiến trúc và báo cáo demo phản ánh đúng hệ thống thực tế.

## Thứ tự thực hiện ngay

```text
1. Xây FastAPI Control API và state machine bền vững cho cycle/audit log
2. Hoàn thiện workflow retraining khép kín trên self-hosted runner
3. Xây Next.js Control Center và nối Control API
4. Đưa drift report và UI approval `RETRAIN_APPROVED` vào workflow
5. Bấm Start trên UI; theo dõi toàn bộ v1 → v2, training tối đa 20 epoch và MLflow
6. Evaluate MSMT17 + Market-1501 và chạy validation gate
7. DVC push + đăng ký MLflow candidate
8. Export ONNX + parity + reference embeddings
9. Deploy staging và chạy integration test từ cycle
10. Phê duyệt production và rollback test hoàn toàn trên UI
11. Monitoring liên tục và hoàn thiện báo cáo/demo
```
