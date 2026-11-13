# Data Inbox và batch dữ liệu mới

Data Inbox là đầu vào duy nhất tạo lifecycle tự động. Giao diện Pipeline không còn tạo cycle thủ công,
nhờ đó một batch không thể vô tình dispatch hai workflow song song.

## Cấu trúc batch

Mỗi batch mới được đặt dưới `datasets/inbox/<batch_id>/`:

```text
datasets/inbox/duke-2026-11-09-001/
├── manifest.json
└── images/
    ├── camera-1-person-10.jpg
    └── camera-2-person-10.jpg
```

Tên ảnh có thể tùy ý vì nhãn nằm trong manifest:

```json
{
  "batch_id": "duke-2026-11-09-001",
  "dataset": "dukemtmc-reid",
  "samples": [
    {"image": "images/camera-1-person-10.jpg", "person_id": 10, "camera_id": 1},
    {"image": "images/camera-2-person-10.jpg", "person_id": 10, "camera_id": 2}
  ]
}
```

`image` phải là đường dẫn tương đối nằm trong batch. `person_id` và `camera_id` phải là số nguyên;
`person_id` âm bị loại và `camera_id` bắt đầu từ 1. Duke train cũ vẫn được nhận ở chế độ tương thích
qua tên file `0001_c2_...jpg`, nhưng batch mới nên luôn có manifest.

## Validation và chống trùng

Mặc định batch cần ít nhất 1.000 ảnh hợp lệ, 50 identity và 2 camera. Control API kiểm tra magic bytes
JPEG/PNG, metadata, đường dẫn an toàn và nội dung trùng. SHA256 tổng hợp bao gồm đường dẫn, nhãn,
kích thước và checksum từng ảnh. Cùng một nội dung/manifest luôn tạo cùng fingerprint và chỉ dispatch
một cycle.

Nếu cùng dataset đang có cycle `PENDING`, `RUNNING` hoặc `WAITING_APPROVAL`, import mới trả HTTP 409.
Runner tạo một dataset view tạm và tự dành riêng held-out query/gallery từ chính batch mới.
Việc tách diễn ra theo person ID, vì vậy một identity không bao giờ xuất hiện đồng thời ở train và
evaluation.

## Cấu hình runner

`.env.mlops` và GitHub secret `MLOPS_ENV_FILE` phải có đường dẫn host mà self-hosted runner đọc được:

```ini
CONTROL_HOST_DATASET_ROOT=/home/monster/conbobietbay/Person-Re-Identification/datasets
CONTROL_INBOX_MIN_VALID_IMAGES=1000
CONTROL_INBOX_MIN_IDENTITIES=50
CONTROL_INBOX_MIN_CAMERAS=2
```

Every accepted batch is split deterministically before training. The control
plane reserves exactly 400 images from identities observed by at least two
cameras: 100 query images and 300 gallery images. Those identities are excluded
completely from training; one image per identity is used as query and gallery
images come from different cameras.
This prevents person-identity leakage while keeping evaluation mandatory for
promotion.

Không commit dữ liệu hoặc `.env.mlops`. Sau khi chép batch vào `datasets/inbox`, mở **Data Inbox**,
chọn batch và bấm **Import selected batch**. Batch hợp lệ tự chạy drift/evaluate source champion rồi
dừng tại `WAITING_APPROVAL`; retraining chỉ bắt đầu sau xác nhận của người vận hành.

## Version lặp vô hạn

Data Inbox không gắn cứng `v1 → v2`. Trước khi tạo cycle, Control API đọc alias `champion` từ
MLflow Model Registry, khóa release đó vào cycle rồi sinh target theo công thức `vN → v(N+1)`.
Nếu MLflow tạm thời không truy cập được, release đã promote gần nhất trong audit database được dùng;
bootstrap v1 chỉ dùng khi chưa có lần promote nào.

```text
champion v1 + batch A → candidate v2
promote v2 thành champion
champion v2 + batch B → candidate v3
promote v3 thành champion
champion v3 + batch C → candidate v4
```

Candidate chưa promote không được dùng làm base. Hệ thống cũng chặn tạo lại cùng target version để
tránh ghi đè checkpoint/artifact. Runner lấy checkpoint champion theo release ID từ MLflow, tạo
runtime config và đường dẫn DVC/MinIO riêng cho target version. Tên stage nội bộ `train_v2` và
`evaluate_v2` được giữ để tương thích database cũ; UI luôn hiển thị source/target thực tế của cycle.

Nếu một cycle đã `FAIL`, target release vẫn bị khóa để người vận hành không vô tình che mất lỗi bằng
một run mới. Trên Pipeline, bấm **Abandon failed attempt** để chuyển cycle sang `CANCELLED`; log,
evidence và audit cũ vẫn được giữ. Sau đó import lại batch đang `BLOCKED`. Data Inbox tái sử dụng đúng
batch fingerprint và tạo cycle mới, thay vì nhân đôi dữ liệu hoặc sửa database thủ công.
