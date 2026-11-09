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
Batch không bị trộn với query/gallery: runner tạo một dataset view tạm, dùng batch làm train split và
giữ Duke `query`/`bounding_box_test` cố định để đánh giá công bằng.

## Cấu hình runner

`.env.mlops` và GitHub secret `MLOPS_ENV_FILE` phải có đường dẫn host mà self-hosted runner đọc được:

```ini
CONTROL_HOST_DATASET_ROOT=/home/monster/conbobietbay/Person-Re-Identification/datasets
CONTROL_INBOX_MIN_VALID_IMAGES=1000
CONTROL_INBOX_MIN_IDENTITIES=50
CONTROL_INBOX_MIN_CAMERAS=2
```

Không commit dữ liệu hoặc `.env.mlops`. Sau khi chép batch vào `datasets/inbox`, mở **Data Inbox**,
chọn batch và bấm **Import selected batch**. Batch hợp lệ tự chạy drift/evaluate-v1 rồi dừng tại
`WAITING_APPROVAL`; retraining chỉ bắt đầu sau xác nhận của người vận hành.
