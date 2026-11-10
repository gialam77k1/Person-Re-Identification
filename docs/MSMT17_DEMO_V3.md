# Batch MSMT17 2.000 ảnh để demo vòng v3

Batch local `datasets/inbox/msmt17-demo-v3-2000-6cam` được tạo từ train split của
`datasets/MSMT17_V1`. Dữ liệu ảnh không được commit; script và quy tắc chọn được version hóa.

## Quy tắc chọn

- Chọn đúng sáu camera `1, 5, 7, 11, 13, 14`. Đây là tổ hợp có coverage tốt nhất từ lần phân tích
  toàn bộ tổ hợp camera trước đó: query coverage `0,7295`, multi-camera train identity coverage
  `0,9760` và bao phủ `1.040/1.041` train identity.
- Lấy đúng `2.000` ảnh theo quota cân bằng: `334, 334, 333, 333, 333, 333`.
- Sampling xác định với seed `20261110`; trong từng camera không lấy lặp và loại trùng nội dung bằng
  SHA256.
- Ảnh được hardlink nếu cùng filesystem để không tốn thêm dung lượng; fallback sang copy nếu cần.
- `manifest.json` ghi person ID, camera ID, checksum, ảnh nguồn và toàn bộ provenance.

Tạo lại batch trên máy mới:

```bash
conda run --no-capture-output -n reid \
  python scripts/create_msmt17_demo_batch.py
```

Trường `source_dataset` trong manifest là `MSMT17_V1`. Trường vận hành `dataset` hiện giữ giá trị
`dukemtmc-reid` vì pipeline đang dùng Duke query/gallery làm protocol đánh giá cố định. Điều này giúp
batch MSMT17 mô phỏng dữ liệu online mới để train candidate kế tiếp, nhưng kết quả v3 vẫn được đo trên
cùng tập test Duke, không dùng 2.000 ảnh train để tự chấm điểm.

## Cách demo v2 → v3

Chỉ dùng batch sau khi v2 đã được promote thành alias MLflow `champion`:

1. Mở `http://localhost:3001/inbox`.
2. Chọn `msmt17-demo-v3-2000-6cam`.
3. Bấm **Import selected batch**.
4. Control API khóa champion v2 và tự tạo target v3.
5. Chờ drift/evaluation kết thúc; xem metric và lý do trên trang Pipeline.
6. Chỉ bấm **Approve Retraining** khi evidence phù hợp với mục tiêu demo.

Nếu v2 chưa là `champion`, không import batch này: hệ thống sẽ lấy champion hiện tại làm source và tên
target có thể không phải v3.
