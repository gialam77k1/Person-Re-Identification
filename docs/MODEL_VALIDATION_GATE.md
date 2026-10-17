# Model validation gate

## Mục tiêu

Gate chặn model không đạt chất lượng trước khi đóng gói Triton hoặc gán alias trong MLflow Model Registry. Mọi candidate đều có báo cáo `PASS`/`FAIL`; candidate `FAIL` vẫn được lưu trong MLflow để điều tra nhưng không được promote.

Luồng kiểm soát:

```text
train/evaluate/export/parity/reference
                  |
                  v
          validate_release
             |         |
           PASS       FAIL
             |         +-> log MLflow, dừng pipeline
             v
       package_triton -> publish/register -> champion hoặc production
```

## Chính sách hiện tại

Ngưỡng được version hóa tại `configs/model_validation.yaml`:

| Kiểm tra | Ngưỡng |
|---|---:|
| Rank-1 | `>= 0,91` |
| mAP | `>= 0,80` |
| mINP | `>= 0,70` |
| ONNX cosine similarity | `>= 0,9999` |
| ONNX max absolute error | `<= 0,0001` |
| Embedding dimension | `512` |
| Rank-1 giảm so với deployment baseline | `<= 0,01` |
| mAP giảm so với deployment baseline | `<= 0,01` |
| mINP giảm so với deployment baseline | `<= 0,02` |

mINP tối thiểu `0,70` được bổ sung vì đây là metric bắt buộc nhưng đề bài chưa cho ngưỡng cụ thể. Deployment regression dùng metric không re-ranking của candidate để so với model champion đang chạy, vì re-ranking không nằm trong serving graph.

## Kiểm tra bắt buộc

`src/model_validation_gate.py` kiểm tra:

- Dataset validation đã PASS.
- Rank-1, mAP và mINP đạt ngưỡng và là số hữu hạn.
- Parity report PASS, cosine similarity và sai số tuyệt đối đạt ngưỡng.
- Input `images: float32[1,3,224,224]` và output `embeddings: float32[1,512]` đúng contract.
- Reference embedding có shape đúng, không chứa `NaN`/`Inf`; PID, camera ID và path có cùng số phần tử.
- Checkpoint, ONNX graph, external weights, preprocessing contract và reference bundle tồn tại, khác rỗng và có SHA256 đầy đủ.
- Candidate không giảm quá mức cho phép so với deployment baseline.

Nếu input bị thiếu hoặc malformed, gate vẫn ghi report `FAIL` với check `gate_execution` thay vì chỉ dừng bằng stack trace.

## Chống bypass

- DVC khai báo `package_triton` phụ thuộc vào report của `validate_release`.
- `src/prepare_triton_model.py` bắt buộc nhận `--validation-report`, kiểm tra trạng thái `PASS` và đối chiếu SHA256 của đúng ONNX graph/external weights trước khi copy.
- `src/publish_model_release.py` bắt buộc nhận `--validation-report`. Model version FAIL vẫn có thể được ghi Registry để truy vết nhưng không được gán alias.
- Report phải có cùng `release_id` với manifest; report của release khác không thể dùng để promote.

## Chạy gate

Cách chuẩn là chạy pipeline:

```bash
./scripts/dvc_local.sh repro
```

Report và manifest:

```text
artifacts/pipeline/<release-slug>/release/validation_report.json
artifacts/pipeline/<release-slug>/release/model_manifest.json
```

Đóng gói thủ công vẫn phải truyền report:

```bash
conda run -n reid python src/prepare_triton_model.py \
  --onnx-path artifacts/pipeline/<release-slug>/onnx/exports/model_embedding.onnx \
  --output-root artifacts/pipeline/<release-slug>/triton/model_repository \
  --validation-report artifacts/pipeline/<release-slug>/release/validation_report.json \
  --model-name reid_embedding \
  --model-version 1 \
  --instance-kind KIND_GPU
```

Publish lên MLflow:

```bash
conda run -n reid python src/publish_model_release.py \
  --manifest model_releases/<model>/<version>/model_manifest.json \
  --validation-report artifacts/pipeline/<release-slug>/release/validation_report.json \
  --run-metadata artifacts/pipeline/<release-slug>/train/logs/mlflow_run.json
```

Các tag/metric MLflow chính:

```text
validation.status = PASS | FAIL
validation.release_id = <release-id>
validation_passed = 1 | 0
validation_failed_checks = <count>
```

Report JSON được lưu ở artifact path `validation/` của run.

## Kiểm thử

```bash
conda run -n reid python -m unittest tests.test_model_validation_gate -v
```

Test bao phủ candidate PASS, quality FAIL, embedding chứa NaN, packaging bị chặn và promotion chỉ nhận report PASS đúng release.

## Kết quả nghiệm thu ngày 2026-10-17

- Đo production champion qua Triton HTTP trên 3.368 query và 15.913 gallery hợp lệ: Rank-1 `0,91360`, mAP `0,80099`, mINP `0,51208`.
- Candidate đạt Rank-1 `0,92072`, mAP `0,88768`, mINP `0,74940`; metric serving-equivalent giảm Rank-1 `0,00178`, còn mAP và mINP tốt hơn baseline.
- Gate trả `PASS`, không có failed check; đủ 18 check và 8 nhóm artifact có SHA256.
- DVC chỉ chạy `package_triton` sau khi `validate_release` PASS; Triton manifest ghi lại `release_id` và đường dẫn report.
- MLflow run `724f910da5684749bcef271f7f7e8d18` có tag `validation.status=PASS`, metric `validation_passed=1`, `validation_failed_checks=0` và artifact `validation/validation_report.json`.
- Chạy lại `dvc repro` bỏ qua toàn bộ stage và báo pipeline up-to-date.
