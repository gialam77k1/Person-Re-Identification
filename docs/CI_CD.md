# CI/CD cho model và inference API

Tài liệu này ghi lại Giai đoạn 8: kiểm tra pull request, phát hành model đã qua validation gate, triển khai staging/production và rollback.

## Những thay đổi đã thực hiện

- Thêm `.github/workflows/ci.yml` để lint, chạy unit/API test, validate config/manifest/DVC metadata, quét model lớn/secret, build và smoke-test API image không cần GPU.
- Thêm `.github/workflows/release.yml` theo luồng `validation gate → build/push image → staging → smoke test → production`.
- Thêm `.github/workflows/rollback.yml` để phục hồi đồng thời API image và model release trước đó.
- Thêm `src/release_deployment.py` để xác minh report/manifest/checksum, kiểm tra immutable image tag, deploy, smoke test và quản lý lịch sử release.
- Thêm `scripts/ci.sh` và `scripts/deploy_local.sh` để chạy cùng logic trên máy local/self-hosted runner.
- Cho phép `docker-compose.mlops.yml` nhận image qua biến `API_IMAGE`.
- FastAPI chỉ nạp model khi ứng dụng startup, nên CI có thể import OpenAPI và smoke-test container mà không cần GPU hoặc artifact DVC.

## Pull request CI

Chạy đầy đủ kiểm tra local:

```bash
conda activate reid
pip install -r requirements-ci.txt
./scripts/ci.sh
```

Hai job GitHub Actions là:

- `Lint, tests, and repository validation`.
- `Build and smoke-test API image without GPU`.

Bộ quét repository từ chối file model/data nhị phân được track, file lớn hơn 5 MiB, private key và các mẫu token phổ biến. Artifact lớn tiếp tục được quản lý bằng DVC.

Để pull request lỗi không thể merge, vào GitHub:

1. `Settings → Branches` hoặc `Settings → Rules → Rulesets`.
2. Bảo vệ nhánh `lam` và nhánh production (`main` nếu dùng).
3. Bật `Require status checks to pass before merging`.
4. Chọn cả hai required checks nêu trên.

Workflow tạo check, còn branch protection là thiết lập cấp repository nên không thể ép buộc chỉ bằng file YAML.

## Chuẩn bị self-hosted runner

Job validation và deployment cần máy Linux local có:

- Docker Engine, Docker Compose và NVIDIA Container Toolkit.
- `DOCKER_CONTEXT=default` trong `.env.mlops` nếu máy đồng thời cài Docker Desktop; context được chọn
  phải có NVIDIA runtime (`docker --context default info`).
- Conda environment `reid` đã cài `requirements.txt`.
- Kết nối được MinIO/DVC remote, Triton, Qdrant và cổng API.
- Quyền đọc GitHub Container Registry.

Tạo GitHub Actions secret `MLOPS_ENV_FILE` chứa nội dung `.env.mlops`. Không đưa file này hoặc credential vào Git.

Sau khi runner hiện trạng thái `Idle`, mở `Actions → Self-hosted runner smoke test → Run workflow`.
Giữ `start_stack=false` và `verify_services=true`. Workflow chỉ kiểm tra stack chính đã được dựng từ
workspace bền vững; không dựng container hoặc bind mount từ checkout tạm của Actions. Chỉ bật
`start_stack` khi chủ động chẩn đoán một checkout có thể bị xóa. Workflow sẽ kiểm tra:

- Git, Docker/Compose, Conda environment `reid`, DVC và NVIDIA driver.
- `.env.mlops` được tạo tạm từ secret và không còn placeholder `change-me`.
- Model repository hợp lệ trước khi bật stack.
- PostgreSQL, MinIO, MLflow, Qdrant, Triton và inference API đều healthy.

Có thể chạy cùng kiểm tra trực tiếp trên máy runner:

```bash
./scripts/self_hosted_runner_smoke.sh prerequisites
./scripts/mlops_stack.sh up
./scripts/self_hosted_runner_smoke.sh services
```

Workflow chỉ cho phép chạy thủ công để code từ pull request không tự động chạy trên máy local.
File `.env.mlops` tạm được xóa ở bước cuối kể cả khi smoke test thất bại.
Workflow cố định `DOCKER_CONTEXT=default` và `DOCKER_HOST=unix:///var/run/docker.sock` để không vô tình
kết nối Docker Desktop. Runner service cũng phải có cùng `DOCKER_HOST`.
Mount node-exporter dùng chế độ read-only tiêu chuẩn để chạy được trên cả Docker Engine native và
Docker Desktop; không yêu cầu root mount propagation kiểu `rslave`.

### Chạy runner như systemd service

Trên máy runner, dừng tiến trình `./run.sh` chạy thủ công rồi chạy script cài đặt. Script tạo systemd
drop-in cố định Docker Engine native, bật tự khởi động và khởi chạy service:

```bash
./scripts/install_self_hosted_runner_service.sh ~/actions-runner "$USER"
```

Lệnh sẽ yêu cầu `sudo` một lần. Kiểm tra lại bằng
`systemctl status "$(cat ~/actions-runner/.service)"`; trạng thái đúng là `active (running)` và `enabled`.

Tạo hai GitHub Environments:

- `staging`.
- `production`, bật `Required reviewers` để tạo bước duyệt thủ công trước production.

Nếu chưa cấu hình required reviewer, workflow vẫn tham chiếu Environment `production` nhưng GitHub sẽ không dừng để xin duyệt.

## Phát hành model

Mở `Actions → Release model service → Run workflow`, nhập:

```text
release_id:   market1501-vit-bnneck:v1-dvc
release_slug: market1501-vit-bnneck-v1-dvc
```

Workflow thực hiện:

1. Pull artifact từ DVC trên self-hosted runner.
2. Chỉ chấp nhận validation report `PASS`, manifest `validated`, release ID khớp và toàn bộ checksum artifact hợp lệ.
3. Build/push API image lên GHCR bằng tag bất biến dạng `<git-sha-7>-model-<version>`.
4. Deploy staging bằng Triton model repository của đúng release.
5. Chờ `/ready` và xác nhận `/model-info.release_id` khớp.
6. Chờ approval của Environment `production` rồi mới deploy production.

Ví dụ image tag:

```text
ghcr.io/gialam77k1/reid-api:1ca60d8-model-v1
```

Model không qua gate dừng ngay ở job `validate-candidate`; image không được build và staging không bị thay đổi.

## Rollback

Mỗi deployment thành công lưu `current.json` và `previous.json`. Mặc định state nằm ngoài Git checkout tại:

```text
../person-reid-deployments/<environment>/
```

Có thể đổi bằng `DEPLOYMENT_STATE_ROOT` trong `.env.mlops`. Không đặt thư mục này bên trong repository vì `actions/checkout` có thể dọn file ignored.

Chạy rollback tại `Actions → Roll back model service`, chọn `staging` hoặc `production`. Workflow sẽ:

1. Đọc release rollback từ state.
2. Pull lại đúng artifact bằng DVC.
3. Pull đúng immutable image.
4. Khởi động lại Triton và API.
5. Chỉ ghi nhận rollback thành công sau khi `/ready` và `/model-info` đạt yêu cầu.

Nếu một deployment đang ở trạng thái `pending` rồi smoke test thất bại, rollback chọn release `current` gần nhất. Nếu deployment mới đã thành công, rollback chọn `previous`. Production rollback cũng đi qua Environment `production` và approval tương ứng.

## Kiểm tra nhanh candidate trên local

```bash
conda run -n reid python src/release_deployment.py verify \
  --release-id market1501-vit-bnneck:v1-dvc \
  --validation-report artifacts/pipeline/market1501-vit-bnneck-v1-dvc/release/validation_report.json \
  --manifest artifacts/pipeline/market1501-vit-bnneck-v1-dvc/release/model_manifest.json
```

Lệnh chỉ đọc và xác minh candidate, không deploy.
