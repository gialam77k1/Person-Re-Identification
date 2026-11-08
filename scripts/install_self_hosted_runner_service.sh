#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="${1:-${HOME}/actions-runner}"
RUNNER_USER="${2:-${SUDO_USER:-${USER}}}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo "$0" "${RUNNER_DIR}" "${RUNNER_USER}"
fi

[[ -x "${RUNNER_DIR}/svc.sh" ]] || {
  echo "Không tìm thấy ${RUNNER_DIR}/svc.sh" >&2
  exit 1
}
[[ -f "${RUNNER_DIR}/.runner" ]] || {
  echo "Runner tại ${RUNNER_DIR} chưa được đăng ký với GitHub." >&2
  exit 1
}

cd "${RUNNER_DIR}"
if [[ ! -f .service ]]; then
  ./svc.sh install "${RUNNER_USER}"
fi

service_name="$(<.service)"
[[ "${service_name}" == actions.runner.*.service ]] || {
  echo "Tên service runner không hợp lệ: ${service_name}" >&2
  exit 1
}

dropin_dir="/etc/systemd/system/${service_name}.d"
install -d -m 0755 "${dropin_dir}"
dropin_file="$(mktemp)"
trap 'rm -f "${dropin_file}"' EXIT
printf '%s\n' \
  '[Unit]' \
  'StartLimitIntervalSec=0' \
  '' \
  '[Service]' \
  'Environment=DOCKER_CONTEXT=default' \
  'Environment=DOCKER_HOST=unix:///var/run/docker.sock' \
  'Restart=always' \
  'RestartSec=10s' >"${dropin_file}"
install -m 0644 "${dropin_file}" "${dropin_dir}/10-docker-native.conf"

systemctl daemon-reload
systemctl enable --now "${service_name}"
systemctl --no-pager --full status "${service_name}"

echo "Self-hosted runner service: READY (${service_name})"
