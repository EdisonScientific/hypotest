#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_opensandbox_stress_test.sh TRACE_JSONL CAPSULE_MAP_JSON [RUN_ID]

Runs the exact GBS1024 OpenSandbox qualification profile and always reconciles
sandboxes bearing its exact run tag before exiting. Secrets and private image
references must be supplied through the environment; this script never sources
or prints an env file.

Required environment:
  OPEN_SANDBOX_DOMAIN
  OPEN_SANDBOX_API_KEY
  GITLAB_IMAGE or OPEN_SANDBOX_IMAGE
  REGISTRY_USERNAME
  REGISTRY_PASSWORD

Optional environment:
  OPEN_SANDBOX_PROTOCOL  Connection protocol understood by the SDK
  MOUNTED_CAPSULE_ROOT   Default: cluster Edison capsule mount
  OUTPUT_DIR             Default: artifacts/qualification
  PYTHON_BIN             Default: .venv/bin/python
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
trace_path="$1"
capsule_map_path="$2"
run_id="${3:-stress-c1024-$(date -u +%Y%m%dT%H%M%SZ)}"
python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
output_dir="${OUTPUT_DIR:-${project_root}/artifacts/qualification}"
mounted_root="${MOUNTED_CAPSULE_ROOT:-/mnt/s3-data/data/bbh/capsules/edison-20260725/}"
output_path="${output_dir}/${run_id}.json"
log_path="${output_dir}/${run_id}.log"
audit_script="${project_root}/scripts/audit_opensandbox_sandboxes.py"
replay_script="${project_root}/scripts/replay_opensandbox_trace.py"

for required_path in "${trace_path}" "${capsule_map_path}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done
if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment not found at ${python_bin}; initialize the project environment first." >&2
    exit 2
fi
if [[ -z "${OPEN_SANDBOX_DOMAIN:-}" || -z "${OPEN_SANDBOX_API_KEY:-}" ]]; then
    echo "OPEN_SANDBOX_DOMAIN and OPEN_SANDBOX_API_KEY are required for a remote-cluster run." >&2
    exit 2
fi
if [[ -z "${GITLAB_IMAGE:-${OPEN_SANDBOX_IMAGE:-}}" ]]; then
    echo "GITLAB_IMAGE or OPEN_SANDBOX_IMAGE is required." >&2
    exit 2
fi
if [[ -z "${REGISTRY_USERNAME:-}" || -z "${REGISTRY_PASSWORD:-}" ]]; then
    echo "REGISTRY_USERNAME and REGISTRY_PASSWORD are required for the private target image." >&2
    exit 2
fi

mkdir -p "${output_dir}"

# A 1024-worker driver requires at least 4,608 descriptors. The replay driver
# validates the effective hard/soft limits as well, so failure here is harmless.
ulimit -n 8192 2>/dev/null || true

json_count() {
    local field="$1"
    "${python_bin}" -c '
import functools
import json
import sys

value = functools.reduce(
    lambda item, key: item.get(key, 0) if isinstance(item, dict) else 0,
    sys.argv[1].split("."),
    json.load(sys.stdin),
)
print(value if isinstance(value, int) else 0)
' "${field}"
}

echo "Auditing active cluster state before ${run_id}..."
active_audit="$("${python_bin}" "${audit_script}" \
    --since-minutes 720 \
    --state Pending \
    --state Running \
    --state Paused)"
printf '%s\n' "${active_audit}"
active_gbs_count="$(printf '%s' "${active_audit}" | json_count 'purposes.gbs1024-trace-replay')"
if [[ "${active_gbs_count}" -ne 0 ]]; then
    echo "Refusing to queue another GBS1024 replay while ${active_gbs_count} tagged sandbox(es) are active." >&2
    echo "Coordinate with the existing run owner or clean its exact run tag before retrying." >&2
    exit 3
fi

cleanup_exact_run() {
    local cleanup_status=0
    local remaining=1

    set +e
    echo "Reconciling exact-tag sandboxes for ${run_id}..."
    "${python_bin}" "${audit_script}" \
        --since-minutes 720 \
        --query-run-id "${run_id}" \
        --kill-run-id "${run_id}"

    for _attempt in {1..12}; do
        cleanup_audit="$("${python_bin}" "${audit_script}" \
            --since-minutes 720 \
            --query-run-id "${run_id}" 2>/dev/null)"
        remaining="$(printf '%s' "${cleanup_audit}" | json_count 'recent_matching_image')"
        if [[ "${remaining}" -eq 0 ]]; then
            printf '%s\n' "${cleanup_audit}"
            cleanup_status=0
            break
        fi
        cleanup_status=1
        sleep 5
    done
    if [[ "${remaining}" -ne 0 ]]; then
        echo "Cleanup did not converge: ${remaining} exact-tag sandbox(es) remain." >&2
    fi
    set -e
    return "${cleanup_status}"
}

cleanup_on_exit() {
    local replay_status=$?
    local cleanup_status=0
    trap - EXIT INT TERM
    cleanup_exact_run || cleanup_status=$?
    if [[ "${replay_status}" -eq 0 && "${cleanup_status}" -ne 0 ]]; then
        exit "${cleanup_status}"
    fi
    exit "${replay_status}"
}

trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting exact GBS1024 replay ${run_id}."
echo "Summary: ${output_path}"
echo "Log: ${log_path}"

set +e
"${python_bin}" "${replay_script}" "${trace_path}" \
    --trace-step 40 \
    --mounted-root "${mounted_root}" \
    --output "${output_path}" \
    --capsule-map "${capsule_map_path}" \
    --concurrency 1024 \
    --preflight-actions 3 \
    --cpu-request 0.25 \
    --memory-request-mb 512 \
    --cpu-limit 4 \
    --memory-limit-mb 65536 \
    --kernel-memory-limit-mb 57344 \
    --ephemeral-storage-gib 50 \
    --cell-timeout-seconds 900 \
    --job-timeout-seconds 10800 \
    --ready-timeout-seconds 900 \
    --lifecycle-create-concurrency 64 \
    --kernel-request-concurrency 128 \
    --create-attempts 3 \
    --ttl-seconds 14400 \
    --progress-seconds 30 \
    --max-wall-seconds 3600 \
    --run-id "${run_id}" \
    2>&1 | tee "${log_path}"
replay_status="${PIPESTATUS[0]}"
set -e

exit "${replay_status}"
