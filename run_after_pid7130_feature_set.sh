#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wait_pid="${NF_WAIT_CHAIN_WAIT_PID:-7130}"
poll_seconds="${NF_WAIT_CHAIN_POLL_SECONDS:-5}"
feature_set_dir="${NF_WAIT_CHAIN_FEATURE_SET_DIR:-yaml/feature_set}"
hpt_c3_dir="${NF_WAIT_CHAIN_HPT_C3_DIR:-yaml/feature_set_HPT_c3}"

echo "[wait-chain] waiting for PID ${wait_pid} to exit naturally"
while ps -p "$wait_pid" >/dev/null 2>&1; do
  sleep "$poll_seconds"
done
echo "[wait-chain] PID ${wait_pid} is gone; starting queued runs"

cd "$repo_root"

uv run main.py --config "${hpt_c3_dir}/wti-case3.yaml" "$@"
uv run main.py --config "${hpt_c3_dir}/brentoil-case3.yaml" --jobs TSMixerx "$@"

mapfile -t feature_configs < <(find "${feature_set_dir}" -maxdepth 1 -name '*.yaml' | sort)
for cfg in "${feature_configs[@]}"; do
  uv run main.py --config "$cfg" "$@"
done
