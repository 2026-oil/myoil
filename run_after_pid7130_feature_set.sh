#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

wait_pid="${NF_WAIT_CHAIN_WAIT_PID:-7130}"
poll_seconds="${NF_WAIT_CHAIN_POLL_SECONDS:-5}"
feature_set_dir="${NF_WAIT_CHAIN_FEATURE_SET_DIR:-yaml/feature_set}"
hpt_c3_dir="${NF_WAIT_CHAIN_HPT_C3_DIR:-yaml/feature_set_HPT_c3}"

wti_cfg="${hpt_c3_dir}/wti-case3.yaml"
brent_cfg="${hpt_c3_dir}/brentoil-case3.yaml"

configs=(
  "$wti_cfg"
  "$brent_cfg"
)

if [[ -d "$feature_set_dir" ]]; then
  while IFS= read -r cfg; do
    configs+=("$cfg")
  done < <(find "$feature_set_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
fi

missing=()
for cfg in "${configs[@]}"; do
  [[ -f "$cfg" ]] || missing+=("$cfg")
done

if ((${#missing[@]})); then
  printf '[wait-chain] missing config: %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "[wait-chain] waiting for PID ${wait_pid} to exit naturally"
while ps -p "$wait_pid" >/dev/null 2>&1; do
  sleep "$poll_seconds"
done
echo "[wait-chain] PID ${wait_pid} is gone; starting queued runs"

run_one() {
  local cfg="$1"
  shift || true
  local cmd=(uv run main.py --config "$cfg")
  if (($#)); then
    cmd+=("$@")
  fi
  printf '[wait-chain] run: %q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

run_one "$wti_cfg" "$@"
run_one "$brent_cfg" --jobs TSMixerx "$@"

for ((idx=2; idx<${#configs[@]}; idx++)); do
  run_one "${configs[idx]}" "$@"
done
