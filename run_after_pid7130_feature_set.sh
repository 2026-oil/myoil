#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

wait_pid="${NF_WAIT_CHAIN_WAIT_PID:-7130}"
poll_seconds="${NF_WAIT_CHAIN_POLL_SECONDS:-30}"
feature_set_dir="${NF_WAIT_CHAIN_FEATURE_SET_DIR:-$repo_root/yaml/feature_set}"
hpt_c3_dir="${NF_WAIT_CHAIN_HPT_C3_DIR:-$repo_root/yaml/feature_set_HPT_c3}"
main_py="${NF_WAIT_CHAIN_MAIN_PY:-main.py}"
runner_bin="${NF_WAIT_CHAIN_RUNNER_BIN:-uv}"

brent_hpt_cfg="$hpt_c3_dir/brentoil-case3.yaml"
wti_hpt_cfg="$hpt_c3_dir/wti-case3.yaml"

declare -a feature_set_configs=()
shopt -s nullglob
for cfg in "$feature_set_dir"/*.yaml; do
  feature_set_configs+=("$cfg")
done
shopt -u nullglob

if ((${#feature_set_configs[@]} == 0)); then
  echo "[wait-chain] no feature-set configs found under $feature_set_dir" >&2
  exit 1
fi

for cfg in "$brent_hpt_cfg" "$wti_hpt_cfg"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[wait-chain] missing required config: $cfg" >&2
    exit 1
  fi
done

run_main() {
  echo "+ $runner_bin run $main_py $*"
  "$runner_bin" run "$main_py" "$@"
}

echo "[wait-chain] repo_root=$repo_root"
echo "[wait-chain] waiting for PID $wait_pid to exit naturally"
while ps -p "$wait_pid" >/dev/null 2>&1; do
  echo "[wait-chain] PID $wait_pid still running; sleeping ${poll_seconds}s"
  sleep "$poll_seconds"
done
echo "[wait-chain] PID $wait_pid is gone; starting queued runs"

run_main --config "$wti_hpt_cfg" "$@"
run_main --config "$brent_hpt_cfg" --jobs TSMixerx "$@"

for cfg in "${feature_set_configs[@]}"; do
  run_main --config "$cfg" "$@"
done
