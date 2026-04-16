#!/usr/bin/env bash
# Run the full retrieval parameter grid sweep.
#
# Usage:
#   bash run_sweep_ret.sh                    # generate configs + run all
#   bash run_sweep_ret.sh --generate-only    # generate configs without running
#   bash run_sweep_ret.sh --skip-generate    # run pre-generated configs only
#
# Extra args after the flags are forwarded to run.sh (e.g. --validate-only).
set -uo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${NF_CASE_PYTHON_BIN:-python3}"
configs_txt="yaml/experiment/sweep_retrieval/configs.txt"

generate_only=0
skip_generate=0
extra_args=()

for arg in "$@"; do
  case "$arg" in
    --generate-only) generate_only=1 ;;
    --skip-generate) skip_generate=1 ;;
    *) extra_args+=("$arg") ;;
  esac
done

# --- Step 1: Generate sweep YAML configs ---
if ((skip_generate == 0)); then
  echo "[sweep] Generating retrieval grid configs..."
  "$python_bin" scripts/sweep_retrieval_grid.py
  if [[ $? -ne 0 ]]; then
    echo "[sweep] Config generation failed" >&2
    exit 1
  fi
fi

if [[ ! -f "$configs_txt" ]]; then
  echo "[sweep] $configs_txt not found. Run without --skip-generate first." >&2
  exit 1
fi

if ((generate_only)); then
  echo "[sweep] --generate-only: skipping execution"
  exit 0
fi

# --- Step 2: Feed configs into run.sh ---
# Use NF_CASE_CONFIGS_FILE so the env does not exceed kernel limits (8748+ paths).
config_count="$(wc -l < "${repo_root}/${configs_txt}")"
echo "[sweep] Launching ${config_count} configs via run.sh (NF_CASE_CONFIGS_FILE)..."

export NF_CASE_CONFIGS_FILE="${repo_root}/${configs_txt}"
unset NF_CASE_CONFIGS 2>/dev/null || true

exec bash "$repo_root/run.sh" "${extra_args[@]+"${extra_args[@]}"}"
