#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

configs=(
  "yaml/experiment/feature_set_aaforecast/baseline.yaml"
  "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml"
)

printf -v nf_case_configs '%s\n' "${configs[@]}"
export NF_CASE_CONFIGS="$nf_case_configs"
export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_matrix}"

echo "[feature_set_aaforecast_matrix] configs=${#configs[@]}"
exec bash "$repo_root/run.sh" "$@"
