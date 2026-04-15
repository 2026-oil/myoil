#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

configs=(
  "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-patchtst-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-timexer-ret_HPO.yaml"
)

printf -v nf_case_configs '%s\n' "${configs[@]}"
export NF_CASE_CONFIGS="$nf_case_configs"
export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_HPO}"

echo "[feature_set_aaforecast_HPO] configs=${#configs[@]}"
exec bash "$repo_root/run.sh" "$@"
