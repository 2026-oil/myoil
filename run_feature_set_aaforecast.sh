#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

configs=(
  "yaml/experiment/feature_set_aaforecast_brent/baseline.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/baseline-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru.yaml"
  "yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer.yaml"
)

printf -v nf_case_configs '%s\n' "${configs[@]}"
export NF_CASE_CONFIGS="$nf_case_configs"
export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_wti_matrix}"
export NF_CASE_TIMESTAMP="${NF_CASE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export NF_FEATURE_SET_RAW_ROOT="${NF_FEATURE_SET_RAW_ROOT:-runs/raw_feature_set_aaforecast_wti}"
export NF_FEATURE_SET_GRAPH_X_START="${NF_FEATURE_SET_GRAPH_X_START:-2025-08-15}"
export NF_FEATURE_SET_GRAPH_X_END="${NF_FEATURE_SET_GRAPH_X_END:-2026-03-09}"

batch_id="$NF_CASE_TIMESTAMP"
summary_json="$repo_root/${NF_CASE_LOG_ROOT}/${batch_id}/summary.json"
raw_batch_root="$repo_root/${NF_FEATURE_SET_RAW_ROOT}/${batch_id}"

post_python="$repo_root/.venv/bin/python"
if [[ ! -x "$post_python" ]]; then
  post_python="${NF_CASE_PYTHON_BIN:-python3}"
fi

echo "[feature_set_aaforecast_wti_matrix] configs=${#configs[@]}"
echo "[feature_set_aaforecast_wti_matrix] batch_id=$batch_id"
echo "[feature_set_aaforecast_wti_matrix] raw_batch_root=$raw_batch_root"

run_status=0
bash "$repo_root/run.sh" "$@" || run_status=$?

post_status=0
if [[ -f "$summary_json" ]]; then
  "$post_python" "$repo_root/scripts/feature_set_aaforecast_postprocess.py" \
    --summary-json "$summary_json" \
    --raw-batch-root "$raw_batch_root" \
    --x-start "$NF_FEATURE_SET_GRAPH_X_START" \
    --x-end "$NF_FEATURE_SET_GRAPH_X_END" || post_status=$?
else
  echo "[feature_set_aaforecast_wti_matrix] missing summary_json=$summary_json" >&2
  post_status=1
fi

if ((run_status != 0)); then
  exit "$run_status"
fi
exit "$post_status"
