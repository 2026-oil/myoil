#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

yes_configs=(
  "yaml/experiment/feature_set_aaforecast_YES_BS/baseline.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/baseline-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-informer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-gru-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-patchtst-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-timexer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-informer.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-gru.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-patchtst.yaml"
  "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-timexer.yaml"
)

no_configs=(
  "yaml/experiment/feature_set_aaforecast_NO_BS/baseline.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/baseline-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-informer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-gru-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-patchtst-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-timexer-ret.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-informer.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-gru.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-patchtst.yaml"
  "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-timexer.yaml"
)

post_python="$repo_root/.venv/bin/python"
if [[ ! -x "$post_python" ]]; then
  post_python="${NF_CASE_PYTHON_BIN:-python3}"
fi

case_log_root_base="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_bs_compare_matrix}"
feature_set_raw_root_base="${NF_FEATURE_SET_RAW_ROOT:-runs/raw_feature_set_aaforecast_bs_compare}"

export NF_CASE_TIMESTAMP="${NF_CASE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export NF_FEATURE_SET_GRAPH_X_START="${NF_FEATURE_SET_GRAPH_X_START:-2025-08-15}"
export NF_FEATURE_SET_GRAPH_X_END="${NF_FEATURE_SET_GRAPH_X_END:-2026-03-09}"

run_one() {
  local label="$1"
  local -n configs_ref="$2"
  shift 2

  printf -v nf_case_configs '%s\n' "${configs_ref[@]}"
  export NF_CASE_CONFIGS="$nf_case_configs"
  export NF_CASE_LOG_ROOT="${case_log_root_base}/${label}"
  export NF_FEATURE_SET_RAW_ROOT="${feature_set_raw_root_base}/${label}"

  local batch_id="$NF_CASE_TIMESTAMP"
  local summary_json="$repo_root/${NF_CASE_LOG_ROOT}/${batch_id}/summary.json"
  local raw_batch_root="$repo_root/${NF_FEATURE_SET_RAW_ROOT}/${batch_id}"

  echo "[feature_set_aaforecast_bs_compare_matrix] label=$label"
  echo "[feature_set_aaforecast_bs_compare_matrix] configs=${#configs_ref[@]}"
  echo "[feature_set_aaforecast_bs_compare_matrix] batch_id=$batch_id"
  echo "[feature_set_aaforecast_bs_compare_matrix] raw_batch_root=$raw_batch_root"

  local run_status=0
  bash "$repo_root/run.sh" "$@" || run_status=$?

  local post_status=0
  if [[ -f "$summary_json" ]]; then
    "$post_python" "$repo_root/scripts/feature_set_aaforecast_postprocess.py" \
      --summary-json "$summary_json" \
      --raw-batch-root "$raw_batch_root" \
      --x-start "$NF_FEATURE_SET_GRAPH_X_START" \
      --x-end "$NF_FEATURE_SET_GRAPH_X_END" || post_status=$?
  else
    echo "[feature_set_aaforecast_bs_compare_matrix] missing summary_json=$summary_json" >&2
    post_status=1
  fi

  if ((run_status != 0)); then
    return "$run_status"
  fi
  return "$post_status"
}

yes_status=0
no_status=0

run_one "YES_BS" yes_configs "$@" || yes_status=$?
run_one "NO_BS" no_configs "$@" || no_status=$?

echo "[feature_set_aaforecast_bs_compare_matrix] status YES_BS=$yes_status NO_BS=$no_status"
if ((yes_status != 0)); then
  exit "$yes_status"
fi
exit "$no_status"

