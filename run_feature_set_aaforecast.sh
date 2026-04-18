#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

batch_id="${NF_CASE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
graph_x_start="${NF_FEATURE_SET_GRAPH_X_START:-2025-08-15}"
graph_x_end="${NF_FEATURE_SET_GRAPH_X_END:-2026-03-09}"
post_python="$repo_root/.venv/bin/python"
if [[ ! -x "$post_python" ]]; then
  post_python="${NF_CASE_PYTHON_BIN:-python3}"
fi

case_selector="all"

case "$case_selector" in
  all)
    target_cases=(wti brent dubai)
    ;;
  dubai|wti|brent)
    target_cases=("$case_selector")
    ;;
  *)
    echo "[run_feature_set_aaforecast] unsupported TARGET_CASE=$case_selector (expected: all, dubai, wti, brent)" >&2
    exit 1
    ;;
esac

run_case() {
  local target_case="$1"
  shift
  local config_dir="yaml/experiment/feature_set_aaforecast_${target_case}"
  local run_label="feature_set_aaforecast_${target_case}_matrix"
  local case_log_root_default="runs/_batch_logs/${run_label}"
  local case_raw_root_default="runs/raw_feature_set_aaforecast_${target_case}"
  local case_log_root="${NF_CASE_LOG_ROOT:-$case_log_root_default}"
  local case_raw_root="${NF_FEATURE_SET_RAW_ROOT:-$case_raw_root_default}"

  if ((${#target_cases[@]} > 1)); then
    if [[ -n "${NF_CASE_LOG_ROOT:-}" ]]; then
      case_log_root="${NF_CASE_LOG_ROOT}/${target_case}"
    fi
    if [[ -n "${NF_FEATURE_SET_RAW_ROOT:-}" ]]; then
      case_raw_root="${NF_FEATURE_SET_RAW_ROOT}/${target_case}"
    fi
  fi

  local configs=(
    "${config_dir}/baseline-ret.yaml"
    "${config_dir}/aaforecast-informer-ret.yaml"
    "${config_dir}/aaforecast-gru-ret.yaml"
    "${config_dir}/aaforecast-timexer-ret.yaml"
  )
  local nf_case_configs
  local summary_json="$repo_root/${case_log_root}/${batch_id}/summary.json"
  local raw_batch_root="$repo_root/${case_raw_root}/${batch_id}"
  local run_status=0
  local post_status=0

  printf -v nf_case_configs '%s\n' "${configs[@]}"
  export TARGET_CASE="$target_case"
  export NF_CASE_CONFIGS="$nf_case_configs"
  export NF_CASE_LOG_ROOT="$case_log_root"
  export NF_CASE_TIMESTAMP="$batch_id"
  export NF_FEATURE_SET_RAW_ROOT="$case_raw_root"
  export NF_FEATURE_SET_GRAPH_X_START="$graph_x_start"
  export NF_FEATURE_SET_GRAPH_X_END="$graph_x_end"

  echo "[$run_label] target_case=$target_case"
  echo "[$run_label] configs=${#configs[@]}"
  echo "[$run_label] batch_id=$batch_id"
  echo "[$run_label] raw_batch_root=$raw_batch_root"

  bash "$repo_root/run.sh" "$@" || run_status=$?

  if [[ -f "$summary_json" ]]; then
    "$post_python" "$repo_root/scripts/feature_set_aaforecast_postprocess.py" \
      --summary-json "$summary_json" \
      --raw-batch-root "$raw_batch_root" \
      --x-start "$NF_FEATURE_SET_GRAPH_X_START" \
      --x-end "$NF_FEATURE_SET_GRAPH_X_END" || post_status=$?
  else
    echo "[$run_label] missing summary_json=$summary_json" >&2
    post_status=1
  fi

  if ((run_status != 0)); then
    return "$run_status"
  fi
  return "$post_status"
}

overall_status=0
for target_case in "${target_cases[@]}"; do
  if run_case "$target_case" "$@"; then
    continue
  else
    case_status=$?
    if ((overall_status == 0)); then
      overall_status=$case_status
    fi
  fi
done

exit "$overall_status"
