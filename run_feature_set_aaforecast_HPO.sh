#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

requested_configs=(
  "yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-patchtst-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-timexer-ret_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-informer_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-gru_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-patchtst_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/aaforecast-timexer_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/baseline_HPO.yaml"
  "yaml/experiment/feature_set_aaforecast/baseline-ret_HPO.yaml"
)

configs=()
for config in "${requested_configs[@]}"; do
  if [[ -f "$config" ]]; then
    configs+=("$config")
  else
    echo "[feature_set_aaforecast_HPO] skip missing: $config" >&2
  fi
done

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "[feature_set_aaforecast_HPO] no runnable *_HPO configs found" >&2
  exit 1
fi

printf -v nf_case_configs '%s\n' "${configs[@]}"
export NF_CASE_CONFIGS="$nf_case_configs"
export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_HPO}"

echo "[feature_set_aaforecast_HPO] configs=${#configs[@]}"
exec bash "$repo_root/run.sh" "$@"


\research\neuralforecast\runs\raw 에 있는 모든 fold 별 결과 그래프 fold 별로 뽑아서 
fold 별로 하나의 plot으로 시각화하고 싶어. 

input 이전길이 와 이후 예측 과 실제값 이런 식으로 나오도록하고
input구간 최근 16주로 짜른 그래프도 하나 나오도록해주고,