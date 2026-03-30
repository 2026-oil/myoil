#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

configs=(
  "yaml/experiment/feature_set/brentoil-case3.yaml"
  "yaml/experiment/feature_set/wti-case3.yaml"
  "yaml/experiment/feature_set/brentoil-case4.yaml"
  "yaml/experiment/feature_set/wti-case4.yaml"
  "yaml/experiment/feature_set/brentoil-case2.yaml"
  "yaml/experiment/feature_set/wti-case2.yaml"
  "yaml/experiment/feature_set/brentoil-case1.yaml"
  "yaml/experiment/feature_set/wti-case1.yaml"

  "yaml/experiment/feature_set_bs/brentoil-case3.yaml"
  "yaml/experiment/feature_set_bs/wti-case3.yaml"
  "yaml/experiment/feature_set_bs/brentoil-case4.yaml"
  "yaml/experiment/feature_set_bs/wti-case4.yaml"
  "yaml/experiment/feature_set_bs/brentoil-case2.yaml"
  "yaml/experiment/feature_set_bs/wti-case2.yaml"
  "yaml/experiment/feature_set_bs/brentoil-case1.yaml"
  "yaml/experiment/feature_set_bs/wti-case1.yaml"

  "yaml/experiment/feature_set_bs_futr_only_Int/brentoil-case3.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/wti-case3.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/brentoil-case4.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/wti-case4.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/brentoil-case2.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/wti-case2.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/brentoil-case1.yaml"
  "yaml/experiment/feature_set_bs_futr_only_Int/wti-case1.yaml"
)

if [[ -n "${NF_ALL_LIMIT:-}" ]]; then
  if [[ ! "${NF_ALL_LIMIT}" =~ ^[0-9]+$ ]] || ((NF_ALL_LIMIT < 1)); then
    echo "[all] invalid NF_ALL_LIMIT=${NF_ALL_LIMIT} (use positive integer)" >&2
    exit 2
  fi
  configs=("${configs[@]:0:NF_ALL_LIMIT}")
fi

printf -v nf_case_configs '%s\n' "${configs[@]}"
export NF_CASE_CONFIGS="$nf_case_configs"

echo "[all] configs=${#configs[@]}"
exec bash "$repo_root/run.sh" "$@"
