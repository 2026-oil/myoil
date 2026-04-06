#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CONFIGS="$(cat <<'EOF'
yaml/experiment/feature_set_aaforecast/brentoil-case1-best-all10.yaml
yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_bs_core.yaml
yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_gprd.yaml
yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_all10.yaml
yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_no_bs_core.yaml
yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_no_gprd.yaml
EOF
)"

export NF_CASE_CONFIGS="${NF_CASE_CONFIGS:-$DEFAULT_CONFIGS}"
export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/brentoil_case1_feature_subset_compare}"

exec bash "$repo_root/run.sh" "$@"
