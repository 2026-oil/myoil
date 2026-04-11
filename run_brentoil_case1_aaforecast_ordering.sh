#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

export NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/brentoil_case1_aaforecast_ordering}"
export NF_CASE_CONFIGS="${NF_CASE_CONFIGS:-$(cat <<'EOF'
yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer.yaml
yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru.yaml
yaml/experiment/feature_set_aaforecast/brentoil-case1-baseline.yaml
EOF
)}"

exec bash "$repo_root/run.sh" "$@"
