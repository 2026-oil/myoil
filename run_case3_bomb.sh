#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${NF_CASE_CONFIGS:-}" ]]; then
  export NF_CASE_CONFIGS="$(cat <<'EOF'
yaml/bomb/brentoil-case3-family-h8-mse.yaml
yaml/bomb/brentoil-case3-family-h8-mse-diff.yaml
yaml/bomb/brentoil-case3-family-h8-exloss.yaml
yaml/bomb/brentoil-case3-family-h8-mse-res-level.yaml
yaml/bomb/brentoil-case3-family-h8-mse-res-delta.yaml
yaml/bomb/brentoil-case3-family-h6-mse.yaml
yaml/bomb/brentoil-case3-family-h4-mse.yaml
yaml/bomb/wti-case3-family-h8-mse.yaml
yaml/bomb/wti-case3-family-h8-mse-diff.yaml
yaml/bomb/wti-case3-family-h8-exloss.yaml
yaml/bomb/wti-case3-family-h8-mse-res-level.yaml
yaml/bomb/wti-case3-family-h8-mse-res-delta.yaml
yaml/bomb/wti-case3-family-h6-mse.yaml
yaml/bomb/wti-case3-family-h4-mse.yaml
EOF
)"
fi

exec "$script_dir/run.sh" "$@"
