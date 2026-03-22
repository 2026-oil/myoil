#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${NF_CASE_CONFIGS:-}" ]]; then
  export NF_CASE_CONFIGS="$(cat <<'EOF'
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i48.yaml
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i48-res-level.yaml
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i48-res-delta.yaml
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i128.yaml
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i128-res-level.yaml
yaml/bomb/brentoil-case3-family-h8-diff-exloss-i128-res-delta.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i48.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i48-res-level.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i48-res-delta.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i128.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i128-res-level.yaml
yaml/bomb/wti-case3-family-h8-diff-exloss-i128-res-delta.yaml
EOF
)"
fi

exec "$script_dir/run.sh" "$@"
