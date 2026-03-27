#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CONFIGS="$(cat <<'EOF'
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

export NF_CASE_CONFIGS="${NF_CASE_CONFIGS:-$DEFAULT_CONFIGS}"
export NF_CASE_GLOBAL_GPU_QUEUE="${NF_CASE_GLOBAL_GPU_QUEUE:-0}"

exec bash "$repo_root/run.sh" "$@"
