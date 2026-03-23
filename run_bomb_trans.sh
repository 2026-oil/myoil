#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${NF_CASE_CONFIGS:-}" ]]; then
  export NF_CASE_CONFIGS="$(cat <<'EOF'
yaml/bomb_trans/brentoil-case3-trans-i128-tail-base.yaml
yaml/bomb_trans/brentoil-case3-trans-i128-tail-res-delta.yaml
yaml/bomb_trans/brentoil-case3-trans-i128-tail-res-level.yaml
yaml/bomb_trans/brentoil-case3-trans-i48-tail-base.yaml
yaml/bomb_trans/brentoil-case3-trans-i48-tail-res-delta.yaml
yaml/bomb_trans/brentoil-case3-trans-i48-tail-res-level.yaml
yaml/bomb_trans/wti-case3-trans-i128-tail-base.yaml
yaml/bomb_trans/wti-case3-trans-i128-tail-res-delta.yaml
yaml/bomb_trans/wti-case3-trans-i128-tail-res-level.yaml
yaml/bomb_trans/wti-case3-trans-i48-tail-base.yaml
yaml/bomb_trans/wti-case3-trans-i48-tail-res-delta.yaml
yaml/bomb_trans/wti-case3-trans-i48-tail-res-level.yaml
EOF
)"
fi

exec "$script_dir/run.sh" "$@"
