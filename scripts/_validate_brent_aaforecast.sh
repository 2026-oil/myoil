#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/.validate_brent_aaforecast.log"
: >"$LOG"
for c in \
  yaml/experiment/feature_set_aaforecast_brent/baseline.yaml \
  yaml/experiment/feature_set_aaforecast_brent/baseline-ret.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer-ret.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-patchtst-ret.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer-ret.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-informer.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-patchtst.yaml \
  yaml/experiment/feature_set_aaforecast_brent/aaforecast-timexer.yaml
do
  echo "=== $c ===" | tee -a "$LOG"
  uv run python main.py --validate-only --config "$c" 2>&1 | tee -a "$LOG"
done
echo "ALL_OK" | tee -a "$LOG"
