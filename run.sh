#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_root="${NF_CASE_LOG_ROOT:-runs/_batch_logs}"
log_dir="${log_root}/${timestamp}"
mkdir -p "$log_dir"
results_tsv="${log_dir}/results.tsv"
summary_json="${log_dir}/summary.json"
summary_txt="${log_dir}/summary.txt"
python_bin="${NF_CASE_PYTHON_BIN:-python3}"
summary_python_bin="${NF_CASE_SUMMARY_PYTHON_BIN:-python3}"
use_pty_mode="${NF_CASE_USE_PTY:-auto}"
patchtst_job="${NF_CASE_PATCHTST_JOB:-PatchTST}"

_should_use_pty() {
  case "$use_pty_mode" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    auto)
      [[ -t 1 ]] || return 1
      command -v script >/dev/null 2>&1 || return 1
      return 0
      ;;
    *)
      echo "[batch] invalid NF_CASE_USE_PTY=$use_pty_mode (use: auto|0|1)" >&2
      return 1
      ;;
  esac
}

_join_argv_for_script() {
  local out=""
  local part
  for part in "$@"; do
    out+=$(printf ' %q' "$part")
  done
  printf '%s' "${out# }"
}

_validate_patchtst_only_args() {
  local arg
  local expects_value="0"
  for arg in "$@"; do
    if [[ "$expects_value" == "1" ]]; then
      expects_value="0"
      continue
    fi
    case "$arg" in
      --jobs|--output-root)
        echo "[batch] $arg is managed by run.sh PatchTST mode; do not pass it explicitly" >&2
        return 1
        ;;
      --jobs=*|--output-root=*)
        echo "[batch] ${arg%%=*} is managed by run.sh PatchTST mode; do not pass it explicitly" >&2
        return 1
        ;;
    esac
  done
}

yaml_list=(
  "yaml/feature_set_HPT_c3/brentoil-case3.yaml"
  "yaml/feature_set_HPT_c3/wti-case3.yaml"
)

if [[ -n "${NF_CASE_CONFIGS:-}" ]]; then
  mapfile -t _raw_configs <<<"${NF_CASE_CONFIGS}"
  configs=()
  for cfg in "${_raw_configs[@]}"; do
    if [[ -n "${cfg//[[:space:]]/}" ]]; then
      configs+=("$cfg")
    fi
  done
else
  configs=("${yaml_list[@]}")
fi

if ! _validate_patchtst_only_args "$@"; then
  exit 2
fi

passed=()
failed=()
missing=()
pty_enabled="0"
if _should_use_pty; then
  pty_enabled="1"
fi
script_start="$(date -u +%FT%TZ)"
printf 'config\tstatus\texit_code\tlog_path\tstarted_at\tfinished_at\n' >"$results_tsv"

echo "[batch] repo_root=$repo_root"
echo "[batch] timestamp=$timestamp"
echo "[batch] log_dir=$log_dir"
echo "[batch] runner=$python_bin main.py"
echo "[batch] job=$patchtst_job"
echo "[batch] use_pty=$pty_enabled (NF_CASE_USE_PTY=$use_pty_mode)"
echo "[batch] extra_args=${*:-<none>}"
echo

for cfg in "${configs[@]}"; do
  cfg_start="$(date -u +%FT%TZ)"
  if [[ ! -f "$cfg" ]]; then
    echo "[batch] missing config: $cfg"
    missing+=("$cfg")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$cfg" "missing" "missing" "" "$cfg_start" "$(date -u +%FT%TZ)" >>"$results_tsv"
    continue
  fi

  echo "================================================================================"
  echo "[batch] start: $cfg"
  echo "--------------------------------------------------------------------------------"

  cfg_name="$(basename -- "$cfg")"
  log_path="${log_dir}/${cfg_name%.yaml}.log"
  if _should_use_pty; then
    cmd="$(_join_argv_for_script "$python_bin" "main.py" "--config" "$cfg" "--jobs" "$patchtst_job" "$@")"
    if script -q -e -c "$cmd" /dev/null 2>&1 | tee "$log_path"; then
      echo "[batch] ok: $cfg (log: $log_path)"
      passed+=("$cfg")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cfg" "passed" "0" "$log_path" "$cfg_start" "$(date -u +%FT%TZ)" >>"$results_tsv"
    else
      status=$?
      echo "[batch] fail: $cfg (exit=$status, log: $log_path)"
      failed+=("$cfg (exit=$status)")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cfg" "failed" "$status" "$log_path" "$cfg_start" "$(date -u +%FT%TZ)" >>"$results_tsv"
    fi
  elif "$python_bin" main.py --config "$cfg" --jobs "$patchtst_job" "$@" 2>&1 | tee "$log_path"; then
    echo "[batch] ok: $cfg (log: $log_path)"
    passed+=("$cfg")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$cfg" "passed" "0" "$log_path" "$cfg_start" "$(date -u +%FT%TZ)" >>"$results_tsv"
  else
    status=$?
    echo "[batch] fail: $cfg (exit=$status, log: $log_path)"
    failed+=("$cfg (exit=$status)")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$cfg" "failed" "$status" "$log_path" "$cfg_start" "$(date -u +%FT%TZ)" >>"$results_tsv"
  fi
  echo
done

echo "================================================================================"
echo "[batch] summary"
echo "[batch] passed=${#passed[@]} failed=${#failed[@]} missing=${#missing[@]}"
if ((${#passed[@]})); then
  printf '[batch] passed:\n'
  printf '  - %s\n' "${passed[@]}"
fi
if ((${#failed[@]})); then
  printf '[batch] failed:\n'
  printf '  - %s\n' "${failed[@]}"
fi
if ((${#missing[@]})); then
  printf '[batch] missing:\n'
  printf '  - %s\n' "${missing[@]}"
fi

"$summary_python_bin" - "$results_tsv" "$summary_json" "$summary_txt" "$repo_root" "$log_dir" "$script_start" "$(date -u +%FT%TZ)" "${*:-}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_txt = Path(sys.argv[3])
repo_root, log_dir, started_at, finished_at, extra_args = sys.argv[4:]

with results_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

passed = [row["config"] for row in rows if row["status"] == "passed"]
failed = [
    f'{row["config"]} (exit={row["exit_code"]})'
    for row in rows
    if row["status"] == "failed"
]
missing = [row["config"] for row in rows if row["status"] == "missing"]

payload = {
    "repo_root": repo_root,
    "log_dir": log_dir,
    "started_at": started_at,
    "finished_at": finished_at,
    "extra_args": extra_args.split() if extra_args else [],
    "passed_count": len(passed),
    "failed_count": len(failed),
    "missing_count": len(missing),
    "passed": passed,
    "failed": failed,
    "missing": missing,
    "results": rows,
}
summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "[batch] summary",
    f"[batch] repo_root={repo_root}",
    f"[batch] log_dir={log_dir}",
    f"[batch] started_at={started_at}",
    f"[batch] finished_at={finished_at}",
    f"[batch] passed={len(passed)} failed={len(failed)} missing={len(missing)}",
]
if passed:
    lines.append("[batch] passed:")
    lines.extend(f"  - {item}" for item in passed)
if failed:
    lines.append("[batch] failed:")
    lines.extend(f"  - {item}" for item in failed)
if missing:
    lines.append("[batch] missing:")
    lines.extend(f"  - {item}" for item in missing)
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[batch] summary_json=$summary_json"
echo "[batch] summary_txt=$summary_txt"

if ((${#failed[@]})) || ((${#missing[@]})); then
  exit_code=1
else
  exit_code=0
fi

exit "$exit_code"
