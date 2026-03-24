#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

log_root="${NF_WTI_STOP_LOG_ROOT:-$repo_root/runs/_batch_logs}"
log_dir_override="${NF_WTI_STOP_LOG_DIR:-}"
target_config="${NF_WTI_STOP_TARGET_CONFIG:-yaml/feature_set_HPT_n100_bs/wti-case3.yaml}"
poll_seconds="${NF_WTI_STOP_POLL_SECONDS:-15}"
stop_signal="${NF_WTI_STOP_SIGNAL:-INT}"
grace_seconds="${NF_WTI_STOP_GRACE_SECONDS:-20}"
run_pid_override="${NF_WTI_STOP_RUN_PID:-}"

resolve_log_dir() {
  if [[ -n "$log_dir_override" ]]; then
    printf '%s\n' "$log_dir_override"
    return 0
  fi

  local latest=""
  if [[ -d "$log_root" ]]; then
    latest="$(find "$log_root" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
  fi
  if [[ -z "$latest" ]]; then
    echo "[watch-stop] no batch log directory found under $log_root" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

infer_run_pid() {
  if [[ -n "$run_pid_override" ]]; then
    printf '%s\n' "$run_pid_override"
    return 0
  fi

  local pid=""
  local cwd=""
  local matches=()
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$repo_root" ]] || continue
    matches+=("$pid")
  done < <(
    ps -eo pid=,args= | awk '
      $0 ~ /(^|[[:space:]])(bash[[:space:]]+)?(\.\/)?run\.sh([[:space:]]|$)/ {
        print $1
      }
    '
  )

  if ((${#matches[@]} == 0)); then
    echo "[watch-stop] no active run.sh process found in $repo_root" >&2
    return 1
  fi

  local newest_index=$((${#matches[@]} - 1))
  if ((${#matches[@]} > 1)); then
    echo "[watch-stop] multiple run.sh PIDs found (${matches[*]}); using newest ${matches[$newest_index]}" >&2
  fi
  printf '%s\n' "${matches[$newest_index]}"
}

target_status_from_results() {
  local results_path="$1"
  [[ -f "$results_path" ]] || return 0
  awk -F'\t' -v config="$target_config" '
    NR > 1 && $1 == config && ($2 == "passed" || $2 == "failed" || $2 == "missing") {
      print $2
      exit
    }
  ' "$results_path"
}

collect_descendant_pids() {
  local root_pid="$1"
  local frontier=("$root_pid")
  local descendants=()
  local current=""
  local child=""

  while ((${#frontier[@]} > 0)); do
    current="${frontier[0]}"
    frontier=("${frontier[@]:1}")
    while read -r child; do
      [[ -n "$child" ]] || continue
      descendants+=("$child")
      frontier+=("$child")
    done < <(ps -eo pid=,ppid= | awk -v target="$current" '$2 == target {print $1}')
  done

  if ((${#descendants[@]})); then
    printf '%s\n' "${descendants[@]}"
  fi
}

signal_pid_list() {
  local signal_name="$1"
  shift || true
  local pid=""
  local live=()

  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" >/dev/null 2>&1; then
      live+=("$pid")
    fi
  done

  for pid in "${live[@]}"; do
    kill "-$signal_name" "$pid" 2>/dev/null || true
  done

  if ((${#live[@]})); then
    printf '%s\n' "${live[*]}"
  fi
}

pid_is_live() {
  local pid="$1"
  local stat=""

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi

  stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR == 1 {print $1}')"
  [[ -n "$stat" ]] || return 1
  [[ "$stat" == Z* ]] && return 1
  return 0
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout="$2"
  local deadline=0

  deadline="$(python3 - "$timeout" <<'PY'
import sys
import time
print(time.time() + float(sys.argv[1]))
PY
)"

  while pid_is_live "$pid"; do
    if ! python3 - "$deadline" <<'PY'
import sys
import time
sys.exit(0 if time.time() < float(sys.argv[1]) else 1)
PY
    then
      return 1
    fi
    sleep "$poll_seconds"
  done
}

log_dir="$(resolve_log_dir)"
results_tsv="${log_dir}/results.tsv"
run_pid="$(infer_run_pid)"

echo "[watch-stop] repo_root=$repo_root"
echo "[watch-stop] log_dir=$log_dir"
echo "[watch-stop] results_tsv=$results_tsv"
echo "[watch-stop] target_config=$target_config"
echo "[watch-stop] run_pid=$run_pid"
echo "[watch-stop] poll_seconds=$poll_seconds"
echo "[watch-stop] stop_signal=$stop_signal grace_seconds=$grace_seconds"

status=""
while true; do
  status="$(target_status_from_results "$results_tsv")"
  if [[ -n "$status" ]]; then
    break
  fi

  if ! pid_is_live "$run_pid"; then
    echo "[watch-stop] run.sh PID $run_pid exited before $target_config completed" >&2
    exit 1
  fi

  sleep "$poll_seconds"
done

echo "[watch-stop] detected $target_config completion with status=$status"

mapfile -t descendants < <(collect_descendant_pids "$run_pid")
signal_targets=("$run_pid" "${descendants[@]}")
signaled="$(signal_pid_list "$stop_signal" "${signal_targets[@]}")"
if [[ -n "$signaled" ]]; then
  echo "[watch-stop] sent SIG$stop_signal to: $signaled"
else
  echo "[watch-stop] nothing left to signal; run may have already exited"
  exit 0
fi

if wait_for_pid_exit "$run_pid" "$grace_seconds"; then
  echo "[watch-stop] run.sh PID $run_pid exited after SIG$stop_signal"
  exit 0
fi

echo "[watch-stop] run.sh PID $run_pid still alive after ${grace_seconds}s; escalating to SIGTERM"
mapfile -t remaining_descendants < <(collect_descendant_pids "$run_pid")
signal_pid_list "TERM" "$run_pid" "${remaining_descendants[@]}" >/dev/null || true

if wait_for_pid_exit "$run_pid" "$grace_seconds"; then
  echo "[watch-stop] run.sh PID $run_pid exited after SIGTERM"
  exit 0
fi

echo "[watch-stop] run.sh PID $run_pid did not exit after SIGTERM" >&2
exit 1
