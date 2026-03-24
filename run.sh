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
global_gpu_queue_mode="${NF_CASE_GLOBAL_GPU_QUEUE:-1}"
gpu_ids_override="${NF_CASE_GPU_IDS:-}"
max_parallel_override="${NF_CASE_MAX_PARALLEL:-}"

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

_validate_run_sh_args() {
  local arg
  local expects_value="0"
  for arg in "$@"; do
    if [[ "$expects_value" == "1" ]]; then
      expects_value="0"
      continue
    fi
    case "$arg" in
      --output-root)
        echo "[batch] $arg is managed by run.sh; do not pass it explicitly" >&2
        return 1
        ;;
      --output-root=*)
        echo "[batch] ${arg%%=*} is managed by run.sh; do not pass it explicitly" >&2
        return 1
        ;;
    esac
  done
}

_parse_truthy_flag() {
  case "${1:-}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *)
      echo "[batch] invalid boolean value: ${1:-<empty>}" >&2
      return 2
      ;;
  esac
}

_resolve_batch_gpu_ids() {
  local raw="${gpu_ids_override}"
  if [[ -n "${raw//[[:space:],]/}" ]]; then
    tr ',[:space:]' '\n\n' <<<"$raw" | sed '/^$/d'
    return 0
  fi
  if ! _parse_truthy_flag "$global_gpu_queue_mode" >/dev/null 2>&1; then
    return 0
  fi
  if ! _parse_truthy_flag "$global_gpu_queue_mode"; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d'
}

_config_worker_devices() {
  local cfg_path="$1"
  local parsed=""
  if [[ -f "$cfg_path" ]]; then
    parsed="$(awk '
      /^[[:space:]]*worker_devices:[[:space:]]*[0-9]+([[:space:]]*#.*)?$/ {
        value = $0
        sub(/^[^:]*:[[:space:]]*/, "", value)
        sub(/[[:space:]#].*$/, "", value)
        print value
        exit
      }
    ' "$cfg_path" 2>/dev/null)"
  fi
  if [[ "$parsed" =~ ^[0-9]+$ ]] && ((parsed > 0)); then
    printf '%s\n' "$parsed"
  else
    printf '1\n'
  fi
}

yaml_list=(  
  "yaml/feature_set_HPT_n100_bs/brentoil-case3.yaml"
  "yaml/feature_set_HPT_n100_bs/wti-case3.yaml"
  "yaml/feature_set_HPT_n100_bs/brentoil-case4.yaml"
  "yaml/feature_set_HPT_n100_bs/wti-case4.yaml"
  
  "yaml/feature_set_HPT_n100_bs/brentoil-case2.yaml"
  "yaml/feature_set_HPT_n100_bs/wti-case2.yaml"
  "yaml/feature_set_HPT_n100_bs/brentoil-case1.yaml"
  "yaml/feature_set_HPT_n100_bs/wti-case1.yaml"
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

if ! _validate_run_sh_args "$@"; then
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
echo "[batch] use_pty=$pty_enabled (NF_CASE_USE_PTY=$use_pty_mode)"
echo "[batch] extra_args=${*:-<none>}"
echo

_parse_truthy_flag "$global_gpu_queue_mode" >/dev/null 2>&1
parse_status=$?
if ((parse_status == 2)); then
  exit 2
fi

mapfile -t batch_gpu_ids < <(_resolve_batch_gpu_ids)
batch_parallel_slots=1
if _parse_truthy_flag "$global_gpu_queue_mode" && ((${#batch_gpu_ids[@]} > 1)); then
  batch_parallel_slots=${#batch_gpu_ids[@]}
fi
if [[ -n "${max_parallel_override}" ]]; then
  if [[ ! "${max_parallel_override}" =~ ^[0-9]+$ ]] || ((max_parallel_override < 1)); then
    echo "[batch] invalid NF_CASE_MAX_PARALLEL=${max_parallel_override} (use positive integer)" >&2
    exit 2
  fi
  if ((batch_parallel_slots > max_parallel_override)); then
    batch_parallel_slots=${max_parallel_override}
  fi
fi

if ((${#batch_gpu_ids[@]})); then
  echo "[batch] gpu_ids=${batch_gpu_ids[*]}"
else
  echo "[batch] gpu_ids=<none>"
fi
echo "[batch] global_gpu_queue=$global_gpu_queue_mode slots=$batch_parallel_slots"
echo

if ((batch_parallel_slots > 1)); then
  config_meta_file="$(mktemp)"
  extra_args_file="$(mktemp)"
  cleanup_temp_files() {
    rm -f "$config_meta_file" "$extra_args_file"
  }
  trap cleanup_temp_files EXIT
  : >"$config_meta_file"
  for cfg in "${configs[@]}"; do
    printf '%s\t%s\n' "$cfg" "$(_config_worker_devices "$cfg")" >>"$config_meta_file"
  done
  printf '%s\n' "$@" >"$extra_args_file"
  gpu_ids_csv=""
  if ((${#batch_gpu_ids[@]})); then
    gpu_ids_csv="$(IFS=,; echo "${batch_gpu_ids[*]}")"
  fi

  if "$summary_python_bin" - "$config_meta_file" "$extra_args_file" "$results_tsv" "$log_dir" "$python_bin" "$repo_root" "$log_root" "$script_start" "$gpu_ids_csv" "$batch_parallel_slots" <<'PY'
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_iso8601(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_duration_history(log_root: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not log_root.exists():
        return durations
    summary_paths = sorted(log_root.glob("*/summary.json"), reverse=True)
    for summary_path in summary_paths:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in payload.get("results", []):
            config = row.get("config")
            if not config or config in durations:
                continue
            if row.get("status") not in {"passed", "failed"}:
                continue
            started_at = parse_iso8601(row.get("started_at", ""))
            finished_at = parse_iso8601(row.get("finished_at", ""))
            if started_at is None or finished_at is None:
                continue
            durations[config] = max((finished_at - started_at).total_seconds(), 0.0)
    return durations


config_meta_path = Path(sys.argv[1])
extra_args_path = Path(sys.argv[2])
results_tsv_path = Path(sys.argv[3])
log_dir = Path(sys.argv[4])
python_bin = sys.argv[5]
repo_root = Path(sys.argv[6])
log_root = Path(sys.argv[7])
gpu_ids_csv = sys.argv[9]
slot_count = max(1, int(sys.argv[10]))


def load_config_meta(path: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for raw_line in load_lines(path):
        config, _, worker_devices = raw_line.partition("\t")
        try:
            parsed_worker_devices = max(1, int(worker_devices.strip() or "1"))
        except ValueError:
            parsed_worker_devices = 1
        items.append(
            {
                "config": config,
                "worker_devices": parsed_worker_devices,
            }
        )
    return items


configs = load_config_meta(config_meta_path)
extra_args = load_lines(extra_args_path)
gpu_ids = [int(part) for part in gpu_ids_csv.split(",") if part]
duration_history = load_duration_history(log_root)
gpu_order = {gpu_id: index for index, gpu_id in enumerate(gpu_ids)}

ordered_configs = [
    entry
    for _, entry in sorted(
        enumerate(configs),
        key=lambda item: (-duration_history.get(str(item[1]["config"]), 0.0), item[0]),
    )
]

results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
with results_tsv_path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    active: list[dict[str, object]] = []
    queue = list(ordered_configs)
    available_gpu_ids = list(gpu_ids)

    def slot_label_for(assigned_gpu_ids: tuple[int, ...]) -> str:
        if not assigned_gpu_ids:
            return "cpu"
        return "gpu" + ",".join(str(gpu_id) for gpu_id in assigned_gpu_ids)

    def launch(entry: dict[str, object]) -> bool:
        config = str(entry["config"])
        worker_devices = int(entry["worker_devices"])
        cfg_path = Path(config)
        cfg_start = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        log_path = log_dir / f"{cfg_path.stem}.log"
        if not cfg_path.is_file():
            print(f"[batch] missing config: {config}", flush=True)
            writer.writerow([config, "missing", "missing", "", cfg_start, datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")])
            return False

        estimate = duration_history.get(config, 0.0)
        assigned_gpu_ids: tuple[int, ...] = ()
        env = os.environ.copy()
        env.pop("NEURALFORECAST_WORKER_DEVICES", None)
        if gpu_ids:
            if worker_devices > len(gpu_ids):
                message = (
                    f"requested worker_devices={worker_devices} exceeds available batch gpu_ids={gpu_ids}"
                )
                log_path.write_text(message + "\n", encoding="utf-8")
                print(f"[batch] fail[config]: {config} ({message}, log: {log_path})", flush=True)
                writer.writerow(
                    [
                        config,
                        "failed",
                        "worker_devices",
                        str(log_path),
                        cfg_start,
                        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ]
                )
                return False
            assigned_gpu_ids = tuple(available_gpu_ids[:worker_devices])
            del available_gpu_ids[:worker_devices]
            assigned_gpu_ids_csv = ",".join(str(gpu_id) for gpu_id in assigned_gpu_ids)
            env["CUDA_VISIBLE_DEVICES"] = assigned_gpu_ids_csv
            env["NEURALFORECAST_ASSIGNED_GPU_IDS"] = assigned_gpu_ids_csv
        else:
            env.pop("NEURALFORECAST_ASSIGNED_GPU_IDS", None)
        slot_label = slot_label_for(assigned_gpu_ids)
        cmd = [python_bin, "main.py", "--config", config, *extra_args]
        print("=" * 80, flush=True)
        print(
            f"[batch] start[{slot_label}] worker_devices={worker_devices} estimate_s={estimate:.1f}: {config}",
            flush=True,
        )
        print("-" * 80, flush=True)
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append(
            {
                "config": config,
                "assigned_gpu_ids": assigned_gpu_ids,
                "slot_label": slot_label,
                "started_at": cfg_start,
                "log_path": str(log_path),
                "process": process,
                "log_handle": log_handle,
            }
        )
        return True

    while queue or active:
        while queue and len(active) < slot_count:
            launchable_index = None
            for index, entry in enumerate(queue):
                required_devices = int(entry["worker_devices"])
                if not gpu_ids or required_devices > len(gpu_ids) or len(available_gpu_ids) >= required_devices:
                    launchable_index = index
                    break
            if launchable_index is None:
                break
            if not launch(queue.pop(launchable_index)):
                continue

        if not active:
            continue

        time.sleep(0.2)
        next_active: list[dict[str, object]] = []
        for item in active:
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                next_active.append(item)
                continue
            process.wait()
            item["log_handle"].close()
            finished_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            config = str(item["config"])
            log_path = str(item["log_path"])
            slot_label = str(item["slot_label"])
            if int(returncode) == 0:
                print(f"[batch] ok[{slot_label}]: {config} (log: {log_path})", flush=True)
                status = "passed"
            else:
                print(
                    f"[batch] fail[{slot_label}]: {config} (exit={returncode}, log: {log_path})",
                    flush=True,
                )
                status = "failed"
            writer.writerow(
                [config, status, str(returncode), log_path, item["started_at"], finished_at]
            )
            available_gpu_ids.extend(int(gpu_id) for gpu_id in item["assigned_gpu_ids"])
            available_gpu_ids.sort(key=lambda gpu_id: gpu_order.get(gpu_id, gpu_id))
        active = next_active
        handle.flush()
PY
  then
    :
  else
    status=$?
  fi
  cleanup_temp_files
  trap - EXIT
else

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
    cmd="$(_join_argv_for_script "$python_bin" "main.py" "--config" "$cfg" "$@")"
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
  elif "$python_bin" main.py --config "$cfg" "$@" 2>&1 | tee "$log_path"; then
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

fi

mapfile -t passed < <(awk -F'\t' 'NR>1 && $2=="passed" {print $1}' "$results_tsv")
mapfile -t failed < <(awk -F'\t' 'NR>1 && $2=="failed" {print $1 " (exit=" $3 ")"}' "$results_tsv")
mapfile -t missing < <(awk -F'\t' 'NR>1 && $2=="missing" {print $1}' "$results_tsv")

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
