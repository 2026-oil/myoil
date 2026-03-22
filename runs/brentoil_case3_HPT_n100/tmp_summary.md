# tmp_summary

- 기준: `main.py` 실행 중 stdout 에 실제로 찍힌 항목만 사용
- 포함: `__NF_PROGRESS__{...}` progress payload, 마지막 JSON 출력(`{"ok": ...}`)
- 제외: `summary.json`, `cv/*.csv`, `best_params.json`, `fit_summary.json`, `manifest`, `config` 같은 실행 후 artifact 파일
- 생성시각(UTC): 2026-03-22T14:46:00.157122+00:00
- run root: `/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100`

## main.py stdout 에서 실제로 보이는 출력 타입
- progress line 필드: `job_name`, `model_index`, `total_models`, `total_steps`, `completed_steps`, `total_folds`, `current_fold`, `phase`, `status`, `detail`, `event`, `progress_pct`, `progress_text`
- 정상 종료 JSON 필드: 단일 job 실행 기준 `ok`, `executed_jobs`, `worker_results`, `summary_artifacts`

## 현재 stdout 기준 상태 요약
| model | stdout.log | progress_count | first event | last event | final JSON visible |
| --- | --- | ---: | --- | --- | --- |
| PatchTST | True | 954 | model-start / running / 0% | fold-start / running / 94% / tune-trial-96/100 / 476/505 | no |
| DLinear | True | 1012 | model-start / running / 0% | model-done / completed / 100% / replay / 505/505 | yes |
| Naive | False | 0 | - | - | no |
| iTransformer | True | 1010 | model-start / running / 0% | fold-start / running / 100% / replay / 504/505 | no |
| LSTM | True | 36 | model-start / running / 0% | fold-start / running / 3% / tune-trial-4/100 / 17/505 | no |
| NHITS | True | 500 | model-start / running / 0% | fold-start / running / 49% / tune-trial-50/100 / 249/505 | no |

## stdout 만 기준으로 본 한줄 해석
- `DLinear` 만 정상 종료 JSON 출력까지 실제로 남아 있습니다.
- `PatchTST`, `LSTM`, `NHITS` 는 progress line 만 있고 종료 JSON 은 없습니다.
- `iTransformer` 는 progress 가 `replay 504/505, 100%` 까지 보이지만 종료 JSON 은 현재 stdout 에 남아 있지 않습니다.
- `Naive` 는 현재 run root 아래 worker stdout.log 가 없어, 실행 출력만 기준으로는 볼 수 있는 항목이 없습니다.

## PatchTST
- stdout.log: `scheduler/workers/PatchTST/stdout.log`
- progress line 개수: `954`
- 첫 progress payload
```json
{
  "job_name": "PatchTST",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 0,
  "total_folds": 5,
  "current_fold": null,
  "phase": null,
  "status": "running",
  "detail": "mode=learned_auto stage=full output_root=/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100/scheduler/workers/PatchTST",
  "event": "model-start",
  "progress_pct": 0,
  "progress_text": "[------------------] 0/505   0%"
}
```
- 마지막 progress payload
```json
{
  "job_name": "PatchTST",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 476,
  "total_folds": 5,
  "current_fold": 1,
  "phase": "tune-trial-96/100",
  "status": "running",
  "detail": null,
  "event": "fold-start",
  "progress_pct": 94,
  "progress_text": "[#################-] 476/505  94%"
}
```
- 마지막 JSON 출력 없음
## DLinear
- stdout.log: `scheduler/workers/DLinear/stdout.log`
- progress line 개수: `1012`
- 첫 progress payload
```json
{
  "job_name": "DLinear",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 0,
  "total_folds": 5,
  "current_fold": null,
  "phase": null,
  "status": "running",
  "detail": "mode=learned_auto stage=full output_root=/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100/scheduler/workers/DLinear",
  "event": "model-start",
  "progress_pct": 0,
  "progress_text": "[------------------] 0/505   0%"
}
```
- 마지막 progress payload
```json
{
  "job_name": "DLinear",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 505,
  "total_folds": 5,
  "current_fold": null,
  "phase": "replay",
  "status": "completed",
  "detail": "run-complete",
  "event": "model-done",
  "progress_pct": 100,
  "progress_text": "[##################] 505/505 100%"
}
```
- 마지막 JSON 출력
```json
{
  "ok": true,
  "executed_jobs": [
    "DLinear"
  ],
  "worker_results": [],
  "summary_artifacts": {}
}
```
## Naive
- stdout.log 없음
- 따라서 현재 남아 있는 `main.py` 실행 출력 기준으로는 확인 가능한 항목이 없음
## iTransformer
- stdout.log: `scheduler/workers/iTransformer/stdout.log`
- progress line 개수: `1010`
- 첫 progress payload
```json
{
  "job_name": "iTransformer",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 0,
  "total_folds": 5,
  "current_fold": null,
  "phase": null,
  "status": "running",
  "detail": "mode=learned_auto stage=full output_root=/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100/scheduler/workers/iTransformer",
  "event": "model-start",
  "progress_pct": 0,
  "progress_text": "[------------------] 0/505   0%"
}
```
- 마지막 progress payload
```json
{
  "job_name": "iTransformer",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 504,
  "total_folds": 5,
  "current_fold": 4,
  "phase": "replay",
  "status": "running",
  "detail": null,
  "event": "fold-start",
  "progress_pct": 100,
  "progress_text": "[##################] 504/505 100%"
}
```
- 마지막 JSON 출력 없음
## LSTM
- stdout.log: `scheduler/workers/LSTM/stdout.log`
- progress line 개수: `36`
- 첫 progress payload
```json
{
  "job_name": "LSTM",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 0,
  "total_folds": 5,
  "current_fold": null,
  "phase": null,
  "status": "running",
  "detail": "mode=learned_auto stage=full output_root=/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100/scheduler/workers/LSTM",
  "event": "model-start",
  "progress_pct": 0,
  "progress_text": "[------------------] 0/505   0%"
}
```
- 마지막 progress payload
```json
{
  "job_name": "LSTM",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 17,
  "total_folds": 5,
  "current_fold": 2,
  "phase": "tune-trial-4/100",
  "status": "running",
  "detail": null,
  "event": "fold-start",
  "progress_pct": 3,
  "progress_text": "[#-----------------] 17/505   3%"
}
```
- 마지막 JSON 출력 없음
## NHITS
- stdout.log: `scheduler/workers/NHITS/stdout.log`
- progress line 개수: `500`
- 첫 progress payload
```json
{
  "job_name": "NHITS",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 0,
  "total_folds": 5,
  "current_fold": null,
  "phase": null,
  "status": "running",
  "detail": "mode=learned_auto stage=full output_root=/home/sonet/.openclaw/workspace/research/neuralforecast/runs/brentoil_case3_HPT_n100/scheduler/workers/NHITS",
  "event": "model-start",
  "progress_pct": 0,
  "progress_text": "[------------------] 0/505   0%"
}
```
- 마지막 progress payload
```json
{
  "job_name": "NHITS",
  "model_index": 1,
  "total_models": 1,
  "total_steps": 505,
  "completed_steps": 249,
  "total_folds": 5,
  "current_fold": 4,
  "phase": "tune-trial-50/100",
  "status": "running",
  "detail": null,
  "event": "fold-start",
  "progress_pct": 49,
  "progress_text": "[#########---------] 249/505  49%"
}
```
- 마지막 JSON 출력 없음
