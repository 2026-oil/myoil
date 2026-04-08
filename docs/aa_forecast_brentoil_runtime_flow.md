# AA-Forecast BrentOil runtime flow

## Scope
This document explains the **actual happy-path runtime orchestration** when running:

- entrypoint: `main.py`
- config: `yaml/experiment/feature_set_aaforecast/aa_forecast_brentoil.yaml`

It focuses on **runtime orchestration, file/function boundaries, data handoffs, study/fold flow, and produced artifacts**.

## Out of scope
- AAForecast internal math / layer architecture / algorithm derivation
- validate-only-only branch details as a primary execution path
- single-job parallel-tuning worker internals
- error/exception branches

## Verified runtime facts for this exact config

| Item | Verified value |
|---|---|
| task.name | `aa_forecast_brentoil` |
| dataset path | `data/df.csv` |
| target | `Com_BrentCrudeOil` |
| inferred frequency | `W-MON` |
| total rows | `584` |
| stage plugin | `aa_forecast` |
| linked plugin config | `yaml/plugins/aa_forecast_brentoil.yaml` |
| selected job | `AAForecast` |
| job validated mode | `learned_auto` |
| training search mode | `training_auto` |
| runtime opt_n_trial | `100` |
| runtime opt_study_count | `5` |
| canonical projection study | `1` |
| execute study indices | `[1, 2, 3, 4, 5]` |
| CV policy | `horizon=4`, `step_size=6`, `n_windows=4` |
| uncertainty | enabled, `sample_count=50` |
| run root | `runs/feature_set_aaforecast_aa_forecast_brentoil` |

## Source / artifact provenance map

| Concern | Main code path | Main artifacts |
|---|---|---|
| entrypoint / bootstrap | `main.py` | none directly |
| config resolution | `app_config.py::load_app_config(...)` | `config/config.resolved.json` |
| job capability report | `runtime_support/runner.py::_validate_jobs(...)` | `config/capability_report.json` |
| run manifest | `runtime_support/runner.py::_build_resolved_artifacts(...)` + `_update_manifest_artifacts(...)` | `manifest/run_manifest.json` |
| aa_forecast stage metadata | `plugins/aa_forecast/runtime.py::materialize_aa_forecast_stage(...)` | `aa_forecast/config/stage_config.json`, `aa_forecast/manifest/stage_manifest.json` |
| study catalog / projection | `runtime_support/runner.py::_run_single_job(...)` | `models/AAForecast/study_catalog.json`, `best_params.json`, `training_best_params.json`, `optuna_study_summary.json` |
| fold predictions / metrics | `runtime_support/runner.py::_run_single_job(...)` + `plugins/aa_forecast/runtime.py::predict_aa_forecast_fold(...)` | `cv/AAForecast_forecasts.csv`, `cv/AAForecast_metrics_by_cutoff.csv` |
| uncertainty artifacts | `plugins/aa_forecast/runtime.py::_write_uncertainty_artifacts(...)` | `aa_forecast/uncertainty/*.json`, `aa_forecast/uncertainty/*.csv` |
| final fit summary | `runtime_support/runner.py::_run_single_job(...)` | `models/AAForecast/fit_summary.json` |

## Mermaid 1 — top-level orchestration

```mermaid
flowchart TD
    U[User command\nuv run python main.py --config yaml/experiment/feature_set_aaforecast/aa_forecast_brentoil.yaml]
    M0[main.py main entrypoint]
    M1[_reject_removed_args\npublic --output-root blocked]
    M2{_needs_reexec?}
    M3[os.execvpe into .venv/bin/python\nset PYTHONPATH and bootstrap env]
    M4[_run_cli\nparse args]
    C0[load_app_config]
    C1[resolve_config_path\nread yaml/experiment/feature_set_aaforecast/aa_forecast_brentoil.yaml]
    C2[detect stage plugin from payload\nconfig key = aa_forecast]
    C3[AAForecast stage validate_route]
    C4[load_aa_forecast_stage1\nread yaml/plugins/aa_forecast_brentoil.yaml]
    C5[plugin expands route into\njobs = single AAForecast job\ntraining_search.enabled = true]
    C6[load yaml/HPO/search_space.yaml\nnormalize payload\napply linked stage config]
    C7[LoadedConfig returned\nnormalized_payload includes aa_forecast.stage1 metadata]
    R0[run_loaded_config]
    R1[_resolve_run_roots\ndefault run root from parent dir + task.name]
    R2[_build_resolved_artifacts]
    A1[write config/config.resolved.json]
    A2[write manifest/run_manifest.json]
    R3[_validate_jobs + _validate_adapters]
    A3[write config/capability_report.json]
    R4[_initialize_study_catalogs]
    R5[resolve_study_selection\nselected study = none\nexecute studies 1 to 5\ncanonical study = 1]
    R6[per job: seed models/AAForecast/study_catalog.json\nand manifest study metadata]
    P0[get active stage plugin aa_forecast]
    P1[materialize_aa_forecast_stage]
    A4[write aa_forecast/config/stage_config.json]
    A5[write aa_forecast/manifest/stage_manifest.json]
    D0{validate_only?}
    D1[return validation payload only]
    D2{single selected job?}
    D3{should parallelize\nsingle-job tuning?}
    D4[_run_single_job_with_parallel_tuning\nworker lane path\nout of primary scope]
    S0[_run_single_job\nmain happy-path execution]
    S1[read data/df.csv\nsort by dt]
    S2[_resolve_freq -> W-MON]
    S3[_build_tscv_splits\n4 expanding-window folds]
    S4{job.validated_mode == learned_auto}
    T0[build study contexts for studies 1..5]
    T1[for each study\n_tune_main_job or load replay result]
    T2[write per-study best_params / training_best_params / optuna_study_summary / metadata]
    T3[build per-study visuals]
    T4[pick canonical projection study = 1]
    T5[copy canonical projection files to models/AAForecast/\nbest_params.json\ntraining_best_params.json\noptuna_study_summary.json]
    T6[build cross-study visuals + stage study catalog\nupdate manifest with study metadata]
    T7[effective_job <- best_params\neffective_training_params <- training best params]
    F0[fold replay loop begins\nusing canonical projected params]
    F1[accumulate cv_rows / metrics_rows / fold_payloads]
    F2[write cv/AAForecast_forecasts.csv\nwrite cv/AAForecast_metrics_by_cutoff.csv]
    F3[write models/AAForecast/fit_summary.json]
    F6[progress.model_finished\nrun-complete]
    O0[print JSON success summary\nexecuted_jobs = AAForecast]

    U --> M0 --> M1 --> M2
    M2 -- yes --> M3 --> M4
    M2 -- no --> M4
    M4 --> C0 --> C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> R0
    R0 --> R1 --> R2
    R2 --> A1
    R2 --> A2
    R0 --> R3 --> A3
    R0 --> R4 --> R5 --> R6
    R0 --> P0 --> P1 --> A4
    P1 --> A5
    R0 --> D0
    D0 -- yes --> D1
    D0 -- no --> D2
    D2 -- no --> O0
    D2 -- yes --> D3
    D3 -- yes --> D4
    D3 -- no --> S0
    S0 --> S1 --> S2 --> S3 --> S4
    S4 -- yes --> T0 --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> F0
    S4 -- no --> F0
    F0 --> F1 --> F2 --> F3 --> F4
    F4 -- yes --> F5 --> F6 --> O0
    F4 -- no --> F6 --> O0

    classDef artifact fill:#eef7ff,stroke:#1d70b8,color:#0b2540;
    classDef decision fill:#fff4db,stroke:#a36a00,color:#4d3400;
    classDef stage fill:#f2f0ff,stroke:#5b3fc4,color:#25145a;
    class A1,A2,A3,A4,A5,T2,T5,T6,F2,F3,O0 artifact;
    class M2,D0,D2,D3,S4,F4 decision;
    class C2,C3,C4,C5,P0,P1,T0,T1,T4,T7,F0 stage;
```

## Mermaid 2 — fold-internal execution

```mermaid
flowchart TD
    R0[_run_single_job]
    R1[source_df = read CSV\nsort by dt]
    R2[splits from build TSCV splits]
    R3[for each fold index train slice and test slice]
    R4[progress fold started phase replay]
    R5[fit and predict fold]

    subgraph FoldSetup[Fold data setup in runner]
        FS1[derive train_df and future_df from source slices]
        FS2[pass run_root because active stage plugin exists]
        FS3[delegate to stage plugin fold path]
    end

    subgraph AAFold[plugins/aa_forecast/runtime.py::predict_aa_forecast_fold(...)]
        AA1[concat train_df and future_df]
        AA2[resolve frequency from fold source data]
        AA3[derive effective config with training override]
        AA4[build fold diff context]
        AA5[transform training frame]
        AA6[build adapter inputs]
        AA7[build model with AA overrides]
        AA8[optional STAR precompute context]
        AA9[NeuralForecast instance single model]
        AA10[NeuralForecast.fit(...)]
        AA11[predict with adapter via nf.predict]
        AA12[extract target prediction frame]
        AA13{uncertainty.enabled?}
        AA14[select uncertainty predictions]
        AA15[repeat stochastic predictions for each dropout candidate]
        AA16[select min-std dropout per horizon step]
        AA17[write uncertainty artifacts]
        AA18[extract target actuals from future_df]
        AA19[return predictions actuals train_end train_df nf]
    end

    subgraph FoldBackInRunner[Back in runtime_support/runner.py]
        FR1[compute metrics]
        FR2[append forecast rows into cv_rows]
        FR3[append cutoff metrics into metrics_rows]
        FR5[progress fold completed]
        FR6{exception raised?}
        FR7[progress error]
        FR8[raise / fail run]
    end

    R0 --> R1 --> R2 --> R3 --> R4 --> R5
    R5 --> FS1 --> FS2 --> FS3 --> AA1 --> AA2 --> AA3 --> AA4 --> AA5 --> AA6 --> AA7 --> AA8 --> AA9 --> AA10 --> AA11 --> AA12 --> AA13
    AA13 -- yes --> AA14 --> AA15 --> AA16 --> AA17 --> AA18 --> AA19
    AA13 -- no --> AA18 --> AA19
    AA19 --> FR1 --> FR2 --> FR3 --> FR4 --> FR5 --> R3
    R5 --> FR6
    FR6 -- yes --> FR7 --> FR8
    FR6 -- no --> FS1

    subgraph Finalization[After all folds]
        Z1[write cv/AAForecast_forecasts.csv]
        Z2[write cv/AAForecast_metrics_by_cutoff.csv]
        Z3[write models/AAForecast/fit_summary.json]
        Z6[progress model finished run complete]
    end

    R3 --> Z1 --> Z2 --> Z3 --> Z4
    Z4 -- false --> Z5 --> Z6
    Z4 -- true --> Z6

    classDef artifact fill:#eef7ff,stroke:#1d70b8,color:#0b2540;
    classDef decision fill:#fff4db,stroke:#a36a00,color:#4d3400;
    classDef phase fill:#f6fff2,stroke:#4f8a10,color:#143800;
    class AA13,FR6,Z4 decision;
    class AA17,Z1,Z2,Z3 artifact;
    class FoldSetup,AAFold,FoldBackInRunner,Finalization phase;
```

## Verified fold windows

| Fold | Train range | Test range | Train rows | Test rows |
|---:|---|---|---:|---:|
| 0 | `2015-01-05 -> 2025-10-06` | `2025-10-13 -> 2025-11-03` | 562 | 4 |
| 1 | `2015-01-05 -> 2025-11-17` | `2025-11-24 -> 2025-12-15` | 568 | 4 |
| 2 | `2015-01-05 -> 2025-12-29` | `2026-01-05 -> 2026-01-26` | 574 | 4 |
| 3 | `2015-01-05 -> 2026-02-09` | `2026-02-16 -> 2026-03-09` | 580 | 4 |

## Notes on what the diagrams intentionally abstract
- The top-level diagram shows **abstract study fan-out** (`studies 1..5`) and **canonical replay** (`study 1`) without expanding every helper beneath `_tune_main_job(...)`.
- The fold diagram shows the **runtime handoff boundaries** and the main data/artifact flow, not every helper used to construct `backcast_panel` or every column written to CSV.
- `validate_only` and worker-internal parallel tuning exist in code, but they are not the primary runtime path documented here.

## Verification commands used for this document
- `UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --validate-only --config yaml/experiment/feature_set_aaforecast/aa_forecast_brentoil.yaml`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... load_app_config(...) / _resolve_run_roots(...) ... PY`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... _build_tscv_splits(...) / _resolve_freq(...) ... PY`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... resolve_study_selection(...) ... PY`
