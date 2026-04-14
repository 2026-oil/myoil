# `uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml` 동작 전체 pseudo

이 문서는 아래 **실행 명령이 내부에서 어떤 순서로 흘러가는지**를 현재 레포 코드 기준으로 pseudo 형태로 정리한 것이다.

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml
```

설명 기준:
- `main.py`
- `app_config.py`
- `runtime_support/runner.py`
- `runtime_support/forecast_models.py`
- `plugins/aa_forecast/runtime.py`
- `neuralforecast/models/aaforecast/model.py`
- `neuralforecast/models/aaforecast/models/informer.py`
- `neuralforecast/models/informer.py`
- validate-only로 갱신한 `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/config/config.resolved.json`

---

## 0. 이번 실행에서 실제로 고정되는 resolved contract

`aaforecast-informer.yaml` + shared setting + plugin YAML 이 합쳐진 뒤 실제 핵심 실행값은 아래처럼 고정된다.

- task name: `brentoil_case1_parity_aaforecast_informer`
- dataset: `data/df.csv`
- target: `Com_BrentCrudeOil`
- hist exog: 10개
- runtime:
  - `transformations_target = diff`
  - `transformations_exog = diff`
- training:
  - `input_size = 64`
  - `max_steps = 800`
  - `val_size = 16`
  - `loss = mse`
- cv:
  - `horizon = 2`
  - `n_windows = 1`
  - `step_size = 4`
- single job:
  - `model = AAForecast`
- AA plugin:
  - `backbone = informer`
  - `hidden_size = 128`
  - `n_head = 4`
  - `encoder_layers = 2`
  - `dropout = 0.0`
  - `linear_hidden_size = 96`
  - `factor = 3`
  - `decoder_hidden_size = 128`
  - `decoder_layers = 4`
  - `season_length = 4`
  - `lowess_frac = 0.35`
  - `lowess_delta = 0.01`
  - `thresh = 3.5`
  - STAR exog = `GPRD_THREAT`
  - retrieval = `false`
  - uncertainty = `true`
  - dropout candidates = `0.005 .. 0.3`
  - sample_count = `30`

즉 이번 명령은 개념적으로

```text
experiment yaml
-> shared settings merge
-> aa_forecast plugin yaml merge
-> resolved AAForecast(informer) job 1개 생성
-> Brent target / hist exog diff 변환
-> AAForecast Informer fold 1회 학습/예측
-> uncertainty dropout replay
-> cv + summary artifact 기록
```

이다.

---

## 1. 최상위 CLI bootstrap pseudo

```python
# file: main.py

def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    reject_removed_args(args)  # --output-root 금지

    if needs_reexec_into_venv():
        env = build_env()
        env['NEURALFORECAST_RUNTIME_BOOTSTRAPPED'] = '1'
        execvpe('.venv/bin/python', ['.venv/bin/python', 'main.py', *args], env)

    os.environ.update(build_env())
    return _run_cli(args, repo_root=WORKSPACE_ROOT)
```

핵심은:
- `uv run python main.py ...` 로 들어와도
- 최종 실행 주체는 `main.py -> runtime_support.runner` 경로
- `.venv/bin/python` 이 있으면 거기로 재실행
- 실제 런타임 진입점은 `_run_cli(...)`

---

## 2. `_run_cli` 에서 config 를 읽는 pseudo

```python
# file: main.py::_run_cli

def _run_cli(argv, repo_root):
    args = parser.parse_args(argv)
    config_path = args.config or args.config_path

    loaded = runtime_support.runner.load_app_config(
        repo_root,
        config_path=config_path,
        config_toml_path=args.config_toml,
        shared_settings_path=args.setting,
    )

    if loaded.jobs_fanout_specs:
        ...  # 이번 케이스는 해당 없음

    runtime_support.runner.run_loaded_config(repo_root, loaded, args)
```

이번 케이스는 `jobs_fanout` 이 없고, 바로 `run_loaded_config(...)` 로 간다.

---

## 3. `load_app_config(...)` 단계 pseudo

이 단계에서 **실험 YAML만 읽는 것이 아니라**, shared settings 와 plugin YAML 까지 합쳐서 최종 runnable config 를 만든다.

```python
# file: app_config.py::load_app_config

def load_app_config(repo_root, config_path):
    source_path, source_type = resolve_config_path(repo_root, config_path)
    payload = load_yaml_or_toml(source_path)

    reject_direct_linked_aa_forecast_config(payload)

    if yaml_uses_repo_shared_settings(source_path):
        shared_settings_payload = load_shared_settings_for_yaml_app_config(repo_root)
        payload = merge_shared_settings_into_payload(payload, shared_settings_payload)

    ensure_plugins_loaded()
    stage_plugin = get_stage_plugin_for_payload(payload)  # aa_forecast plugin 선택

    raw_stage_config = stage_plugin.normalize_config(payload['aa_forecast'])
    stage_payload_probe = stage_plugin.validate_route(...)

    if search_space_needed(payload, stage_payload_probe):
        search_space_contract = load_search_space_contract(repo_root)

    config = _normalize_payload(
        payload,
        dataset_base_dir=...,
        search_space=...,
        stage_scope='aa_forecast',
    )

    if aa_forecast.config_path exists:
        stage_plugin_loaded = stage_plugin.load_stage(...)
        config = stage_plugin.apply_stage_to_config(config, stage_plugin_loaded)

    config = _resolve_aa_forecast_stage_plugin_config(config)
    normalized_payload = config.to_dict() + stage_plugin.stage_normalized_payload(...)

    return LoadedConfig(config=config, normalized_payload=normalized_payload, ...)
```

### 여기서 실제로 일어나는 중요한 일

1. `yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml` 를 읽는다.
2. `yaml/setting/setting.yaml` 를 합쳐서 `input_size=64`, `horizon=2`, `diff` 설정 등을 채운다.
3. `aa_forecast.config_path = yaml/plugins/aa_forecast/aa_forecast_parity_informer_stability_dh.yaml` 를 따라가서 plugin YAML 을 읽는다.
4. plugin 이 top-level job 으로 `AAForecast` 를 소유하도록 구성한다.
5. STAR exog / non-STAR exog 를 실제 dataset.hist_exog_cols 기준으로 resolve 한다.
6. 결과적으로 top-level `jobs = [{model: "AAForecast", ...}]` 하나가 만들어진다.

즉 사용자 YAML은 직접 Informer 를 top-level 로 실행하는 것이 아니라,
실제로는

```text
experiment config
+ shared setting
+ aa_forecast linked config
-> resolved AAForecast(informer backbone) runtime contract
```

으로 바뀐다.

---

## 4. `run_loaded_config(...)` 단계 pseudo

```python
# file: runtime_support/runner.py::run_loaded_config

def run_loaded_config(repo_root, loaded, args):
    selected_jobs = _selected_jobs(loaded, args.jobs)   # => [AAForecast]
    run_root = _resolve_run_roots(repo_root, loaded, output_root=args.output_root)

    paths = _build_resolved_artifacts(repo_root, loaded, run_root)
    _validate_jobs(loaded, selected_jobs, capability_report_path)
    _validate_adapters(loaded, selected_jobs)
    _initialize_study_catalogs(run_root, loaded, selected_jobs)

    if active_stage_plugin exists:
        stage_plugin.materialize_stage(...)

    if args.validate_only:
        return {ok: True, ...}

    if len(selected_jobs) == 1:
        _prune_model_run_artifacts(run_root, 'AAForecast')
        _run_single_job(loaded, AAForecast_job, run_root, manifest_path=...)
        summary_artifacts = _build_summary_artifacts(run_root)
        print({ok: True, executed_jobs: ['AAForecast'], ...})
        return ...
```

### 이 단계에서 생기는 파일

실행 초반부터 아래 artifact 들이 먼저 준비된다.

```text
runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/
  config/config.resolved.json
  config/capability_report.json
  manifest/run_manifest.json
  aa_forecast/config/stage_config.json
  aa_forecast/manifest/stage_manifest.json
```

즉 **학습을 시작하기 전에** 이미
- resolved config
- capability report
- manifest
- aa_forecast stage config
가 남는다.

---

## 5. single-job replay loop pseudo

이번 config 는 auto tuning 이 아니라 `learned_fixed` 이므로, Optuna main search loop 없이 바로 fold replay 로 간다.

```python
# file: runtime_support/runner.py::_run_single_job

def _run_single_job(loaded, job, run_root):
    source_df = pd.read_csv(dataset.path).sort_values(dt_col)
    freq = _resolve_freq(loaded, source_df)
    splits = _build_tscv_splits(len(source_df), loaded.config.cv)
    # 이번 케이스: n_windows=1 이므로 split 1개

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        target_predictions, target_actuals, cutoff, train_df, nf = _fit_and_predict_fold(
            loaded,
            job,
            source_df=source_df,
            freq=freq,
            train_idx=train_idx,
            test_idx=test_idx,
            training_override={},
            run_root=run_root,
        )

        write_loss_curve_artifact(...)
        metrics = compute_metrics(target_actuals, target_predictions['AAForecast'])
        append_cv_rows(...)
        append_metrics_rows(...)

    write_csv(run_root / 'cv/AAForecast_forecasts.csv', cv_rows)
    write_csv(run_root / 'cv/AAForecast_metrics_by_cutoff.csv', metrics_rows)
    write_json(run_root / 'models/AAForecast/fit_summary.json', ...)
```

이번 config 기준 핵심은:
- dataset 전체를 읽은 뒤
- 마지막 1개 evaluation window 에 대해
- train / future 를 자르고
- 그 fold 하나를 AAForecast Informer path 로 fit/predict 한다.

---

## 6. fold 진입 pseudo: plugin-owned top-level path

`AAForecast` 는 aa_forecast stage plugin 이 직접 소유하는 top-level job 이다.
그래서 `_fit_and_predict_fold(...)` 는 generic learned model path로 가지 않고 plugin 의 `predict_fold(...)` 로 바로 보낸다.

```python
# file: runtime_support/runner.py::_fit_and_predict_fold

def _fit_and_predict_fold(loaded, job, train_idx, test_idx, ...):
    train_df = source_df.iloc[train_idx]
    future_df = source_df.iloc[test_idx]

    plugin = _plugin_owned_top_level_job(loaded, job.model)
    if plugin is not None:  # AAForecast => True
        return plugin.predict_fold(
            loaded,
            job,
            train_df=train_df,
            future_df=future_df,
            run_root=run_root,
            params_override=params_override,
            training_override=training_override,
        )
```

그리고 aa_forecast plugin 은 내부적으로 아래 함수를 호출한다.

```python
# file: plugins/aa_forecast/plugin.py

def predict_fold(...):
    return predict_aa_forecast_fold(...)
```

---

## 7. `predict_aa_forecast_fold(...)` 전체 pseudo

이 함수가 이번 실행의 **실질적 AAForecast runtime 중심**이다.

```python
# file: plugins/aa_forecast/runtime.py::predict_aa_forecast_fold

def predict_aa_forecast_fold(loaded, job, train_df, future_df, run_root):
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    source_df = concat(train_df, future_df)
    freq = resolve_freq(loaded, source_df)

    effective_config = _effective_config(loaded, training_override)

    diff_context = _build_fold_diff_context(loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)

    adapter_inputs = _build_adapter_inputs(
        loaded,
        transformed_train_df,
        future_df,
        job,
        dt_col,
    )

    merged_params_override = {
        **_aa_params_override(effective_config),
        **(params_override or {}),
    }

    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get('n_series'),
        params_override=merged_params_override,
    )

    model.set_star_precompute_context(enabled=True, fold_key=...)

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(adapter_inputs.fit_df, static_df=adapter_inputs.static_df, val_size=16)

    predict_kwargs = {'df': adapter_inputs.fit_df, 'static_df': adapter_inputs.static_df}
    if adapter_inputs.futr_df is not None:
        predict_kwargs['futr_df'] = adapter_inputs.futr_df
    predictions = nf.predict(**predict_kwargs)

    target_predictions = predictions[predictions['unique_id'] == target_col].reset_index(drop=True)
    target_predictions = restore_target_predictions(
        target_predictions,
        prediction_col='AAForecast',
        diff_context=diff_context,
    )

    context_frame, context_active = _build_fold_context_frame(model=nf.models[0], train_df=transformed_train_df, ...)
    write_context_artifacts(...)

    if uncertainty.enabled:
        uncertainty_summary = _select_uncertainty_predictions(...)
        target_predictions['AAForecast'] = uncertainty_summary['mean']
        write_uncertainty_artifacts(...)

    if retrieval.enabled:
        ...  # 이번 config는 false 이므로 skip

    return target_predictions, target_actuals, cutoff, train_df, nf
```

### 이번 케이스에서 중요한 branch

#### 7-1. diff transform 이 먼저 적용된다

이번 resolved config 는
- `transformations_target = diff`
- `transformations_exog = diff`

이므로 모델에 들어가는 학습용 fold 는 raw level 이 아니라

```text
train_df(raw)
-> diff_context 생성
-> transformed_train_df(target diff + hist exog diff)
```

를 거친다.

#### 7-2. future_df 는 예측 대상 원본 구간이다

학습 입력은 diff-space 로 바뀌지만,
최종 반환 prediction 은 `_restore_target_predictions(...)` 를 통해 다시 target scale 로 복원된다.

#### 7-3. retrieval path 는 완전히 비활성화다

이번 plugin config 는 `retrieval.enabled = false` 이므로
- event memory bank 구축
- neighbor retrieval
- posthoc blend
는 실행되지 않는다.

#### 7-4. uncertainty replay 는 활성화다

기본 예측 후 다시 여러 dropout 후보를 사용해 repeated predict 를 돌리고,
그 결과에서 selected path의 mean/std 와 broadcast된 selected dropout 을 정리한다.

---

## 8. `build_model(...)` 에서 실제 AAForecast Informer 인스턴스가 만들어지는 pseudo

```python
# file: runtime_support/forecast_models.py::build_model

def build_model(config, job, params_override):
    shared_kwargs = {
        'h': config.cv.horizon,
        'input_size': config.training.input_size,
        'max_steps': config.training.max_steps,
        'scaler_type': config.training.scaler_type,
        'loss': resolve_loss(config.training.loss),
        'valid_loss': resolve_loss(config.training.loss),
        'hist_exog_list': config.dataset.hist_exog_cols,
        ...
    }

    accepted = filter_kwargs_for_model_signature(
        shared_kwargs + job.params + params_override
    )

    model = AAForecast(**accepted)
    return model
```

이번 케이스에서 `params_override` 로 실제 주입되는 AA-specific 핵심 값은 대략 아래다.

```python
{
  'backbone': 'informer',
  'thresh': 3.5,
  'star_hist_exog_list': ['GPRD_THREAT'],
  'non_star_hist_exog_list': [
      'BS_Core_Index_A', 'GPRD', 'GPRD_ACT', 'BS_Core_Index_B',
      'BS_Core_Index_C', 'Idx_OVX', 'Com_LMEX',
      'Com_BloombergCommodity_BCOM', 'Idx_DxyUSD'
  ],
  'star_hist_exog_tail_modes': ['upward'],
  'lowess_frac': 0.35,
  'lowess_delta': 0.01,
  'uncertainty_enabled': True,
  'uncertainty_dropout_candidates': [...],
  'uncertainty_sample_count': 30,
}
```

즉 최종적으로 만들어지는 것은

```text
AAForecast(backbone='informer', diff-transformed inputs, uncertainty on, retrieval off)
```

이다.

---

## 9. AAForecast Informer 내부 forward pseudo

이제부터는 `nf.fit(...)` / `nf.predict(...)` 내부에서 실제 모델 forward 가 어떻게 흘러가는지다.

### 9-1. constructor 단계 pseudo

```python
# file: neuralforecast/models/aaforecast/model.py::__init__

class AAForecast(BaseModel):
    def __init__(..., backbone='informer', ...):
        self.backbone = 'informer'
        self.star = STARFeatureExtractor(...)

        feature_size = (
            1                      # insample target
            + len(non_star_hist_exog_list)
            + 4                    # target trend/seasonal/anomaly/residual
            + 4 * len(star_hist_exog_list)
        )

        self.encoder = build_aaforecast_backbone('informer', feature_size=feature_size, ...)
        self.attention = CriticalSparseAttention(hidden_size=self.encoder_hidden_size, ...)
        self.sequence_adapter = Linear(input_size, h) if h > input_size else None

        self.event_summary_projector = MLP(...)
        self.event_trajectory_projector = MLP(...)
        self.regime_time_projector = MLP(...)
        self.memory_query_projector = MLP(...)
        self.memory_key_projector = Linear(...)
        self.memory_value_projector = Linear(...)
        self.memory_token_shock_head = MLP(...)
        self.memory_token_shock_gate = MLP(...)
        self.informer_decoder = InformerHorizonAwareHead(...)
```

이번 resolved config 에서는
- STAR exog 1개 (`GPRD_THREAT`)
- non-STAR exog 9개
- target 1개
이므로 encoder feature composition 은

```text
1 + 9 + 4 + (4 * 1) = 18 channels
```

기준으로 흘러간다.

---

### 9-2. STAR feature 생성 pseudo

```python
# file: neuralforecast/models/aaforecast/model.py::_compute_star_outputs

def _compute_star_outputs(insample_y, hist_exog):
    target_star = STAR(insample_y, tail_modes=('two_sided',))

    star_hist_exog = select_hist_exog(hist_exog, star_hist_exog_indices)
    non_star_hist_exog = select_hist_exog(hist_exog, non_star_hist_exog_indices)

    star_hist_outputs = STAR(star_hist_exog, tail_modes=('upward',))
    non_star_star_outputs = STAR(non_star_hist_exog, tail_modes=('two_sided', ...))

    target_count = count_active_channels(target_star['critical_mask'])
    star_hist_count = count_active_channels(star_hist_outputs['critical_mask'])
    non_star_star_count = count_active_channels(non_star_star_outputs['critical_mask'])

    regime_activity, regime_count = build_non_star_regime_activity(...)
    event_summary = build_event_summary(...)
    event_trajectory = build_event_trajectory(...)
    non_star_regime = build_non_star_regime(...)

    return {
        target_trend,
        target_seasonal,
        target_anomalies,
        target_residual,
        star_hist_trend,
        star_hist_seasonal,
        star_hist_anomalies,
        star_hist_residual,
        critical_mask,
        count_active_channels,
        channel_activity,
        event_summary,
        event_trajectory,
        non_star_regime,
        regime_intensity,
        regime_density,
    }
```

즉 raw/diff window 자체가 사라지는 것은 아니다.
실제 encoder input 에는 `insample_y` 와 non-STAR exog 가 그대로 남아 있고,
그 옆에 STAR 기반 분해/이벤트 요약 채널이 추가된다.

---

### 9-3. encoder input 조립 pseudo

```python
# file: neuralforecast/models/aaforecast/model.py::forward

def forward(windows_batch):
    insample_y = windows_batch['insample_y']     # diff target window
    hist_exog = windows_batch['hist_exog']       # diff hist exog window

    star_payload = _compute_star_outputs(insample_y, hist_exog)

    encoder_parts = []
    encoder_parts.append(insample_y)
    encoder_parts.append(non_star_hist_exog)
    encoder_parts.extend([
        target_trend,
        target_seasonal,
        target_anomalies,
        target_residual,
    ])
    encoder_parts.extend([
        star_hist_trend,
        star_hist_seasonal,
        star_hist_anomalies,
        star_hist_residual,
    ])

    encoder_input = torch.cat(encoder_parts, dim=2)
```

즉 이번 Informer encoder 입력은 개념적으로

```text
[target diff]
+ [non-star exog diff 9개]
+ [target STAR 4채널]
+ [GPRD_THREAT STAR 4채널]
```

이다.

---

### 9-4. Informer backbone adapter pseudo

```python
# file: neuralforecast/models/aaforecast/models/informer.py

class InformerBackboneAdapter:
    def forward(inputs):
        signal = inputs[..., :1]   # 첫 채널만 main signal
        exog = inputs[..., 1:]     # 나머지는 exogenous marks
        return InformerEncoderOnly(signal, exog)
```

그리고 `InformerEncoderOnly` 내부는 원래 Informer 구성요소를 encoder-only 형태로 쓴다.

```python
# file: neuralforecast/models/informer.py

class InformerEncoderOnly(nn.Module):
    def __init__(...):
        self.enc_embedding = DataEmbedding(c_in=1, exog_input_size=feature_size-1, ...)
        self.encoder = TransEncoder([
            TransEncoderLayer(
                AttentionLayer(ProbAttention(...), hidden_size, n_head),
                ...
            )
            for _ in range(encoder_layers)
        ])

    def forward(signal, exog):
        enc = self.enc_embedding(signal, exog)
        enc_out, _ = self.encoder(enc, attn_mask=None)
        return enc_out
```

즉 AAForecast Informer adapter 는

```text
AA 18채널 전체를 그대로 c_in 으로 넣는 방식이 아니라
첫 채널만 signal
나머지 17채널은 Informer exogenous marks
```

로 routing 한다.

---

### 9-5. sparse attention + event/regime pseudo

Informer encoder output 이후에는 time-state projection + anomaly-aware sparse attention 이 붙는다.

```python
backbone_states = self.encoder(encoder_input)
hidden_states = self.encoder.project_to_time_states(backbone_states)

critical_mask = star_payload['critical_mask']
count_active_channels = star_payload['count_active_channels']
channel_activity = star_payload['channel_activity']

event_summary = star_payload['event_summary']
event_trajectory = star_payload['event_trajectory']
non_star_regime = star_payload['non_star_regime']
regime_intensity = star_payload['regime_intensity']
regime_density = star_payload['regime_density']

attention_hidden_states = hidden_states + project_regime_time_context(regime_intensity, regime_density)
attended_states, _ = self.attention(
    attention_hidden_states,
    critical_mask,
    count_active_channels,
    channel_activity,
)
```

즉 Informer encoder state 위에 다시
- critical mask
- active channel count
- channel activity
를 이용한 **anomaly-aware sparse attention** 이 한 번 더 적용된다.

---

### 9-6. Informer horizon-aware decoder pseudo

```python
# file: neuralforecast/models/aaforecast/model.py::_decode_informer_forecast

def _decode_informer_forecast(...):
    hidden_aligned, attended_aligned = _build_time_decoder_features(...)

    regime_time_latent = _project_regime_time_context(regime_intensity, regime_density)
    regime_time_aligned = _align_horizon(regime_time_latent, h=self.h, ...)

    event_context = _project_event_summary(event_summary)
    event_path = _project_event_trajectory(event_trajectory)

    pooled_context = _build_memory_pooled_context(
        hidden_states,
        attended_states,
        event_context,
        event_path,
        non_star_regime,
        regime_intensity,
        regime_density,
    )

    decoder_input = torch.cat([
        hidden_aligned + regime_time_aligned,
        attended_aligned + regime_time_aligned,
    ], dim=-1)

    delta_forecast = self.informer_decoder(
        decoder_input,
        event_context,
        event_path,
        non_star_regime,
        pooled_context,
        memory_signal,
        anchor_value=anchor_level[:, -1, :],
        memory_token=memory_token,
        memory_bank=memory_bank,
    )

    anchor = anchor_level[:, -1:, :]
    return anchor + delta_forecast
```

여기서 핵심은 Informer decoder 가 단순히 encoder hidden 만 보는 것이 아니라,

- `event_summary`
- `event_trajectory`
- `non_star_regime`
- `pooled_context`
- `memory_token`
- `memory_bank`
- `anchor(last input diff input)`

을 함께 받아서 horizon별 delta 를 만든다는 점이다.

마지막 반환은

```text
final_diff_prediction = last_diff_input_anchor + delta_forecast
```

형태다.

여기서 anchor 는 raw level 이 아니라 현재 fold의 `insample_y` 마지막 값, 즉 이번 run 에서는 **diff-space anchor** 다.
그 다음 runtime 바깥에서 `_restore_target_predictions(...)` 가 적용되므로,
최종 CSV 예측이 다시 Brent level scale 로 복원된다.

---

## 10. uncertainty selection pseudo

이번 config 에서는 uncertainty 가 켜져 있으므로, 첫 predict 이후 여러 dropout 후보로 같은 fold prediction 을 반복한다.

```python
# file: plugins/aa_forecast/runtime.py::_select_uncertainty_predictions

def _select_uncertainty_predictions(...):
    for dropout_p in dropout_candidates:
        model.configure_stochastic_inference(enabled=True, dropout_p=dropout_p)

        for sample_idx in range(sample_count):
            predictions = nf.predict(..., random_seed=sample_seed)
            restored = restore_target_predictions(...)
            samples.append(restored)
            collect_decoder_debug_metrics(...)

        candidate_means.append(mean(samples))
        candidate_stds.append(std(samples))
        candidate_semantic_scores.append(...)

    selected_path_idx = choose_single_dropout_path(...)
    selected_dropout = repeat(selected_dropout_scalar, horizon)
    return {
        'mean': selected_mean,
        'std': selected_std,
        'selected_dropout': selected_dropout,
        'candidate_stats': ...,
        'candidate_samples': ...,
    }
```

이번 run 에서는 대략 아래 artifact 가 생긴다.

```text
aa_forecast/uncertainty/<cutoff>.json
aa_forecast/uncertainty/<cutoff>.csv
aa_forecast/uncertainty/<cutoff>.candidate_stats.csv
aa_forecast/uncertainty/<cutoff>.candidate_samples.csv
```

즉 최종 forecast 는 “단일 deterministic 1회 예측값”이 아니라,
**dropout candidate들 중 선택된 단일 path의 uncertainty-aware 결과** 로 덮어써진다.

---

## 11. retrieval 이 꺼져 있을 때의 pseudo

이번 config 는 `retrieval.enabled = false` 이므로 아래 블록은 진입하지 않는다.

```python
if retrieval_cfg.enabled:
    build_event_memory_bank(...)
    build_event_query(...)
    retrieve_event_neighbors(...)
    blend_retrieval_result(...)
```

즉 이번 명령은

```text
STAR/event-aware informer
+ uncertainty replay
- retrieval blend
```

구성이다.

---

## 12. 최종 산출물 pseudo

full command 가 끝나면 큰 흐름은 아래처럼 남는다.

```text
runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/
  config/
    config.resolved.json
    capability_report.json
  manifest/
    run_manifest.json
  aa_forecast/
    config/stage_config.json
    manifest/stage_manifest.json
    context/<cutoff>.csv
    context/<cutoff>.json
    uncertainty/<cutoff>.json
    uncertainty/<cutoff>.csv
    uncertainty/<cutoff>.candidate_stats.csv
    uncertainty/<cutoff>.candidate_samples.csv
  cv/
    AAForecast_forecasts.csv
    AAForecast_metrics_by_cutoff.csv
  models/AAForecast/
    fit_summary.json
    folds/fold_0/loss_curve.png
    folds/fold_0/loss_curve_every_10_global_steps.csv
  summary/
    leaderboard.csv
    result.csv
    sample.md
    last_fold_combined.png
    ...
```

단, `--validate-only` 는 학습/예측 전에 멈추므로 아래 초기 artifact 까지는 보장한다.
- resolved config
- run manifest
- stage config
- stage manifest
- capability report

---

## 13. 한 번에 보는 end-to-end pseudo

```python
command = "uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml"

# CLI bootstrap
argv = parse(command)
reexec_into_venv_if_needed()

# Config resolution
experiment_yaml = load_yaml('yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml')
shared_setting = load_yaml('yaml/setting/setting.yaml')
plugin_yaml = load_yaml('yaml/plugins/aa_forecast/aa_forecast_parity_informer_stability_dh.yaml')
resolved = merge_and_normalize(experiment_yaml, shared_setting, plugin_yaml)
# => one AAForecast job with informer backbone

# Run setup
run_root = 'runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer'
write_resolved_config_and_manifest(run_root, resolved)
materialize_aa_forecast_stage(run_root, resolved)

# CV (n_windows=1)
source_df = read_csv('data/df.csv').sort_by('dt')
train_df, future_df = build_last_window_split(source_df, horizon=2, step_size=4, n_windows=1)

# Fold preprocessing
diff_context = build_diff_context(train_df, target_diff=True, exog_diff=True)
transformed_train_df = apply_diff_transform(train_df, diff_context)
adapter_inputs = build_adapter_inputs(transformed_train_df, future_df)

# Model build
model = AAForecast(
    backbone='informer',
    input_size=64,
    h=2,
    star_hist_exog_list=['GPRD_THREAT'],
    non_star_hist_exog_list=[...9 vars...],
    uncertainty_enabled=True,
    uncertainty_dropout_candidates=[...],
    uncertainty_sample_count=30,
    ...
)
# retrieval on/off 는 model ctor 인자가 아니라 stage_cfg.retrieval.enabled runtime branch 로 제어됨

# Fit / predict
nf = NeuralForecast([model], freq=infer_freq(source_df['dt']))
nf.fit(adapter_inputs.fit_df, static_df=adapter_inputs.static_df, val_size=16)
base_predictions = nf.predict(df=adapter_inputs.fit_df, static_df=adapter_inputs.static_df, futr_df=adapter_inputs.futr_df)
restored_predictions = restore_diff_predictions(
    filter_target_rows(base_predictions, model_col='AAForecast'),
    diff_context=diff_context,
)

# Internal AAForecast Informer forward
star_payload = STAR(diff_target_window, diff_hist_exog_window)
encoder_input = concat(
    target_diff,
    non_star_hist_exog_diff,
    target_STAR_4channels,
    GPRD_THREAT_STAR_4channels,
)
backbone_states = InformerEncoderOnly(signal=encoder_input[..., :1], exog=encoder_input[..., 1:])
hidden_states = project_to_time_states(backbone_states)
attended_states = anomaly_sparse_attention(hidden_states, critical_mask, channel_activity)
delta_forecast = horizon_aware_decoder(
    hidden_states,
    attended_states,
    event_summary,
    event_trajectory,
    non_star_regime,
    pooled_context,
    memory_token,
    memory_bank,
)
model_output = last_diff_input_anchor + delta_forecast

# Uncertainty replay
selected_predictions = select_uncertainty_predictions_via_dropout_replay(nf, restored_predictions)

# No retrieval blend
final_predictions = selected_predictions

# Artifact writeback
write_context_artifacts(run_root)
write_uncertainty_artifacts(run_root)
write_cv_metrics_and_forecasts(run_root)
write_summary_artifacts(run_root)
```

---

## 14. 요약 한 줄

이번 명령의 본질은

```text
Brent/Exog raw rows
-> fold split
-> target/exog diff transform
-> STAR decomposition + event summary
-> Informer encoder-only backbone
-> anomaly-aware sparse attention
-> horizon-aware decoder with event/regime/memory inputs
-> last-anchor add-back
-> runtime diff restore
-> uncertainty dropout replay
-> final forecast/metrics/summary artifacts 저장
```

이다.
