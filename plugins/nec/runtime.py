from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from neuralforecast import NeuralForecast

from app_config import JobConfig, LoadedConfig
from runtime_support.adapters import build_univariate_inputs
from runtime_support.forecast_models import build_model, capabilities_for
from .config import NecConfig, nec_active_hist_columns, nec_branch_configs
from .losses import NecClassifierLoss, NecSelectivePointLoss

_PROBABILITY_COLUMN = "__nec_probability_feature"


@dataclass(frozen=True)
class _BranchRunResult:
    name: str
    model_name: str
    variables: tuple[str, ...]
    predictions: np.ndarray
    raw_predictions: np.ndarray
    actual: np.ndarray
    sampled_series_count: int
    oversampled_window_count: int
    fitted_model: Any | None = None


@dataclass(frozen=True)
class _NecCheckpointBundle:
    branch_results: dict[str, _BranchRunResult]

    def save(self, path: str | Path) -> None:
        import fsspec
        import torch

        payload: dict[str, Any] = {
            "format": "nec-branch-checkpoint-bundle-v1",
            "branches": {},
        }
        for branch_name, branch_result in self.branch_results.items():
            fitted_model = branch_result.fitted_model
            if fitted_model is None or not hasattr(fitted_model, "save"):
                raise TypeError(
                    f"NEC branch {branch_name} does not expose a savable fitted model"
                )
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_path = Path(tmp_dir) / f"{branch_name}.pt"
                fitted_model.save(checkpoint_path)
                payload["branches"][branch_name] = {
                    "model_name": branch_result.model_name,
                    "variables": list(branch_result.variables),
                    "checkpoint_bytes": checkpoint_path.read_bytes(),
                }
        with fsspec.open(path, "wb") as handle:
            torch.save(payload, handle)


def _stage_root(run_root: Path) -> Path:
    root = run_root / "nec"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    return root


def _summary_root(run_root: Path) -> Path:
    root = run_root / "summary" / "nec"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_freq(train_df: pd.DataFrame, dt_col: str) -> str:
    series = pd.to_datetime(train_df[dt_col])
    inferred = pd.infer_freq(series)
    if inferred is not None:
        return inferred
    if len(series) < 2:
        raise ValueError("NEC could not infer dataset frequency from fewer than 2 timestamps")
    return pd.tseries.frequencies.to_offset(series.iloc[1] - series.iloc[0]).freqstr


def _branch_target_column(branch_name: str) -> str:
    return f"__nec_{branch_name}_target"


def _next_fold_index(run_root: Path) -> int:
    classifier_dir = _summary_root(run_root) / "classifier"
    if not classifier_dir.exists():
        return 0
    existing = sorted(classifier_dir.glob("fold_*.csv"))
    return len(existing)


class _NecPredictor:
    def __init__(self, *, loaded: LoadedConfig, train_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
        self.loaded = loaded
        self.train_df = train_df.reset_index(drop=True).copy()
        self.future_df = future_df.reset_index(drop=True).copy()
        self.config: NecConfig = loaded.config.stage_plugin_config
        self.target_col = loaded.config.dataset.target_col
        self.dt_col = loaded.config.dataset.dt_col
        self.history_steps = loaded.config.training.input_size
        self.horizon = len(self.future_df)
        self.freq = _infer_freq(self.train_df, self.dt_col)
        self.active_hist_columns = nec_active_hist_columns(self.config)
        self._validate_frames()
        self.diff_norm_target, self.target_mean, self.target_std = self._diff_normalize(
            self.train_df[self.target_col].astype(float).to_numpy()
        )
        self.normalized_columns = self._build_normalized_columns()
        self.probability_feature = self._build_probability_feature()
        self.extreme_flags = (np.abs(self.diff_norm_target) > self.config.preprocessing.epsilon).astype(np.float32)
        self.normal_feature_matrix = self._feature_matrix_for_branch("normal")
        self.extreme_feature_matrix = self._feature_matrix_for_branch("extreme")
        self.classifier_feature_matrix = self._feature_matrix_for_branch("classifier")
        total_windows = len(self.diff_norm_target) - self.history_steps - self.horizon + 1
        if total_windows <= 0:
            raise ValueError("NEC requires at least training.input_size + horizon training rows in each fold")
        self.actual_target = self.future_df[self.target_col].astype(float).to_numpy(dtype=np.float32)
        self.actual_classifier = self._future_extreme_flags()

    def _uses_direct_branch_fit(self, branch_model_name: str) -> bool:
        return bool(capabilities_for(branch_model_name).multivariate)

    def _validate_frames(self) -> None:
        required_columns = [self.target_col, *self.active_hist_columns]
        missing = [column for column in required_columns if column not in self.train_df.columns]
        if missing:
            raise ValueError(
                "NEC configured branch variables are missing from the training frame: " + ", ".join(missing)
            )
        for column in required_columns:
            if self.train_df[column].isna().any():
                raise ValueError(f"NEC does not support NaN values in training column {column!r}")
        for column in self.active_hist_columns:
            if self.future_df[column].isna().any():
                raise ValueError(f"NEC does not support NaN values in future column {column!r}")

    def _diff_normalize(self, values: np.ndarray) -> tuple[np.ndarray, float, float]:
        if self.config.preprocessing.mode != "diff_std":
            raise ValueError("NEC currently supports only diff_std preprocessing")
        diffs = np.concatenate(([0.0], np.diff(values.astype(float))))
        mean = float(diffs.mean())
        std = float(diffs.std())
        if std <= 0:
            raise ValueError("NEC diff preprocessing requires non-zero standard deviation")
        return ((diffs - mean) / std).astype(np.float32), mean, std

    def _build_normalized_columns(self) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {self.target_col: self.diff_norm_target}
        for column in self.active_hist_columns:
            values = self.train_df[column].astype(float).to_numpy()
            diff_norm, _mean, _std = self._diff_normalize(values)
            normalized[column] = diff_norm.astype(np.float32)
        return normalized

    def _build_probability_feature(self) -> np.ndarray:
        gm = GaussianMixture(
            n_components=self.config.preprocessing.gmm_components,
            random_state=self.loaded.config.runtime.random_seed,
        )
        target_prob = self.diff_norm_target.reshape(-1, 1)
        gm.fit(target_prob)
        proba = gm.predict_proba(target_prob)
        weights = gm.weights_
        prob_in_distribution = (proba * weights).sum(axis=1)
        prob_like_outlier = 1.0 - prob_in_distribution
        return prob_like_outlier.reshape(-1, 1).astype(np.float32)

    def _feature_columns_for_branch(self, branch_name: str) -> tuple[str, ...]:
        branch = nec_branch_configs(self.config)[branch_name]
        columns = [self.target_col, *branch.variables]
        if branch_name in {"classifier", "extreme"}:
            columns.append(_PROBABILITY_COLUMN)
        return tuple(columns)

    def _feature_matrix_for_branch(self, branch_name: str) -> np.ndarray:
        arrays = []
        for column in self._feature_columns_for_branch(branch_name):
            if column == _PROBABILITY_COLUMN:
                arrays.append(self.probability_feature)
            else:
                arrays.append(self.normalized_columns[column].reshape(-1, 1))
        return np.concatenate(arrays, axis=1).astype(np.float32)

    def _branch_training_frame(self, branch_name: str) -> pd.DataFrame:
        branch = nec_branch_configs(self.config)[branch_name]
        payload: dict[str, Any] = {
            self.dt_col: pd.to_datetime(self.train_df[self.dt_col]).reset_index(drop=True),
            _branch_target_column(branch_name): (
                self.extreme_flags if branch_name == "classifier" else self.diff_norm_target
            ),
        }
        for column in branch.variables:
            payload[column] = self.normalized_columns[column]
        if branch_name in {"classifier", "extreme"}:
            payload[_PROBABILITY_COLUMN] = self.probability_feature.reshape(-1)
        return pd.DataFrame(payload)

    def _branch_config(self, branch_name: str) -> tuple[LoadedConfig, JobConfig]:
        branch = nec_branch_configs(self.config)[branch_name]
        hist_cols = list(branch.variables)
        if branch_name in {"classifier", "extreme"}:
            hist_cols.append(_PROBABILITY_COLUMN)
        dataset = replace(
            self.loaded.config.dataset,
            target_col=_branch_target_column(branch_name),
            hist_exog_cols=tuple(hist_cols),
            futr_exog_cols=(),
            static_exog_cols=(),
        )
        training = replace(self.loaded.config.training, scaler_type="identity")
        config = replace(self.loaded.config, dataset=dataset, training=training)
        job = JobConfig(model=branch.model, params=dict(branch.model_params))
        return replace(self.loaded, config=config), job

    def _effective_val_size(self, branch_loaded: LoadedConfig) -> int:
        return max(branch_loaded.config.training.val_size, self.horizon)

    def _training_sample_length(self, branch_loaded: LoadedConfig) -> int:
        return self.history_steps + self._effective_val_size(branch_loaded) + self.horizon

    def _window_extreme_flags(self, start: int) -> np.ndarray:
        prediction_start = start + self.history_steps
        prediction_end = prediction_start + self.horizon
        return self.extreme_flags[prediction_start:prediction_end]

    def _sampled_branch_fit_df(
        self,
        branch_name: str,
        train_frame: pd.DataFrame,
        branch_loaded: LoadedConfig,
    ) -> tuple[pd.DataFrame, int]:
        sequence_length = self._training_sample_length(branch_loaded)
        candidate_count = len(train_frame) - sequence_length + 1
        if candidate_count <= 0:
            raise ValueError(
                "NEC requires enough rows to build sampled training windows with validation context "
                f"(need {sequence_length}, got {len(train_frame)})"
            )
        windows: list[tuple[int, int]] = [
            (start, start + sequence_length) for start in range(candidate_count)
        ]
        branch_cfg = nec_branch_configs(self.config)[branch_name]
        oversampled_window_count = 0
        if branch_cfg.oversample_extreme_windows:
            extreme_windows = [
                (start, end)
                for start, end in windows
                if bool(self._window_extreme_flags(start).any())
            ]
            windows.extend(extreme_windows)
            oversampled_window_count = len(extreme_windows)
        sampled_frames: list[pd.DataFrame] = []
        for sample_idx, (start, end) in enumerate(windows):
            sample = train_frame.iloc[start:end].copy()
            sample.rename(
                columns={
                    self.dt_col: "ds",
                    branch_loaded.config.dataset.target_col: "y",
                },
                inplace=True,
            )
            sample["ds"] = pd.to_datetime(sample["ds"])
            sample.insert(
                0,
                "unique_id",
                f"{branch_loaded.config.dataset.target_col}__sample_{sample_idx:05d}",
            )
            sampled_frames.append(sample)
        return pd.concat(sampled_frames, ignore_index=True), oversampled_window_count

    def _branch_losses(self, branch_name: str) -> tuple[Any, Any]:
        branch_cfg = nec_branch_configs(self.config)[branch_name]
        if branch_name == "classifier":
            alpha = 2.0 if branch_cfg.alpha is None else branch_cfg.alpha
            beta = 0.5 if branch_cfg.beta is None else branch_cfg.beta
            loss = NecClassifierLoss(alpha=alpha, beta=beta)
            return loss, loss
        base_loss = (
            "mae" if self.loaded.config.training.loss.lower() == "mae" else "mse"
        )
        loss = NecSelectivePointLoss(
            epsilon=self.config.preprocessing.epsilon,
            branch_name="normal" if branch_name == "normal" else "extreme",
            base_loss=base_loss,
        )
        return loss, loss

    def _predict_branch(self, branch_name: str) -> _BranchRunResult:
        branch_loaded, branch_job = self._branch_config(branch_name)
        train_frame = self._branch_training_frame(branch_name)
        inference_inputs = build_univariate_inputs(
            train_frame,
            branch_job,
            dataset=branch_loaded.config.dataset,
            dt_col=self.dt_col,
            future_df=None,
        )
        if self._uses_direct_branch_fit(branch_job.model):
            fit_df = inference_inputs.fit_df
            oversampled_window_count = 0
            sampled_series_count = int(fit_df["unique_id"].nunique())
        else:
            fit_df, oversampled_window_count = self._sampled_branch_fit_df(
                branch_name,
                train_frame,
                branch_loaded,
            )
            sampled_series_count = int(fit_df["unique_id"].nunique())
        training_loss, valid_loss = self._branch_losses(branch_name)
        model = build_model(
            branch_loaded.config,
            branch_job,
            params_override=branch_job.params,
            loss_override=training_loss,
            valid_loss_override=valid_loss,
        )
        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(
            fit_df,
            static_df=None,
            val_size=self._effective_val_size(branch_loaded),
        )
        predictions = nf.predict(
            df=inference_inputs.fit_df,
            static_df=inference_inputs.static_df,
        )
        prediction_values = (
            predictions.loc[predictions["unique_id"] == branch_loaded.config.dataset.target_col, branch_job.model]
            .reset_index(drop=True)
            .astype(float)
            .to_numpy(dtype=np.float32)
        )
        if len(prediction_values) != self.horizon:
            raise ValueError(
                f"NEC {branch_name} branch returned {len(prediction_values)} predictions; expected horizon={self.horizon}"
            )
        if branch_name == "classifier":
            classifier_probs = (
                1.0 / (1.0 + np.exp(-prediction_values.astype(np.float64)))
            ).astype(np.float32)
            return _BranchRunResult(
                name=branch_name,
                model_name=branch_job.model,
                variables=nec_branch_configs(self.config)[branch_name].variables,
                predictions=classifier_probs,
                raw_predictions=prediction_values,
                actual=self.actual_classifier,
                sampled_series_count=sampled_series_count,
                oversampled_window_count=oversampled_window_count,
                fitted_model=nf.models[0],
            )
        level_predictions = self._diff_denormalize(prediction_values)
        return _BranchRunResult(
            name=branch_name,
            model_name=branch_job.model,
            variables=nec_branch_configs(self.config)[branch_name].variables,
            predictions=level_predictions,
            raw_predictions=prediction_values,
            actual=self.actual_target,
            sampled_series_count=sampled_series_count,
            oversampled_window_count=oversampled_window_count,
            fitted_model=nf.models[0],
        )

    def _diff_denormalize(self, pred_diff_norm: np.ndarray) -> np.ndarray:
        denorm = pred_diff_norm.astype(float) * self.target_std + self.target_mean
        output = np.zeros_like(denorm, dtype=float)
        previous = float(self.train_df[self.target_col].iloc[-1])
        for idx, diff in enumerate(denorm):
            previous = previous + float(diff)
            output[idx] = previous
        return output.astype(np.float32)

    def _future_extreme_flags(self) -> np.ndarray:
        raw_target = self.train_df[self.target_col].astype(float).tolist() + self.future_df[self.target_col].astype(float).tolist()
        diffs = np.diff(np.asarray(raw_target, dtype=float))
        normalized = ((diffs[-self.horizon :] - self.target_mean) / self.target_std).astype(np.float32)
        return (np.abs(normalized) > self.config.preprocessing.epsilon).astype(np.float32)

    def _merge_predictions(
        self,
        *,
        classifier_probs: np.ndarray,
        normal_level: np.ndarray,
        extreme_level: np.ndarray,
    ) -> np.ndarray:
        mode = self.config.inference.mode
        classifier_probs = classifier_probs.astype(np.float32)
        if mode == "soft_weighted":
            return (
                classifier_probs * extreme_level
                + (1.0 - classifier_probs) * normal_level
            ).astype(np.float32)
        if mode == "soft_weighted_inverse":
            return (
                (1.0 - classifier_probs) * extreme_level
                + classifier_probs * normal_level
            ).astype(np.float32)
        gates = (classifier_probs >= self.config.inference.threshold).astype(np.float32)
        if mode == "hard_threshold_inverse":
            gates = (classifier_probs < self.config.inference.threshold).astype(np.float32)
        return np.where(gates == 1, extreme_level, normal_level).astype(np.float32)

    def run(self) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None, dict[str, _BranchRunResult]]:
        branch_results = {
            branch_name: self._predict_branch(branch_name)
            for branch_name in ("normal", "extreme", "classifier")
        }
        classifier_probs = branch_results["classifier"].predictions
        normal_level = branch_results["normal"].predictions
        extreme_level = branch_results["extreme"].predictions
        merged = self._merge_predictions(
            classifier_probs=classifier_probs,
            normal_level=normal_level,
            extreme_level=extreme_level,
        )
        predictions = pd.DataFrame(
            {
                "unique_id": [self.target_col] * self.horizon,
                "ds": pd.to_datetime(self.future_df[self.dt_col]).reset_index(drop=True),
                "NEC": merged,
            }
        )
        return (
            predictions,
            self.future_df[self.target_col].reset_index(drop=True),
            pd.to_datetime(self.train_df[self.dt_col].iloc[-1]),
            self.train_df,
            None,
            branch_results,
        )


def _plot_branch_result(path: Path, *, ds: pd.Series, actual: np.ndarray, predicted: np.ndarray, title: str, actual_label: str, predicted_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(pd.to_datetime(ds), actual, label=actual_label, linewidth=2.2, color="black")
    axis.plot(pd.to_datetime(ds), predicted, label=predicted_label, linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("ds")
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_branch_summary_artifacts(
    run_root: Path,
    *,
    fold_idx: int,
    ds: pd.Series,
    branch_results: dict[str, _BranchRunResult],
) -> None:
    summary_root = _summary_root(run_root)
    for branch_name, branch_result in branch_results.items():
        branch_dir = summary_root / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        csv_path = branch_dir / f"fold_{fold_idx:03d}.csv"
        png_path = branch_dir / f"fold_{fold_idx:03d}.png"
        payload = pd.DataFrame(
            {
                "branch": [branch_name] * len(ds),
                "model": [branch_result.model_name] * len(ds),
                "ds": pd.to_datetime(ds),
                "actual": branch_result.actual,
                "predicted": branch_result.predictions,
            }
        )
        if branch_name == "classifier":
            payload["predicted_probability"] = branch_result.predictions
            payload["predicted_raw"] = branch_result.raw_predictions
            actual_label = "actual_extreme_flag"
            predicted_label = "predicted_probability"
        else:
            actual_label = "actual_target"
            predicted_label = branch_name
        payload.to_csv(csv_path, index=False)
        _plot_branch_result(
            png_path,
            ds=pd.to_datetime(ds),
            actual=branch_result.actual,
            predicted=branch_result.predictions,
            title=f"NEC {branch_name} fold {fold_idx:03d}",
            actual_label=actual_label,
            predicted_label=predicted_label,
        )


def prepare_nec_fold_inputs(
    loaded: LoadedConfig,
    job: JobConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    run_root: Path | None = None,
) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, None]:
    del job, run_root
    return loaded, train_df, future_df, None


def predict_nec_fold(
    loaded: LoadedConfig,
    job: JobConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    run_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
    predictor = _NecPredictor(loaded=loaded, train_df=train_df, future_df=future_df)
    result = predictor.run()
    predictions, actuals, train_end_ds, prepared_train_df, curve_frame, branch_results = result
    if run_root is not None:
        stage_root = _stage_root(run_root)
        fold_idx = _next_fold_index(run_root)
        summary = {
            "history_steps_source": "training.input_size",
            "history_steps_value": predictor.history_steps,
            "probability_feature_forced": True,
            "active_hist_columns": list(predictor.active_hist_columns),
            "merge_mode": predictor.config.inference.mode,
            "inference_threshold": predictor.config.inference.threshold,
            "branches": {
                branch_name: {
                    "model": branch_result.model_name,
                    "variables": list(branch_result.variables),
                    "predicted_row_count": len(branch_result.predictions),
                    "sampled_series_count": branch_result.sampled_series_count,
                    "oversampled_window_count": branch_result.oversampled_window_count,
                }
                for branch_name, branch_result in branch_results.items()
            },
        }
        _write_json(stage_root / "nec_fold_summary.json", summary)
        _write_branch_summary_artifacts(
            run_root,
            fold_idx=fold_idx,
            ds=predictions["ds"],
            branch_results=branch_results,
        )
    checkpoint_bundle = _NecCheckpointBundle(branch_results=branch_results)
    return predictions, actuals, train_end_ds, prepared_train_df, checkpoint_bundle


def materialize_nec_stage(
    *,
    loaded: LoadedConfig,
    selected_jobs: Iterable[Any],
    run_root: Path,
    main_resolved_path: Path,
    main_capability_path: Path,
    main_manifest_path: Path,
    entrypoint_version: str,
    validate_only: bool,
) -> dict[str, Any]:
    del entrypoint_version, selected_jobs
    if not loaded.config.stage_plugin_config.enabled:
        return {}
    stage_root = _stage_root(run_root)
    stage_loaded = loaded.stage_plugin_loaded
    if stage_loaded is None:
        return {}
    resolved_path = stage_root / "config" / "config.resolved.json"
    manifest_path = stage_root / "manifest" / "run_manifest.json"
    _write_json(resolved_path, stage_loaded.normalized_payload)
    metadata = {
        "selected_config_path": str(stage_loaded.source_path),
        "paper_faithful_preprocessing": stage_loaded.config.preprocessing.mode,
        "paper_overrides_shared_scaler": True,
        "inference_mode": stage_loaded.config.inference.mode,
        "inference_threshold": stage_loaded.config.inference.threshold,
        "shared_scaler_type_seen": loaded.config.training.scaler_type,
        "active_hist_columns": list(nec_active_hist_columns(loaded.config.stage_plugin_config)),
        "history_steps_source": "training.input_size",
        "history_steps_value": loaded.config.training.input_size,
        "probability_feature_forced": True,
        "gmm_components": stage_loaded.config.preprocessing.gmm_components,
        "epsilon": stage_loaded.config.preprocessing.epsilon,
        "validate_only": validate_only,
        "stage1_resolved_config_path": str(resolved_path),
        "summary_nec_root": str(run_root / "summary" / "nec"),
        "branches": {
            name: {
                "model": branch.model,
                "variables": list(branch.variables),
                "model_params": dict(branch.model_params),
                "alpha": branch.alpha,
                "beta": branch.beta,
                "oversample_extreme_windows": branch.oversample_extreme_windows,
            }
            for name, branch in nec_branch_configs(loaded.config.stage_plugin_config).items()
        },
    }
    manifest_payload = {
        "entrypoint_version": "nec-stage-plugin",
        "selected_config_path": str(stage_loaded.source_path),
        "paper_overrides_shared_scaler": True,
    }
    _write_json(manifest_path, manifest_payload)
    metadata["stage1_manifest_path"] = str(manifest_path)
    loaded.normalized_payload.setdefault("nec", {}).update(metadata)
    for path in (main_resolved_path, main_capability_path, main_manifest_path):
        payload = _load_json(path)
        payload.setdefault("nec", {})
        payload["nec"].update(metadata)
        _write_json(path, payload)
    return metadata


def load_nec_stage_config(_repo_root: Path, loaded: LoadedConfig) -> LoadedConfig | None:
    del _repo_root
    if not loaded.config.stage_plugin_config.enabled or loaded.stage_plugin_loaded is None:
        return None
    stage_loaded = loaded.stage_plugin_loaded
    return LoadedConfig(
        config=replace(loaded.config, stage_plugin_config=stage_loaded.config),
        source_path=stage_loaded.source_path,
        source_type=stage_loaded.source_type,
        normalized_payload=stage_loaded.normalized_payload,
        input_hash=stage_loaded.input_hash,
        resolved_hash=stage_loaded.resolved_hash,
        search_space_path=stage_loaded.search_space_path,
        search_space_hash=stage_loaded.search_space_hash,
        search_space_payload=stage_loaded.search_space_payload,
    )
