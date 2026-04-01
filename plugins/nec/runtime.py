from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app_config import JobConfig, LoadedConfig
from .config import NecClassifierConfig, NecConfig, NecForecastConfig


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stage_root(run_root: Path) -> Path:
    root = run_root / "nec"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if layer_dim > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=effective_dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class _RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        x = self.fc1(self.bn1(encoded[:, -1, :]))
        x = self.fc2(self.bn2(x))
        x = self.fc3(self.bn3(x))
        return x


class _ClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(self.bn1(encoded[:, -1, :])))


class _Regressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, layer_dim, dropout)
        self.head = _RegressionHead(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, layer_dim, dropout)
        self.head = _ClassifierHead(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _NecPredictor:
    def __init__(self, *, loaded: LoadedConfig, train_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
        self.loaded = loaded
        self.train_df = train_df.reset_index(drop=True).copy()
        self.future_df = future_df.reset_index(drop=True).copy()
        self.config: NecConfig = loaded.config.stage_plugin_config
        self.target_col = loaded.config.dataset.target_col
        self.dt_col = loaded.config.dataset.dt_col
        self.history_steps = self.config.history_steps or loaded.config.training.input_size
        self.horizon = len(self.future_df)
        self.device = _device()
        self.rng = np.random.default_rng(self.loaded.config.runtime.random_seed)
        self._validate_frames()
        (
            self.feature_matrix,
            self.diff_norm_target,
            self.target_mean,
            self.target_std,
            self.extreme_flags,
        ) = self._build_training_features()
        total_windows = len(self.diff_norm_target) - self.history_steps - self.horizon + 1
        if total_windows <= 0:
            raise ValueError("NEC requires at least history_steps + horizon training rows in each fold")
        self.train_positions, self.val_positions = self._split_positions(total_windows)
        if not self.train_positions:
            raise ValueError("NEC requires at least one training window in each fold")
        if not self.val_positions:
            raise ValueError("NEC requires at least one validation window in each fold")

    def _validate_frames(self) -> None:
        required_columns = [self.target_col, *self.config.hist_columns]
        missing = [c for c in required_columns if c not in self.train_df.columns]
        if missing:
            raise ValueError("NEC configured hist_columns are missing from the training frame: " + ", ".join(missing))
        for column in required_columns:
            if self.train_df[column].isna().any():
                raise ValueError(f"NEC does not support NaN values in training column {column!r}")

    def _diff_normalize(self, values: np.ndarray) -> tuple[np.ndarray, float, float]:
        if self.config.preprocessing.mode != "diff_std":
            raise ValueError("NEC currently supports only diff_std preprocessing")
        diffs = np.concatenate(([0.0], np.diff(values.astype(float))))
        mean = float(diffs.mean())
        std = float(diffs.std())
        if std <= 0:
            raise ValueError("NEC diff preprocessing requires non-zero standard deviation")
        return ((diffs - mean) / std).astype(np.float32), mean, std

    def _build_training_features(self) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        target = self.train_df[self.target_col].astype(float).to_numpy()
        diff_norm_target, mean, std = self._diff_normalize(target)
        features = [diff_norm_target.reshape(-1, 1)]
        for column in self.config.hist_columns:
            column_values = self.train_df[column].astype(float).to_numpy()
            diff_norm, _col_mean, _col_std = self._diff_normalize(column_values)
            features.append(diff_norm.reshape(-1, 1))
        if self.config.preprocessing.probability_feature:
            gm = GaussianMixture(
                n_components=self.config.preprocessing.gmm_components,
                random_state=self.loaded.config.runtime.random_seed,
            )
            target_prob = diff_norm_target.reshape(-1, 1)
            gm.fit(target_prob)
            proba = gm.predict_proba(target_prob)
            weights = gm.weights_
            prob_in_distribution = (proba * weights).sum(axis=1)
            prob_like_outlier = 1.0 - prob_in_distribution
            features.append(prob_like_outlier.reshape(-1, 1))
        matrix = np.concatenate(features, axis=1).astype(np.float32)
        extreme_flags = (np.abs(diff_norm_target) > self.config.preprocessing.epsilon).astype(np.float32)
        return matrix, diff_norm_target.astype(np.float32), mean, std, extreme_flags

    def _split_positions(self, total_windows: int) -> tuple[list[int], list[int]]:
        positions = list(range(self.history_steps, len(self.diff_norm_target) - self.horizon + 1))
        val_windows = min(max(1, self.config.validation.windows), len(positions) - 1)
        if val_windows <= 0 or val_windows >= len(positions):
            raise ValueError("NEC validation split leaves no training windows")
        return positions[:-val_windows], positions[-val_windows:]

    def _window_inputs(self, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.feature_matrix[start - self.history_steps : start]
        y_reg = self.diff_norm_target[start : start + self.horizon]
        y_cls = self.extreme_flags[start : start + self.horizon]
        return x, y_reg, y_cls

    def _sample_positions(self, positions: list[int], model_cfg: NecForecastConfig | NecClassifierConfig) -> list[int]:
        if not positions:
            raise ValueError("NEC cannot sample from an empty position set")
        if not model_cfg.oversampling:
            size = model_cfg.train_volume
            chosen = self.rng.choice(positions, size=size, replace=size > len(positions))
            return [int(item) for item in chosen.tolist()]
        extreme_positions = [pos for pos in positions if self.extreme_flags[pos : pos + self.horizon].sum() >= 1]
        normal_positions = [pos for pos in positions if self.extreme_flags[pos : pos + self.horizon].sum() == 0]
        if not extreme_positions:
            raise ValueError("NEC oversampling requested but no extreme windows exist in training fold")
        normal_target = int(round(model_cfg.train_volume * model_cfg.normal_ratio))
        extreme_target = model_cfg.train_volume - normal_target
        sampled: list[int] = []
        sampled.extend(
            int(item)
            for item in self.rng.choice(extreme_positions, size=extreme_target, replace=extreme_target > len(extreme_positions)).tolist()
        )
        if normal_target:
            if not normal_positions:
                raise ValueError("NEC oversampling requested normal windows but none exist in training fold")
            sampled.extend(
                int(item)
                for item in self.rng.choice(normal_positions, size=normal_target, replace=normal_target > len(normal_positions)).tolist()
            )
        self.rng.shuffle(sampled)
        return sampled

    def _build_loader(self, starts: list[int], *, classifier_mode: bool, batch_size: int) -> DataLoader:
        x_batches = []
        y_batches = []
        for start in starts:
            x, y_reg, y_cls = self._window_inputs(start)
            x_batches.append(x)
            y_batches.append(y_cls if classifier_mode else y_reg)
        x_tensor = torch.tensor(np.asarray(x_batches), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_batches), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        effective_batch = min(batch_size, len(dataset))
        if effective_batch < 2:
            raise ValueError("NEC effective batch size must be at least 2")
        return DataLoader(dataset, batch_size=effective_batch, shuffle=True, num_workers=0, drop_last=True)

    def _train_regressor(self, model_cfg: NecForecastConfig, *, normal_role: bool) -> tuple[_Encoder, _RegressionHead, float]:
        starts = self._sample_positions(self.train_positions, model_cfg)
        loader = self._build_loader(starts, classifier_mode=False, batch_size=model_cfg.batch_size)
        encoder = _Encoder(self.feature_matrix.shape[1], model_cfg.hidden_dim, model_cfg.layer_dim, model_cfg.dropout).to(self.device)
        head = _RegressionHead(model_cfg.hidden_dim, self.horizon).to(self.device)
        encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=model_cfg.encoder_lr)
        head_optimizer = torch.optim.Adam(head.parameters(), lr=model_cfg.head_lr)
        best_loss = float("inf")
        best_encoder = encoder.state_dict()
        best_head = head.state_dict()
        early_stop = 0
        previous_loss = float("inf")
        for _epoch in range(model_cfg.epochs):
            encoder.train(); head.train()
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                encoder_optimizer.zero_grad(); head_optimizer.zero_grad()
                predictions = head(encoder(x_batch))
                if normal_role:
                    mask = (torch.abs(y_batch) <= self.config.preprocessing.epsilon).float()
                else:
                    mask = (torch.abs(y_batch) > self.config.preprocessing.epsilon).float()
                losses = (predictions - y_batch) ** 2 * mask
                loss = losses.sum() / torch.clamp(mask.sum(), min=1.0)
                loss.backward(); encoder_optimizer.step(); head_optimizer.step()
            encoder.eval(); head.eval()
            val_loss = self._evaluate_regressor(encoder, head, normal_role=normal_role)
            if val_loss < best_loss:
                best_loss = val_loss
                best_encoder = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if val_loss > previous_loss:
                early_stop += 1
            else:
                early_stop = 0
            if early_stop >= model_cfg.early_stop_patience:
                break
            previous_loss = val_loss
        encoder.load_state_dict(best_encoder)
        head.load_state_dict(best_head)
        encoder.eval(); head.eval()
        return encoder, head, best_loss

    def _train_classifier(self, model_cfg: NecClassifierConfig) -> tuple[_Encoder, _ClassifierHead, float]:
        starts = self._sample_positions(self.train_positions, model_cfg)
        x_batches = []
        y_batches = []
        for start in starts:
            x, _y_reg, y_cls = self._window_inputs(start)
            x_batches.append(x[:, :1])
            y_batches.append(y_cls)
        x_tensor = torch.tensor(np.asarray(x_batches), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_batches), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        effective_batch = min(model_cfg.batch_size, len(dataset))
        if effective_batch < 2:
            raise ValueError("NEC effective batch size must be at least 2")
        loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True, num_workers=0, drop_last=True)
        encoder = _Encoder(1, model_cfg.hidden_dim, model_cfg.layer_dim, model_cfg.dropout).to(self.device)
        head = _ClassifierHead(model_cfg.hidden_dim, self.horizon).to(self.device)
        encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=model_cfg.encoder_lr)
        head_optimizer = torch.optim.Adam(head.parameters(), lr=model_cfg.head_lr)
        bce = nn.BCELoss(reduction="sum")
        mse = nn.MSELoss(reduction="sum")
        best_f1 = -1.0
        best_encoder = encoder.state_dict(); best_head = head.state_dict(); early_stop = 0; previous = -1.0
        for _epoch in range(model_cfg.epochs):
            encoder.train(); head.train()
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device); y_batch = y_batch.to(self.device)
                encoder_optimizer.zero_grad(); head_optimizer.zero_grad()
                predictions = head(encoder(x_batch))
                loss = model_cfg.bce_weight * bce(predictions ** 2, y_batch) + model_cfg.mse_weight * mse(predictions, y_batch)
                loss.backward(); encoder_optimizer.step(); head_optimizer.step()
            encoder.eval(); head.eval()
            score = self._evaluate_classifier(encoder, head)
            if score > best_f1:
                best_f1 = score
                best_encoder = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if score < previous:
                early_stop += 1
            else:
                early_stop = 0
            if score >= 1.0 or early_stop >= model_cfg.early_stop_patience:
                break
            previous = score
        encoder.load_state_dict(best_encoder)
        head.load_state_dict(best_head)
        encoder.eval(); head.eval()
        return encoder, head, best_f1

    def _predict_sequence(self, encoder: _Encoder, head: nn.Module, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x[np.newaxis, ...], dtype=torch.float32, device=self.device)
            preds = head(encoder(x_tensor)).detach().cpu().numpy()[0]
        return preds.astype(np.float32)

    def _diff_denormalize(self, pred_diff_norm: np.ndarray) -> np.ndarray:
        denorm = pred_diff_norm.astype(float) * self.target_std + self.target_mean
        output = np.zeros_like(denorm, dtype=float)
        previous = float(self.train_df[self.target_col].iloc[-1])
        for idx, diff in enumerate(denorm):
            previous = previous + float(diff)
            output[idx] = previous
        return output.astype(np.float32)

    def _evaluate_regressor(self, encoder: _Encoder, head: _RegressionHead, *, normal_role: bool) -> float:
        rmses = []
        for start in self.val_positions:
            x, y_reg, _y_cls = self._window_inputs(start)
            pred_norm = self._predict_sequence(encoder, head, x)
            pred_level = self._diff_denormalize(pred_norm)
            actual = self.train_df[self.target_col].astype(float).to_numpy()[start : start + self.horizon]
            rmses.append(math.sqrt(float(np.square(actual - pred_level).mean())))
        return float(np.mean(rmses))

    def _evaluate_classifier(self, encoder: _Encoder, head: _ClassifierHead) -> float:
        all_true: list[int] = []
        all_pred: list[int] = []
        for start in self.val_positions:
            x, _y_reg, y_cls = self._window_inputs(start)
            logits = self._predict_sequence(encoder, head, x[:, :1])
            preds = (logits >= 0.5).astype(int)
            all_true.extend(int(v) for v in y_cls)
            all_pred.extend(int(v) for v in preds)
        tp = sum(1 for p, a in zip(all_pred, all_true) if p == a == 1)
        fp = sum(1 for p, a in zip(all_pred, all_true) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(all_pred, all_true) if p == 0 and a == 1)
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run(self) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
        if not any(self.extreme_flags[pos : pos + self.horizon].sum() >= 1 for pos in self.train_positions):
            raise ValueError("NEC oversampling requested but no extreme window exists in training fold")
        normal_encoder, normal_head, normal_val = self._train_regressor(self.config.normal, normal_role=True)
        extreme_encoder, extreme_head, extreme_val = self._train_regressor(self.config.extreme, normal_role=False)
        classifier_encoder, classifier_head, classifier_f1 = self._train_classifier(self.config.classifier)

        history = self.feature_matrix[-self.history_steps :]
        classifier_probs = self._predict_sequence(classifier_encoder, classifier_head, history[:, :1])
        normal_norm = self._predict_sequence(normal_encoder, normal_head, history)
        extreme_norm = self._predict_sequence(extreme_encoder, extreme_head, history)
        normal_level = self._diff_denormalize(normal_norm)
        extreme_level = self._diff_denormalize(extreme_norm)
        gates = (classifier_probs >= 0.5).astype(np.float32)
        merged = np.where(gates == 1, extreme_level, normal_level).astype(np.float32)
        predictions = pd.DataFrame(
            {
                "unique_id": [self.target_col] * self.horizon,
                "ds": pd.to_datetime(self.future_df[self.dt_col]).reset_index(drop=True),
                "NEC": merged,
            }
        )
        curve_frame = pd.DataFrame(
            {
                "normal_val_rmse": [normal_val] * self.horizon,
                "extreme_val_rmse": [extreme_val] * self.horizon,
                "classifier_val_f1": [classifier_f1] * self.horizon,
            }
        )
        return (
            predictions,
            self.future_df[self.target_col].reset_index(drop=True),
            pd.to_datetime(self.train_df[self.dt_col].iloc[-1]),
            self.train_df,
            curve_frame,
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
    variant = str(job.params.get("variant", "paper")).strip().lower()
    if variant != "paper":
        raise ValueError("NEC currently supports only jobs[NEC].params.variant=paper")
    predictor = _NecPredictor(loaded=loaded, train_df=train_df, future_df=future_df)
    result = predictor.run()
    if run_root is not None:
        stage_root = _stage_root(run_root)
        summary = {
            "history_steps": predictor.history_steps,
            "use_probability_feature": predictor.config.preprocessing.probability_feature,
            "hist_columns_used": list(predictor.config.hist_columns),
        }
        _write_json(stage_root / "nec_fold_summary.json", summary)
    return result


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
        "shared_scaler_type_seen": loaded.config.training.scaler_type,
        "feature_columns": [loaded.config.dataset.target_col, *stage_loaded.config.hist_columns],
        "use_probability_feature": stage_loaded.config.preprocessing.probability_feature,
        "gmm_components": stage_loaded.config.preprocessing.gmm_components,
        "validate_only": validate_only,
        "stage1_resolved_config_path": str(resolved_path),
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
