from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from data_utils import (
    ScalerState,
    compute_router_features_from_window,
    fit_scaler,
    inverse_transform_values,
    make_window_arrays,
    transform_values,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(array: np.ndarray | torch.Tensor, device: torch.device = DEVICE) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _extract_pred(output):
    return output[0] if isinstance(output, tuple) else output


def _finite_loss(loss: torch.Tensor) -> bool:
    return bool(torch.isfinite(loss).item() and loss.detach().item() <= 100.0)


class _LinearTwoHeadNet(nn.Module):
    def __init__(self, context_length: int, hidden_dim: int, output_dim: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


class _OrderAwareDeltaNet(nn.Module):
    def __init__(self, context_length: int, hidden_dim: int, delta_hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, delta_hidden_dim),
            nn.ReLU(),
        )
        self.h1_head = nn.Linear(delta_hidden_dim, 1)
        self.delta_head = nn.Linear(delta_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        h1 = self.h1_head(z).squeeze(-1)
        delta_raw = self.delta_head(z).squeeze(-1)
        delta = F.softplus(delta_raw)
        h2 = h1 + delta
        return torch.stack([h1, h2], dim=-1), delta_raw


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        context_length: int,
        horizon: int,
        patch_length: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if context_length < patch_length:
            raise ValueError("context_length must be >= patch_length")
        self.context_length = context_length
        self.horizon = horizon
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = 1 + (context_length - patch_length) // stride
        self.patch_proj = nn.Linear(patch_length, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        tokens = self.patch_proj(patches)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class _ITransformerNet(nn.Module):
    def __init__(
        self,
        context_length: int,
        horizon: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.series_norm = nn.LayerNorm(context_length)
        self.time_mixer = nn.Linear(context_length, context_length, bias=False)
        self.value_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.series_norm(x)
        x = self.time_mixer(x)
        tokens = self.value_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class _GatedEnsembleNet(nn.Module):
    def __init__(
        self,
        context_length: int,
        trend_context_length: int,
        horizon: int,
        hidden_dim: int,
        router_hidden_dim: int,
        router_temperature: float,
        uniform_router: bool,
    ):
        super().__init__()
        self.context_length = context_length
        self.trend_context_length = trend_context_length
        self.horizon = horizon
        self.router_temperature = router_temperature
        self.uniform_router = uniform_router
        self.trend_expert = _LinearTwoHeadNet(trend_context_length, hidden_dim, horizon)
        self.reversion_expert = _LinearTwoHeadNet(context_length, hidden_dim, horizon)
        self.router = nn.Sequential(
            nn.Linear(3, router_hidden_dim),
            nn.Tanh(),
            nn.Linear(router_hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, router_features: Optional[torch.Tensor] = None):
        trend_x = x[:, -self.trend_context_length :]
        revert_x = x[:, -self.context_length :]
        trend_pred = self.trend_expert(trend_x)
        revert_pred = self.reversion_expert(revert_x)
        if self.uniform_router:
            weights = torch.full((x.size(0), 2), 0.5, dtype=x.dtype, device=x.device)
        else:
            if router_features is None:
                router_features = torch.stack(
                    [torch.std(x[:, -8:], dim=1, unbiased=False),
                     (x[:, -1] - x[:, -4]) / 3.0 if x.size(1) >= 4 else torch.zeros(x.size(0), device=x.device),
                     torch.mean(x[:, -4:], dim=1) if x.size(1) >= 4 else torch.mean(x, dim=1)],
                    dim=-1,
                )
            logits = self.router(router_features) / max(self.router_temperature, 1e-6)
            weights = torch.softmax(logits, dim=-1)
        blend = weights[:, :1] * trend_pred + weights[:, 1:] * revert_pred
        blend = torch.stack([blend[:, 0], torch.maximum(blend[:, 1], blend[:, 0] + 1e-6)], dim=-1)
        return blend, trend_pred, revert_pred, weights


class BaseTwoStepForecaster:
    def __init__(
        self,
        *,
        context_length: int,
        horizon: int,
        batch_size: int,
        max_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
        calibration_weight: float,
        scheduler_type: str,
        hidden_dim: int = 32,
        scaler_kind: str = "robust",
        enforce_order: bool = False,
        device: torch.device = DEVICE,
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.calibration_weight = calibration_weight
        self.scheduler_type = scheduler_type
        self.hidden_dim = hidden_dim
        self.scaler_kind = scaler_kind
        self.enforce_order = enforce_order
        self.device = device
        self.model: Optional[nn.Module] = None
        self.scaler_: Optional[ScalerState] = None
        self.calibration_: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    def make_model(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self.model(x)

    @staticmethod
    def _extract_predictions(output) -> torch.Tensor:
        return _extract_pred(output)

    def loss_from_output(self, output, target: torch.Tensor) -> torch.Tensor:
        preds = self._extract_predictions(output)
        return torch.mean(torch.abs(preds - target))

    def _make_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int):
        dataset = torch.utils.data.TensorDataset(_as_tensor(x, self.device), _as_tensor(y, self.device))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, generator=generator)

    def _check_gradients(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    def _train_module(
        self,
        module: nn.Module,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        lr: float,
        max_epochs: int,
        seed: int,
        stop_fn: Optional[Callable[[], bool]] = None,
        loss_fn: Optional[Callable[[object, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        train_loader = self._make_loader(train_x, train_y, shuffle=True, seed=seed)
        optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=self.weight_decay)
        scheduler = None
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
        best_state = deepcopy(module.state_dict())
        best_val = float("inf")
        patience = 0
        for _ in range(max_epochs):
            if stop_fn is not None and stop_fn():
                break
            module.train()
            for xb, yb in train_loader:
                if stop_fn is not None and stop_fn():
                    break
                optimizer.zero_grad(set_to_none=True)
                output = module(xb)
                loss = loss_fn(output, yb) if loss_fn is not None else torch.mean(torch.abs(_extract_pred(output) - yb))
                if not _finite_loss(loss):
                    print("FAIL: NaN/divergence detected")
                    raise RuntimeError("NaN/divergence detected")
                loss.backward()
                total_norm = float(clip_grad_norm_(module.parameters(), self.gradient_clip_norm))
                if not np.isfinite(total_norm) or total_norm > 100.0 or not self._check_gradients():
                    print("FAIL: NaN/divergence detected")
                    raise RuntimeError("NaN/divergence detected")
                optimizer.step()
            val_loss = self._evaluate_module(module, val_x, val_y, loss_fn=loss_fn)
            if not np.isfinite(val_loss) or val_loss > 100.0:
                print("FAIL: NaN/divergence detected")
                raise RuntimeError("NaN/divergence detected")
            if val_loss + 1e-12 < best_val:
                best_val = val_loss
                best_state = deepcopy(module.state_dict())
                patience = 0
            else:
                patience += 1
            if scheduler is not None:
                scheduler.step()
            if patience >= self.patience:
                break
        module.load_state_dict(best_state)

    def _evaluate_module(
        self,
        module: nn.Module,
        x: np.ndarray,
        y: np.ndarray,
        *,
        loss_fn: Optional[Callable[[object, torch.Tensor], torch.Tensor]] = None,
    ) -> float:
        if len(x) == 0:
            return float("inf")
        module.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in self._make_loader(x, y, shuffle=False, seed=0):
                output = module(xb)
                loss = loss_fn(output, yb) if loss_fn is not None else torch.mean(torch.abs(_extract_pred(output) - yb))
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("inf")

    def fit(
        self,
        raw_series: np.ndarray,
        *,
        train_end: int,
        val_end: int,
        seed: int,
        max_epochs_override: Optional[int] = None,
        stop_fn: Optional[Callable[[], bool]] = None,
    ):
        raw_series = np.asarray(raw_series, dtype=np.float32)
        self.scaler_ = fit_scaler(raw_series[:train_end], kind=self.scaler_kind)
        scaled = transform_values(raw_series, self.scaler_)
        train_x, train_y = make_window_arrays(scaled, self.context_length, self.horizon, self.context_length, train_end)
        val_x, val_y = make_window_arrays(scaled, self.context_length, self.horizon, train_end, val_end)
        if len(train_x) == 0 or len(val_x) == 0:
            raise ValueError("Insufficient windows for training/validation")
        self.model = self.make_model().to(self.device)
        self._train_module(
            self.model,
            train_x,
            train_y,
            val_x,
            val_y,
            lr=self.learning_rate,
            max_epochs=max_epochs_override or self.max_epochs,
            seed=seed,
            stop_fn=stop_fn,
            loss_fn=lambda output, target: self.loss_from_output(output, target),
        )
        self.calibrate(raw_series, val_end=val_end, train_end=train_end)
        return self

    def _predict_scaled_batch(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        self.model.eval()
        preds = []
        with torch.no_grad():
            zeros = np.zeros((len(x), self.horizon), dtype=np.float32)
            for xb, _ in self._make_loader(x, zeros, shuffle=False, seed=0):
                output = self.forward(xb)
                preds.append(_extract_pred(output).detach().cpu().numpy())
        return np.asarray(np.concatenate(preds, axis=0), dtype=np.float32) if preds else np.zeros((0, self.horizon), dtype=np.float32)

    def calibrate(self, raw_series: np.ndarray, *, val_end: int, train_end: int) -> None:
        if self.calibration_weight <= 0.0:
            self.calibration_ = (1.0, 0.0, 0.0)
            return
        scaled = transform_values(np.asarray(raw_series, dtype=np.float32), self.scaler_)
        val_x, val_y = make_window_arrays(scaled, self.context_length, self.horizon, train_end, val_end)
        if len(val_x) == 0:
            self.calibration_ = (1.0, 0.0, 0.0)
            return
        pred_scaled = self._predict_scaled_batch(val_x)
        pred = inverse_transform_values(pred_scaled, self.scaler_)
        target = inverse_transform_values(val_y, self.scaler_)
        X = np.column_stack([pred.reshape(-1), np.ones(pred.size, dtype=np.float32)])
        coef, *_ = np.linalg.lstsq(X, target.reshape(-1), rcond=None)
        a = float(max(coef[0], 1e-6))
        b = float(coef[1])
        self.calibration_ = (a, b, self.calibration_weight)

    def _apply_calibration(self, pred: np.ndarray) -> np.ndarray:
        a, b, w = self.calibration_
        calibrated = a * pred + b
        blended = (1.0 - w) * pred + w * calibrated
        if self.enforce_order and blended.shape[-1] >= 2:
            blended = blended.copy()
            blended[..., 1] = np.maximum(blended[..., 1], blended[..., 0] + 1e-6)
        return blended

    def predict(self, raw_series: np.ndarray, *, test_start: int) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Model has not been fit")
        raw_series = np.asarray(raw_series, dtype=np.float32)
        if test_start < self.context_length:
            raise ValueError("test_start is too early for the chosen context length")
        scaled = transform_values(raw_series, self.scaler_)
        context = scaled[test_start - self.context_length : test_start]
        output = self.forward(_as_tensor(context[None, :], self.device))
        pred_scaled = self._extract_predictions(output).detach().cpu().numpy()
        pred = inverse_transform_values(pred_scaled, self.scaler_)
        pred = self._apply_calibration(pred)
        return np.asarray(pred[0], dtype=np.float32)


class FoldwiseRobustNLinearTwoHeadForecaster(BaseTwoStepForecaster):
    def __init__(self, **kwargs):
        super().__init__(scaler_kind="robust", enforce_order=False, **kwargs)

    def make_model(self) -> nn.Module:
        return _LinearTwoHeadNet(self.context_length, self.hidden_dim, self.horizon)


class StandardScaledNLinearTwoHeadForecaster(FoldwiseRobustNLinearTwoHeadForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler_kind = "standard"


class OrderAwareNLinearDeltaTwoHeadForecaster(BaseTwoStepForecaster):
    def __init__(
        self,
        *,
        batch_dim: int,
        lambda_order: float,
        order_margin: float,
        **kwargs,
    ):
        super().__init__(scaler_kind="robust", enforce_order=True, **kwargs)
        self.batch_dim = batch_dim
        self.lambda_order = lambda_order
        self.order_margin = order_margin

    def make_model(self) -> nn.Module:
        return _OrderAwareDeltaNet(self.context_length, self.hidden_dim, self.batch_dim)

    def compute_order_penalty(self, preds: torch.Tensor) -> torch.Tensor:
        gap = preds[:, 1] - preds[:, 0]
        return torch.relu(self.order_margin - gap).mean()

    def loss_from_output(self, output, target: torch.Tensor) -> torch.Tensor:
        preds = self._extract_predictions(output)
        mae = torch.mean(torch.abs(preds - target))
        penalty = self.compute_order_penalty(preds)
        return mae + self.lambda_order * penalty


class DirectNLinearTwoHeadForecaster(OrderAwareNLinearDeltaTwoHeadForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enforce_order = False

    def make_model(self) -> nn.Module:
        return _LinearTwoHeadNet(self.context_length, self.hidden_dim, self.horizon)


class PatchTSTTwoStepForecaster(BaseTwoStepForecaster):
    def __init__(
        self,
        *,
        patch_length: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(scaler_kind="robust", enforce_order=False, hidden_dim=d_model, **kwargs)
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

    def make_model(self) -> nn.Module:
        return _PatchTSTNet(
            context_length=self.context_length,
            horizon=self.horizon,
            patch_length=self.patch_length,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )


class ITransformerTwoStepForecaster(BaseTwoStepForecaster):
    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(scaler_kind="robust", enforce_order=False, hidden_dim=d_model, **kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

    def make_model(self) -> nn.Module:
        return _ITransformerNet(
            context_length=self.context_length,
            horizon=self.horizon,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )


class VolatilityGatedNLinearEnsemble(BaseTwoStepForecaster):
    def __init__(
        self,
        *,
        trend_context_length: int,
        router_hidden_dim: int,
        router_temperature: float,
        lambda_balance: float,
        lambda_entropy: float,
        lambda_expert: float,
        learning_rate_router: float,
        **kwargs,
    ):
        super().__init__(scaler_kind="robust", enforce_order=True, **kwargs)
        self.trend_context_length = trend_context_length
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.lambda_balance = lambda_balance
        self.lambda_entropy = lambda_entropy
        self.lambda_expert = lambda_expert
        self.learning_rate_router = learning_rate_router
        self.uniform_router = False

    def make_model(self) -> nn.Module:
        return _GatedEnsembleNet(
            context_length=self.context_length,
            trend_context_length=self.trend_context_length,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            router_hidden_dim=self.router_hidden_dim,
            router_temperature=self.router_temperature,
            uniform_router=self.uniform_router,
        )

    def train_expert(
        self,
        expert: nn.Module,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        lr: float,
        max_epochs: int,
        seed: int,
        stop_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._train_module(
            expert,
            train_x,
            train_y,
            val_x,
            val_y,
            lr=lr,
            max_epochs=max_epochs,
            seed=seed,
            stop_fn=stop_fn,
            loss_fn=lambda output, target: torch.mean(torch.abs(_extract_pred(output) - target)),
        )

    def update_router(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        seed: int,
        max_epochs: int,
        stop_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        if self.uniform_router:
            return
        features_train = np.stack([compute_router_features_from_window(x) for x in train_x], axis=0)
        features_val = np.stack([compute_router_features_from_window(x) for x in val_x], axis=0)
        dataset = torch.utils.data.TensorDataset(
            _as_tensor(train_x, self.device),
            _as_tensor(train_y, self.device),
            _as_tensor(features_train, self.device),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        optimizer = torch.optim.AdamW(self.model.router.parameters(), lr=self.learning_rate_router, weight_decay=self.weight_decay)
        scheduler = None
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
        best_state = deepcopy(self.model.router.state_dict())
        best_val = float("inf")
        patience = 0
        for _ in range(max_epochs):
            if stop_fn is not None and stop_fn():
                break
            self.model.train()
            for xb, yb, fb in loader:
                if stop_fn is not None and stop_fn():
                    break
                optimizer.zero_grad(set_to_none=True)
                blend, trend_pred, revert_pred, weights = self.model(xb, router_features=fb)
                mae = torch.mean(torch.abs(blend - yb))
                mean_weights = weights.mean(dim=0)
                uniform = torch.full_like(mean_weights, 0.5)
                balance = torch.sum(mean_weights * (torch.log(mean_weights + 1e-8) - torch.log(uniform)))
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                expert_loss = torch.mean(torch.abs(trend_pred - yb)) + torch.mean(torch.abs(revert_pred - yb))
                loss = mae + self.lambda_balance * balance - self.lambda_entropy * entropy + self.lambda_expert * expert_loss
                if not _finite_loss(loss):
                    print("FAIL: NaN/divergence detected")
                    raise RuntimeError("NaN/divergence detected")
                loss.backward()
                total_norm = float(clip_grad_norm_(self.model.router.parameters(), self.gradient_clip_norm))
                if not np.isfinite(total_norm) or total_norm > 100.0:
                    print("FAIL: NaN/divergence detected")
                    raise RuntimeError("NaN/divergence detected")
                optimizer.step()
            val_features = torch.as_tensor(features_val, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                blend, trend_pred, revert_pred, weights = self.model(_as_tensor(val_x, self.device), router_features=val_features)
                val_loss = torch.mean(torch.abs(blend - _as_tensor(val_y, self.device))).item()
            if not np.isfinite(val_loss) or val_loss > 100.0:
                print("FAIL: NaN/divergence detected")
                raise RuntimeError("NaN/divergence detected")
            if val_loss + 1e-12 < best_val:
                best_val = val_loss
                best_state = deepcopy(self.model.router.state_dict())
                patience = 0
            else:
                patience += 1
            if scheduler is not None:
                scheduler.step()
            if patience >= self.patience:
                break
        self.model.router.load_state_dict(best_state)

    def fit(
        self,
        raw_series: np.ndarray,
        *,
        train_end: int,
        val_end: int,
        seed: int,
        max_epochs_override: Optional[int] = None,
        stop_fn: Optional[Callable[[], bool]] = None,
    ):
        raw_series = np.asarray(raw_series, dtype=np.float32)
        self.scaler_ = fit_scaler(raw_series[:train_end], kind=self.scaler_kind)
        scaled = transform_values(raw_series, self.scaler_)
        train_x, train_y = make_window_arrays(scaled, self.context_length, self.horizon, self.context_length, train_end)
        val_x, val_y = make_window_arrays(scaled, self.context_length, self.horizon, train_end, val_end)
        if len(train_x) == 0 or len(val_x) == 0:
            raise ValueError("Insufficient windows for training/validation")
        self.model = self.make_model().to(self.device)
        trend_train_x = train_x[:, -self.trend_context_length :]
        trend_val_x = val_x[:, -self.trend_context_length :]
        self.train_expert(
            self.model.trend_expert,
            trend_train_x,
            train_y,
            trend_val_x,
            val_y,
            lr=self.learning_rate,
            max_epochs=max_epochs_override or self.max_epochs,
            seed=seed,
            stop_fn=stop_fn,
        )
        self.train_expert(
            self.model.reversion_expert,
            train_x,
            train_y,
            val_x,
            val_y,
            lr=self.learning_rate,
            max_epochs=max_epochs_override or self.max_epochs,
            seed=seed + 1,
            stop_fn=stop_fn,
        )
        for param in self.model.trend_expert.parameters():
            param.requires_grad = False
        for param in self.model.reversion_expert.parameters():
            param.requires_grad = False
        self.update_router(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=seed,
            max_epochs=max_epochs_override or self.max_epochs,
            stop_fn=stop_fn,
        )
        self.calibrate(raw_series, val_end=val_end, train_end=train_end)
        return self

    def forward(self, x: torch.Tensor):
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self.model(x)

    def predict(self, raw_series: np.ndarray, *, test_start: int) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Model has not been fit")
        raw_series = np.asarray(raw_series, dtype=np.float32)
        scaled = transform_values(raw_series, self.scaler_)
        context = scaled[test_start - self.context_length : test_start]
        output = self.forward(_as_tensor(context[None, :], self.device))
        pred_scaled = self._extract_predictions(output).detach().cpu().numpy()
        pred = inverse_transform_values(pred_scaled, self.scaler_)
        pred = self._apply_calibration(pred)
        return np.asarray(pred[0], dtype=np.float32)


class UniformRouterNLinearEnsemble(VolatilityGatedNLinearEnsemble):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uniform_router = True
