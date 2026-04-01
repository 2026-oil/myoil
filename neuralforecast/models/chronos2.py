from __future__ import annotations

__all__ = ["Chronos2"]

from typing import Any

import fsspec
import json
import numpy as np
import shutil
import subprocess
import torch

from ..losses.pytorch import MAE


def _resolve_device(explicit_device: str | None, accelerator: str | None) -> str:
    if explicit_device is not None:
        if explicit_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return explicit_device
    if accelerator in {"gpu", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Chronos2:
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1,
        val_check_steps: int = 1,
        batch_size: int = 1,
        valid_batch_size: int | None = None,
        windows_batch_size: int = 1,
        inference_windows_batch_size: int | None = None,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
        n_series: int | None = None,
        step_size: int = 1,
        early_stop_patience_steps: int = -1,
        scaler_type: str = "identity",
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
        exclude_insample_y: bool = False,
        random_seed: int | None = 1,
        alias: str | None = None,
        model_id: str = "amazon/chronos-2",
        torch_dtype: str = "float32",
        device: str | None = None,
        inference_batch_size: int = 16,
        limit_prediction_length: bool = True,
        optimizer=None,
        optimizer_kwargs: dict[str, Any] | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        **trainer_kwargs,
    ):
        if loss.outputsize_multiplier > 1:
            raise ValueError(
                "Chronos2 only supports point losses in this inference-only integration."
            )
        if n_series not in (None, 1):
            raise ValueError("Chronos2 currently supports univariate jobs only.")
        if futr_exog_list or hist_exog_list or stat_exog_list:
            raise ValueError(
                "Chronos2 wrapper does not support exogenous variables in this first inference-only cut."
            )
        del (
            max_steps,
            val_check_steps,
            batch_size,
            valid_batch_size,
            windows_batch_size,
            inference_windows_batch_size,
            start_padding_enabled,
            training_data_availability_threshold,
            step_size,
            scaler_type,
            exclude_insample_y,
            optimizer,
            optimizer_kwargs,
            dataloader_kwargs,
        )
        self.h = h
        self.input_size = input_size
        self.loss = loss
        self.valid_loss = loss if valid_loss is None else valid_loss
        self.early_stop_patience_steps = early_stop_patience_steps
        self.random_seed = random_seed
        self.alias = alias
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.limit_prediction_length = limit_prediction_length
        self.accelerator = trainer_kwargs.get("accelerator")
        self.trainer_kwargs = trainer_kwargs
        self.futr_exog_list: list[str] = []
        self.hist_exog_list: list[str] = []
        self.stat_exog_list: list[str] = []
        self.test_size = 0
        self._init_kwargs = {
            "h": h,
            "input_size": input_size,
            "loss": loss,
            "valid_loss": valid_loss,
            "early_stop_patience_steps": early_stop_patience_steps,
            "random_seed": random_seed,
            "alias": alias,
            "model_id": model_id,
            "torch_dtype": torch_dtype,
            "device": device,
            "inference_batch_size": inference_batch_size,
            "limit_prediction_length": limit_prediction_length,
            **trainer_kwargs,
        }

    def __repr__(self) -> str:
        return type(self).__name__ if self.alias is None else self.alias

    def fit(
        self,
        dataset,
        val_size: int = 0,
        test_size: int = 0,
        random_seed: int | None = None,
        distributed_config=None,
    ):
        del dataset, val_size, distributed_config
        if random_seed is not None:
            self.random_seed = random_seed
        self.test_size = test_size
        return self

    @staticmethod
    def _contexts_from_dataset(dataset) -> list[torch.Tensor]:
        contexts: list[torch.Tensor] = []
        target_col = dataset.y_idx
        for start, end in zip(dataset.indptr[:-1], dataset.indptr[1:]):
            series = dataset.temporal[start:end, target_col].detach().cpu()
            series = series[~torch.isnan(series)]
            if series.numel() == 0:
                raise ValueError("Chronos2 received an empty target history for prediction.")
            contexts.append(series.to(torch.float32))
        return contexts

    def _predict_with_subprocess(
        self, contexts: list[torch.Tensor], prediction_length: int
    ) -> list[list[float] | list[list[float]]]:
        if shutil.which("uv") is None:
            raise RuntimeError("Chronos2 inference requires `uv` to be installed and available on PATH.")
        payload = {
            "model_id": self.model_id,
            "torch_dtype": self.torch_dtype,
            "device": _resolve_device(self.device, self.accelerator),
            "inference_batch_size": self.inference_batch_size,
            "limit_prediction_length": self.limit_prediction_length,
            "prediction_length": prediction_length,
            "contexts": [context.tolist() for context in contexts],
        }
        script = """
import json
import sys
import torch
from chronos import Chronos2Pipeline

payload = json.load(sys.stdin)
torch_dtype = getattr(torch, payload["torch_dtype"])
pipeline = Chronos2Pipeline.from_pretrained(
    payload["model_id"],
    device_map=payload["device"],
    torch_dtype=torch_dtype,
)
contexts = [torch.tensor(values, dtype=torch_dtype) for values in payload["contexts"]]
_, mean = pipeline.predict_quantiles(
    inputs=contexts,
    prediction_length=payload["prediction_length"],
    quantile_levels=[0.5],
    batch_size=payload["inference_batch_size"],
    limit_prediction_length=payload["limit_prediction_length"],
)
json.dump([torch.as_tensor(item, dtype=torch.float32).cpu().tolist() for item in mean], sys.stdout)
"""
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "chronos-forecasting>=2.0,<3",
                "python",
                "-c",
                script,
            ],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Chronos2 subprocess inference failed: "
                + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
            )
        return json.loads(completed.stdout)

    def predict(
        self,
        dataset,
        test_size: int | None = None,
        step_size: int = 1,
        random_seed: int | None = None,
        quantiles=None,
        h: int | None = None,
        explainer_config=None,
        **data_module_kwargs,
    ) -> np.ndarray:
        del test_size, step_size, random_seed, explainer_config, data_module_kwargs
        if quantiles is not None:
            raise ValueError("Chronos2 quantile prediction is not exposed through this runtime wrapper yet.")
        prediction_length = self.h if h is None else h
        contexts = self._contexts_from_dataset(dataset)
        mean = self._predict_with_subprocess(contexts, prediction_length)
        outputs: list[np.ndarray] = []
        for forecast in mean:
            point = torch.as_tensor(forecast, dtype=torch.float32).detach().cpu()
            if point.ndim == 2:
                if point.shape[0] != 1:
                    raise ValueError(
                        "Chronos2 wrapper expected univariate output per job but received multivariate predictions."
                    )
                point = point.squeeze(0)
            outputs.append(point.reshape(-1, 1).numpy())
        return np.concatenate(outputs, axis=0)

    def get_test_size(self) -> int:
        return self.test_size

    def set_test_size(self, test_size: int) -> None:
        self.test_size = test_size

    def save(self, path: str) -> None:
        with fsspec.open(path, "wb") as f:
            torch.save({"init_kwargs": self._init_kwargs}, f)

    @classmethod
    def load(cls, path: str, **kwargs):
        if "weights_only" in __import__("inspect").signature(torch.load).parameters:
            kwargs["weights_only"] = False
        with fsspec.open(path, "rb") as f:
            payload = torch.load(f, **kwargs)
        return cls(**payload["init_kwargs"])
