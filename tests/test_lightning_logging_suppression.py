import logging
import sys
import types

import pytorch_lightning as pl

from neuralforecast.common._base_model import (
    DistributedConfig,
    _suppress_lightning_info_logs,
)
from tests.dummy.dummy_models import DummyUnivariate


LOGGER_NAMES = (
    "pytorch_lightning",
    "pytorch_lightning.accelerators.cuda",
    "lightning_fabric",
    "lightning_utilities",
)


def _capture_logger_levels() -> dict[str, int]:
    return {name: logging.getLogger(name).level for name in LOGGER_NAMES}


def _set_logger_level(level: int) -> None:
    for name in LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


def test_suppress_lightning_info_logs_temporarily_elevates_cuda_banner_logger() -> None:
    original_levels = _capture_logger_levels()
    try:
        _set_logger_level(logging.INFO)
        info_levels = _capture_logger_levels()

        with _suppress_lightning_info_logs():
            for name in LOGGER_NAMES:
                assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING

        for name, level in info_levels.items():
            assert logging.getLogger(name).level == level
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def test_fit_distributed_applies_lightning_log_suppression(monkeypatch) -> None:
    original_levels = _capture_logger_levels()
    distributor_module_names = (
        "pyspark",
        "pyspark.ml",
        "pyspark.ml.torch",
        "pyspark.ml.torch.distributor",
    )
    previous_modules = {name: sys.modules.get(name) for name in distributor_module_names}
    original_trainer = pl.Trainer

    class FakeTrainer:
        init_levels = None
        fit_levels = None

        def __init__(self, *args, **kwargs):
            FakeTrainer.init_levels = {
                name: logging.getLogger(name).getEffectiveLevel() for name in LOGGER_NAMES
            }
            self.callback_metrics = {}

        def fit(self, model=None, datamodule=None):
            FakeTrainer.fit_levels = {
                name: logging.getLogger(name).getEffectiveLevel() for name in LOGGER_NAMES
            }

    class FakeTorchDistributor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def run(self, fn, **kwargs):
            return fn(**kwargs)

    try:
        _set_logger_level(logging.INFO)
        info_levels = _capture_logger_levels()
        monkeypatch.setattr(pl, "Trainer", FakeTrainer)

        pyspark_module = types.ModuleType("pyspark")
        ml_module = types.ModuleType("pyspark.ml")
        torch_module = types.ModuleType("pyspark.ml.torch")
        distributor_module = types.ModuleType("pyspark.ml.torch.distributor")
        setattr(distributor_module, "TorchDistributor", FakeTorchDistributor)

        sys.modules["pyspark"] = pyspark_module
        sys.modules["pyspark.ml"] = ml_module
        sys.modules["pyspark.ml.torch"] = torch_module
        sys.modules["pyspark.ml.torch.distributor"] = distributor_module

        model = DummyUnivariate(
            h=2,
            input_size=4,
            max_steps=1,
            val_check_steps=1,
            accelerator="gpu",
            devices=2,
            enable_progress_bar=False,
            logger=False,
        )
        for key in ("_max_lr", "_lr_scheduler_cls", "_lr_scheduler_kwargs"):
            model.hparams.pop(key, None)
        distributed_config = DistributedConfig(
            partitions_path="/tmp/distributed-partitions",
            num_nodes=1,
            devices=2,
        )

        model._fit_distributed(
            distributed_config=distributed_config,
            datamodule=object(),
            val_size=1,
            test_size=0,
        )

        assert FakeTrainer.init_levels is not None
        assert FakeTrainer.fit_levels is not None
        for levels in (FakeTrainer.init_levels, FakeTrainer.fit_levels):
            for name in LOGGER_NAMES:
                assert levels[name] == logging.WARNING
        for name, level in info_levels.items():
            assert logging.getLogger(name).level == level
    finally:
        monkeypatch.setattr(pl, "Trainer", original_trainer)
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
