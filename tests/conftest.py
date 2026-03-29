import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["DEVICE"] = "cpu"

import logging

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    yield


@pytest.fixture(scope="session", autouse=True)
def _register_dummy_models():
    """Register DummyUnivariate/DummyMultivariate into MODEL_CLASSES for tests."""
    from tests.dummy.dummy_models import DummyMultivariate, DummyUnivariate

    from runtime_support.forecast_models import MODEL_CLASSES

    MODEL_CLASSES["DummyUnivariate"] = DummyUnivariate
    MODEL_CLASSES["DummyMultivariate"] = DummyMultivariate
    yield
    MODEL_CLASSES.pop("DummyUnivariate", None)
    MODEL_CLASSES.pop("DummyMultivariate", None)
