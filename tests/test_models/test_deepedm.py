from neuralforecast.common._model_checks import check_model
from neuralforecast.models.deepedm import DeepEDM


def test_deepedm(suppress_warnings):
    check_model(DeepEDM, ["airpassengers"])
