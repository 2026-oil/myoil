from neuralforecast.common._model_checks import check_model
from neuralforecast.models.deformtime import DeformTime


def test_deformtime_model(suppress_warnings):
    check_model(DeformTime, ["airpassengers"])
