from neuralforecast.common._model_checks import check_model
from neuralforecast.models.deformabletst import DeformableTST


def test_deformabletst_model(suppress_warnings):
    check_model(DeformableTST, ["airpassengers"])
