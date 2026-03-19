from neuralforecast.common._model_checks import check_model
from neuralforecast.models.nonstationary_transformer import NonstationaryTransformer


def test_nonstationary_transformer(suppress_warnings):
    check_model(NonstationaryTransformer, ["airpassengers"])
