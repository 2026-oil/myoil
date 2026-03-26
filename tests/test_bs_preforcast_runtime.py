from __future__ import annotations

import bs_preforcast.runtime as bs_runtime
import residual.runtime as residual_runtime


def test_residual_runtime_uses_authoritative_bs_preforcast_runtime_apis() -> None:
    assert residual_runtime.prepare_bs_preforcast_fold_inputs is bs_runtime.prepare_bs_preforcast_fold_inputs
    assert residual_runtime.materialize_bs_preforcast_stage is bs_runtime.materialize_bs_preforcast_stage
