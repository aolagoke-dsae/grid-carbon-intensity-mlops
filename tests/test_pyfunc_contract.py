"""Unit tests for ConformalCarbonForecaster monotonicity & Serverless requirements (Gate G3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster


class _CrossingStubModel:
    """Stub regressor intentionally producing crossing & negative predictions to test invariants."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.shift, dtype=float)


def test_gate_g3_pyfunc_non_crossing_and_non_negative() -> None:
    models = {
        "DE-LU": {
            "q05": _CrossingStubModel(250.0),
            "q50": _CrossingStubModel(180.0),
            "q95": _CrossingStubModel(-15.0),
        },
        "FR": {
            "q05": _CrossingStubModel(20.0),
            "q50": _CrossingStubModel(30.0),
            "q95": _CrossingStubModel(45.0),
        },
    }
    offsets = {
        ("DE-LU", "short_1_6h"): 12.5,
        ("FR", "short_1_6h"): 4.0,
    }
    forecaster = ConformalCarbonForecaster(models_by_zone=models, conformal_offsets=offsets)

    assert "lightgbm>=4.3.0" in forecaster.pip_requirements
    assert "scipy>=1.13.0" in forecaster.pip_requirements

    sample = pd.DataFrame(
        {
            "zone": ["DE-LU", "FR"],
            "horizon_h": list(range(1, 3)),
            "horizon_bucket": ["short_1_6h", "short_1_6h"],
            **{col: list(range(1, 3)) for col in CAUSAL_FEATURE_COLS if col != "horizon_h"},
        }
    )

    preds = forecaster.predict(context=None, model_input=sample)
    assert list(preds.columns) == ["p05", "p50", "p95", "interval_width"]
    assert (preds["p05"] >= 0.0).all()
    assert (preds["p05"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p95"]).all()
    assert np.allclose(preds["interval_width"], preds["p95"] - preds["p05"])
