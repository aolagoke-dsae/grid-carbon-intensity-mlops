"""Unit tests for ConformalCarbonForecaster & automated Gate-Val promotion (Gate G3)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS, build_horizon_features
from carbon_mlops.models.evaluate_gate import evaluate_and_promote_champion
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster
from carbon_mlops.models.train import train_split_cqr
from tests.test_no_leakage import _make_synthetic_bronze


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
        ("DE-LU", "short_1_6h"): (12.5, 18.0),
        ("FR", "short_1_6h"): (4.0, 6.0),
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


def test_us22_mlflow_registry_and_champion_promotion_gate(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    cfg = CarbonMLOpsConfig()

    bronze = _make_synthetic_bronze(hours=240)
    feats = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    artifacts = train_split_cqr(
        feats,
        cfg=cfg,
        n_estimators=15,
        register_mlflow=True,
        tracking_uri=tracking_uri,
    )
    assert artifacts.run_id is not None
    assert artifacts.model_version is not None

    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_metric(artifacts.run_id, "gate_val_coverage", 0.905)
    client.log_metric(artifacts.run_id, "gate_val_mean_pinball", 8.50)

    decision = evaluate_and_promote_champion(
        cfg=cfg,
        model_name="conformal_carbon_forecaster",
        client=client,
    )
    assert decision.promoted is True
    champ = client.get_model_version_by_alias("conformal_carbon_forecaster", "champion")
    assert str(champ.version) == str(artifacts.model_version)

    artifacts_bad = train_split_cqr(
        feats,
        cfg=cfg,
        n_estimators=10,
        register_mlflow=True,
        tracking_uri=tracking_uri,
    )
    assert artifacts_bad.run_id is not None
    client.log_metric(artifacts_bad.run_id, "gate_val_coverage", 0.72)
    client.log_metric(artifacts_bad.run_id, "gate_val_mean_pinball", 7.10)

    rejected = evaluate_and_promote_champion(
        cfg=cfg,
        model_name="conformal_carbon_forecaster",
        client=client,
    )
    assert rejected.promoted is False
    champ_after = client.get_model_version_by_alias("conformal_carbon_forecaster", "champion")
    assert str(champ_after.version) == str(artifacts.model_version)
