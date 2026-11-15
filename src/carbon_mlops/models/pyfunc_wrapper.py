"""Custom MLflow PyFunc wrapper enforcing Asymmetric Split CQR & Serverless requirements."""

from __future__ import annotations

import logging
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS

logger = logging.getLogger(__name__)


def enforce_cqr_invariants(
    raw_p05: np.ndarray,
    raw_p50: np.ndarray,
    raw_p95: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply two-sided CQR margins and enforce 0 <= p05 <= p50 <= p95 non-crossing."""
    p05 = np.clip(raw_p05 - q_low, a_min=0.0, a_max=None)
    p95 = np.clip(raw_p95 + q_high, a_min=0.0, a_max=None)
    p50 = np.clip(raw_p50, a_min=0.0, a_max=None)

    stacked = np.sort(np.column_stack([p05, p50, p95]), axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


class ConformalCarbonForecaster(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapping per-zone LightGBM quantile models + two-sided CQR offsets."""

    def __init__(
        self,
        models_by_zone: dict[str, dict[str, Any]],
        conformal_offsets: dict[tuple[str, str], tuple[float, float] | float],
        feature_cols: tuple[str, ...] = CAUSAL_FEATURE_COLS,
        pip_requirements: tuple[str, ...] | None = None,
    ) -> None:
        cfg = CarbonMLOpsConfig()
        self.models_by_zone = models_by_zone
        normalized: dict[str, tuple[float, float]] = {}
        for (z, b), val in conformal_offsets.items():
            if isinstance(val, (tuple, list)):
                lo, hi = val
                normalized[f"{z}|{b}"] = (float(lo), float(hi))
            else:
                lo, hi = val
                normalized[f"{z}|{b}"] = (float(lo), float(hi))
        self.conformal_offsets = normalized
        self.feature_cols = list(feature_cols)
        self.pip_requirements = list(pip_requirements or cfg.udf_pip_requirements)
        self._cfg = cfg

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return calibrated [p05, p50, p95] predictions with guaranteed 0 <= p05 <= p50 <= p95."""
        if model_input.empty:
            return pd.DataFrame(columns=["p05", "p50", "p95", "interval_width"])

        out = pd.DataFrame(index=model_input.index, columns=["p05", "p50", "p95"], dtype=float)

        for zone, grp in model_input.groupby("zone", sort=False):
            if zone not in self.models_by_zone:
                raise ValueError(f"Unsupported zone in inference input: {zone}")

            z_models = self.models_by_zone[zone]
            x_mat = grp[self.feature_cols]

            raw_p05 = np.asarray(z_models["q05"].predict(x_mat), dtype=float)
            raw_p50 = np.asarray(z_models["q50"].predict(x_mat), dtype=float)
            raw_p95 = np.asarray(z_models["q95"].predict(x_mat), dtype=float)

            buckets = (
                grp["horizon_bucket"]
                if "horizon_bucket" in grp.columns
                else grp["horizon_h"].map(self._cfg.bucket_for_horizon)
            )
            pairs = [self.conformal_offsets.get(f"{zone}|{b}", (0.0, 0.0)) for b in buckets]
            q_low = np.array([lo for lo, _ in pairs], dtype=float)
            q_high = np.array([hi for _, hi in pairs], dtype=float)

            p05, p50, p95 = enforce_cqr_invariants(raw_p05, raw_p50, raw_p95, q_low, q_high)
            out.loc[grp.index, "p05"] = p05
            out.loc[grp.index, "p50"] = p50
            out.loc[grp.index, "p95"] = p95

        out["interval_width"] = out["p95"] - out["p05"]
        return out
