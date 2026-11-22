"""Custom MLflow PyFunc wrapper enforcing per-horizon Split CQR & Serverless requirements."""

from __future__ import annotations

import logging
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS

logger = logging.getLogger(__name__)

TARGET_MODES: tuple[str, ...] = ("delta", "level")


def enforce_cqr_invariants(
    raw_p05: np.ndarray,
    raw_p50: np.ndarray,
    raw_p95: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply two-sided CQR margins and enforce 0 <= p05 <= p50 <= p95 non-crossing.

    Crossing is repaired by sorting the three values of each row. The bounds are then the
    smallest and largest of them, and the reported p50 is the middle one. When the model's
    median lies outside its calibrated bounds, p50 is therefore the nearer bound and not
    the model's own median.
    """
    p05 = np.clip(raw_p05 - q_low, a_min=0.0, a_max=None)
    p95 = np.clip(raw_p95 + q_high, a_min=0.0, a_max=None)
    p50 = np.clip(raw_p50, a_min=0.0, a_max=None)

    stacked = np.sort(np.column_stack([p05, p50, p95]), axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def _offset_key(zone: str, horizon_h: int) -> str:
    """String key (pickle- and JSON-friendly) of one (zone, horizon) calibration cell."""
    return f"{zone}|{int(horizon_h)}"


class ConformalCarbonForecaster(mlflow.pyfunc.PythonModel):
    """Per-zone LightGBM quantile models plus per-(zone, horizon_h) two-sided CQR offsets.

    target_mode "delta" means the base models predict the change from the origin intensity,
    so persistence is the zero prediction; "level" means they predict the intensity itself.
    Offsets are keyed by horizon_h taken from the input row, never by a caller-supplied
    bucket label, and a row without a matching offset is an error, not a silent fallback.
    """

    def __init__(
        self,
        models_by_zone: dict[str, dict[str, Any]],
        conformal_offsets: dict[tuple[str, int], tuple[float, float]],
        feature_cols: tuple[str, ...] = CAUSAL_FEATURE_COLS,
        pip_requirements: tuple[str, ...] | None = None,
        target_mode: str = "delta",
        calibrated_through: str | None = None,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
        self.models_by_zone = models_by_zone
        self.conformal_offsets: dict[str, tuple[float, float]] = {
            _offset_key(zone, horizon_h): (float(lo), float(hi))
            for (zone, horizon_h), (lo, hi) in conformal_offsets.items()
        }
        self.feature_cols = list(feature_cols)
        self.pip_requirements = list(
            pip_requirements or CarbonMLOpsConfig().udf_pip_requirements
        )
        self.target_mode = target_mode
        self.calibrated_through = calibrated_through

    def with_offsets(
        self,
        conformal_offsets: dict[tuple[str, int], tuple[float, float]],
        calibrated_through: str | None = None,
    ) -> ConformalCarbonForecaster:
        """Return a forecaster sharing the fitted base models but carrying new offsets."""
        return ConformalCarbonForecaster(
            models_by_zone=self.models_by_zone,
            conformal_offsets=conformal_offsets,
            feature_cols=tuple(self.feature_cols),
            pip_requirements=tuple(self.pip_requirements),
            target_mode=self.target_mode,
            calibrated_through=calibrated_through,
        )

    def predict_raw_quantiles(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """Return uncalibrated level-scale quantiles (raw_q05, raw_q50, raw_q95) per row."""
        out = pd.DataFrame(
            index=model_input.index, columns=["raw_q05", "raw_q50", "raw_q95"], dtype=float
        )
        for zone, grp in model_input.groupby("zone", sort=False):
            if zone not in self.models_by_zone:
                raise ValueError(f"Unsupported zone in inference input: {zone}")
            z_models = self.models_by_zone[zone]
            x_mat = grp[self.feature_cols]
            base = (
                grp["carbon_intensity_origin"].to_numpy(dtype=float)
                if self.target_mode == "delta"
                else 0.0
            )
            for key in ("q05", "q50", "q95"):
                pred = np.asarray(z_models[key].predict(x_mat), dtype=float)
                out.loc[grp.index, f"raw_{key}"] = pred + base
        return out

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return calibrated [p05, p50, p95] predictions with guaranteed 0 <= p05 <= p50 <= p95."""
        if model_input.empty:
            return pd.DataFrame(columns=["p05", "p50", "p95", "interval_width"])

        raw = self.predict_raw_quantiles(model_input)
        keys = [
            _offset_key(zone, horizon_h)
            for zone, horizon_h in zip(model_input["zone"], model_input["horizon_h"], strict=True)
        ]
        missing = sorted({k for k in keys if k not in self.conformal_offsets})
        if missing:
            raise ValueError(f"No conformal offsets for zone|horizon_h: {missing[:5]}")
        q_low = np.array([self.conformal_offsets[k][0] for k in keys], dtype=float)
        q_high = np.array([self.conformal_offsets[k][1] for k in keys], dtype=float)

        p05, p50, p95 = enforce_cqr_invariants(
            raw["raw_q05"].to_numpy(dtype=float),
            raw["raw_q50"].to_numpy(dtype=float),
            raw["raw_q95"].to_numpy(dtype=float),
            q_low,
            q_high,
        )
        out = pd.DataFrame({"p05": p05, "p50": p50, "p95": p95}, index=model_input.index)
        out["interval_width"] = out["p95"] - out["p05"]
        return out
