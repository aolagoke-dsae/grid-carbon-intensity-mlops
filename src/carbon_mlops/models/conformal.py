"""Per-horizon two-sided Split CQR with scheduled (rolling-window) recalibration.

Conformal offsets are not frozen at training time. They are recomputed on a schedule from
the most recent realized forecast errors, because the error distribution of grid carbon
intensity drifts with the season. Each recalibration uses only rows whose target was
already observed at the recalibration time.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.models.pyfunc_wrapper import enforce_cqr_invariants

logger = logging.getLogger(__name__)

OffsetKey = tuple[str, int]
Offsets = dict[OffsetKey, tuple[float, float]]

RAW_QUANTILE_COLS: tuple[str, ...] = ("raw_q05", "raw_q50", "raw_q95")
LOW_SAMPLE_WARNING_THRESHOLD = 100


class RawQuantileModel(Protocol):
    """Anything that can return uncalibrated level-scale quantiles for horizon feature rows."""

    def predict_raw_quantiles(self, model_input: pd.DataFrame) -> pd.DataFrame: ...


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: the ceil((n + 1) * level)-th smallest score."""
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate on an empty set of conformity scores")
    # Taken as an explicit order statistic: np.quantile(..., method="higher") interpolates
    # on (n - 1) and lands one rank too high.
    rank = min(n, int(np.ceil((n + 1) * level - 1e-9)))
    return float(np.partition(np.asarray(scores, dtype=float), rank - 1)[rank - 1])


def add_conformity_scores(model: RawQuantileModel, labelled_df: pd.DataFrame) -> pd.DataFrame:
    """Attach raw quantiles and two-sided CQR scores to rows with a realized target.

    low_score  = raw_q05 - y  (positive when the truth fell below the lower quantile)
    high_score = y - raw_q95  (positive when the truth rose above the upper quantile)
    """
    if labelled_df["target_carbon_intensity"].isna().any():
        raise ValueError("Conformity scores need realized targets on every row")
    out = labelled_df.reset_index(drop=True).copy()
    raw = model.predict_raw_quantiles(out)
    for col in RAW_QUANTILE_COLS:
        out[col] = raw[col].to_numpy()
    y = out["target_carbon_intensity"].to_numpy()
    out["low_score"] = out["raw_q05"].to_numpy() - y
    out["high_score"] = y - out["raw_q95"].to_numpy()
    return out


def select_calibration_window(
    scored_df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """Rows usable for a recalibration at `as_of`: target already observed, origin in window."""
    window_start = as_of - pd.Timedelta(days=window_days)
    usable = (scored_df["target_timestamp"] < as_of) & (
        scored_df["origin_timestamp"] >= window_start
    )
    return scored_df[usable]


def calibrate_offsets(window_df: pd.DataFrame, nominal_coverage: float) -> Offsets:
    """Compute (lower, upper) conformal offsets for every (zone, horizon_h) in the window.

    Each tail is calibrated separately at level 1 - alpha / 2, which keeps the interval
    asymmetric where the errors are and bounds joint miscoverage by alpha.
    """
    tail_level = 1.0 - (1.0 - nominal_coverage) / 2.0
    offsets: Offsets = {}
    smallest = None
    for (zone, horizon_h), grp in window_df.groupby(["zone", "horizon_h"], sort=True):
        n = len(grp)
        smallest = n if smallest is None else min(smallest, n)
        offsets[(str(zone), int(horizon_h))] = (
            conformal_quantile(grp["low_score"].to_numpy(), tail_level),
            conformal_quantile(grp["high_score"].to_numpy(), tail_level),
        )
    if not offsets:
        raise ValueError("Calibration window is empty: no conformity scores to calibrate on")
    if smallest is not None and smallest < LOW_SAMPLE_WARNING_THRESHOLD:
        logger.warning(
            "Smallest calibration group has only %d scores; offsets will be conservative",
            smallest,
        )
    return offsets


def apply_offsets(scored_df: pd.DataFrame, offsets: Offsets) -> pd.DataFrame:
    """Turn raw quantiles into calibrated [p05, p50, p95] using per-(zone, horizon) offsets."""
    keys = list(zip(scored_df["zone"], scored_df["horizon_h"].astype(int), strict=True))
    missing = sorted({k for k in keys if k not in offsets})
    if missing:
        raise ValueError(f"No conformal offsets for (zone, horizon_h): {missing[:5]}")
    q_low = np.array([offsets[k][0] for k in keys], dtype=float)
    q_high = np.array([offsets[k][1] for k in keys], dtype=float)
    p05, p50, p95 = enforce_cqr_invariants(
        scored_df["raw_q05"].to_numpy(dtype=float),
        scored_df["raw_q50"].to_numpy(dtype=float),
        scored_df["raw_q95"].to_numpy(dtype=float),
        q_low,
        q_high,
    )
    out = scored_df.copy()
    out["p05"], out["p50"], out["p95"] = p05, p50, p95
    out["interval_width"] = out["p95"] - out["p05"]
    return out


def walk_forward_calibrated_predictions(
    scored_df: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    cfg: CarbonMLOpsConfig | None = None,
) -> pd.DataFrame:
    """Replay the recalibration schedule over [eval_start, eval_end] and predict out of sample.

    Every `recalibration_step_days` the offsets are recomputed from the trailing
    `recalibration_window_days`, using only targets observed before that moment, and are
    then applied to the origins of the following step. This is how the model is operated
    in production, so the result is an honest estimate of deployed coverage.
    """
    cfg = cfg or CarbonMLOpsConfig()
    step = pd.Timedelta(days=cfg.recalibration_step_days)
    parts: list[pd.DataFrame] = []
    block_start = eval_start
    while block_start <= eval_end:
        block_end = block_start + step
        origin = scored_df["origin_timestamp"]
        block = scored_df[(origin >= block_start) & (origin < block_end) & (origin <= eval_end)]
        if not block.empty:
            window = select_calibration_window(
                scored_df, as_of=block_start, window_days=cfg.recalibration_window_days
            )
            offsets = calibrate_offsets(window, cfg.nominal_coverage)
            calibrated = apply_offsets(block, offsets)
            calibrated["calibrated_as_of"] = block_start
            parts.append(calibrated)
        block_start = block_end
    if not parts:
        raise ValueError("No rows fall inside the walk-forward evaluation period")
    return pd.concat(parts, ignore_index=True)
