"""Calibration summary and H1-H3 tests on walk-forward predictions.

Implements docs/ANALYSIS_PLAN_H1_H3.md. Every function takes a table of calibrated
walk-forward predictions, so the same code runs on Gate-Val (for development) and on the
Locked Test (once). Nothing here trains, tunes or recalibrates a model.

Uncertainty is quantified with a moving-block bootstrap over calendar days, because hourly
forecasts are autocorrelated and each target hour appears at up to 24 lead times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS
from carbon_mlops.models.conformal import RawQuantileModel

logger = logging.getLogger(__name__)

ALPHA = 0.01
BOOTSTRAP_SEED = 20261115
BLOCK_DAYS = 7
N_RESAMPLES = 10_000
N_PERMUTATION_REPEATS = 20
N_SINGLE_FEATURE_REPEATS = 5
LOCAL_TIMEZONE = "Europe/Berlin"
MIDDAY_LOCAL_HOURS: tuple[int, ...] = (11, 12, 13, 14)
SHORT_BUCKET = "short_1_6h"
DAY_AHEAD_BUCKET = "day_ahead_13_24h"

ORIGIN_STATE_FEATURES: tuple[str, ...] = (
    "carbon_intensity_origin",
    "ci_lag_1h",
    "ci_lag_3h",
    "ci_lag_6h",
    "ci_lag_24h",
    "ci_roll24_mean",
    "ci_roll24_std",
    "wind_share_origin",
    "solar_share_origin",
    "renewable_share_origin",
    "nuclear_share_origin",
    "fossil_share_origin",
    "gen_total_mwh_origin",
)
CALENDAR_FEATURES: tuple[str, ...] = (
    "target_hour_sin",
    "target_hour_cos",
    "target_dow",
    "target_month",
    "is_weekend",
)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "origin_state": ORIGIN_STATE_FEATURES,
    "calendar": CALENDAR_FEATURES,
}

RESULT_COLUMNS: tuple[str, ...] = (
    "hypothesis",
    "part",
    "analysis",
    "zone",
    "quantity",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "n",
    "supported",
    "note",
)


@dataclass(frozen=True)
class AnalysisOutputs:
    """Tables produced by one run of the analysis."""

    calibration: pd.DataFrame
    results: pd.DataFrame


# --------------------------------------------------------------------------------------
# Moving-block bootstrap over days
# --------------------------------------------------------------------------------------


def moving_block_day_samples(
    n_days: int,
    n_resamples: int = N_RESAMPLES,
    block_days: int = BLOCK_DAYS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Return an (n_resamples, n_days) array of day positions built from consecutive blocks."""
    if n_days <= 0:
        raise ValueError("Need at least one day to bootstrap")
    block = min(block_days, n_days)
    n_blocks = int(np.ceil(n_days / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days - block + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    samples = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)
    return samples[:, :n_days]


def day_positions(timestamps: pd.Series, first_day: pd.Timestamp) -> np.ndarray:
    """Position of each timestamp's UTC calendar day, counted from `first_day`."""
    days = pd.to_datetime(timestamps).dt.floor("D")
    return ((days - first_day) / pd.Timedelta(days=1)).round().astype(int).to_numpy()


def _sum_by_day(values: np.ndarray, positions: np.ndarray, n_days: int) -> np.ndarray:
    return np.bincount(positions, weights=values.astype(float), minlength=n_days)


def bootstrap_ratio(
    numerator_by_day: np.ndarray,
    denominator_by_day: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Resampled ratio of sums, e.g. covered rows over all rows. NaN where the sum is 0."""
    num = numerator_by_day[samples].sum(axis=1)
    den = denominator_by_day[samples].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def percentile_interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval of bootstrap draws, ignoring undefined draws."""
    tail = (1.0 - level) / 2.0
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if np.isinf(finite).any():
        # Interpolating between infinite draws is undefined; take order statistics instead.
        low = np.quantile(finite, tail, method="lower")
        high = np.quantile(finite, 1.0 - tail, method="higher")
        return float(low), float(high)
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def _row(**kwargs: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(RESULT_COLUMNS)
    row.update(kwargs)
    return row


# --------------------------------------------------------------------------------------
# Section 4: headline calibration
# --------------------------------------------------------------------------------------


def attach_target_renewable_share(preds_df: pd.DataFrame, bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Add the realized renewable share at each row's target hour."""
    lookup = bronze_df[["zone", "timestamp_utc", "renewable_share"]].rename(
        columns={"timestamp_utc": "target_timestamp", "renewable_share": "renewable_share_target"}
    )
    lookup["target_timestamp"] = pd.to_datetime(lookup["target_timestamp"])
    return preds_df.merge(lookup, on=["zone", "target_timestamp"], how="left")


def _pinball_rows(y: np.ndarray, pred: np.ndarray, alpha: float) -> np.ndarray:
    diff = y - pred
    return np.maximum(alpha * diff, (alpha - 1.0) * diff)


def calibration_table(
    preds_df: pd.DataFrame,
    bronze_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Coverage (with block-bootstrap interval), miss direction, width and pinball losses.

    One row per zone and breakdown level: overall, horizon bucket, lead time, renewable-share
    quartile at the target hour, calendar month and target hour of day.
    """
    df = attach_target_renewable_share(preds_df, bronze_df)
    first_day = df["origin_timestamp"].dt.floor("D").min()
    df["day_pos"] = day_positions(df["origin_timestamp"], first_day)
    n_days = int(df["day_pos"].max()) + 1
    samples = moving_block_day_samples(n_days, n_resamples=n_resamples)

    y = df["target_carbon_intensity"].to_numpy()
    df["covered"] = (y >= df["p05"].to_numpy()) & (y <= df["p95"].to_numpy())
    df["below"] = y < df["p05"].to_numpy()
    df["above"] = y > df["p95"].to_numpy()
    df["pin_p05"] = _pinball_rows(y, df["p05"].to_numpy(), 0.05)
    df["pin_p50"] = _pinball_rows(y, df["p50"].to_numpy(), 0.50)
    df["pin_p95"] = _pinball_rows(y, df["p95"].to_numpy(), 0.95)
    df["pin_persistence"] = _pinball_rows(y, df["carbon_intensity_origin"].to_numpy(), 0.50)
    df["month"] = df["target_timestamp"].dt.strftime("%Y-%m")
    df["target_hour_utc"] = df["target_timestamp"].dt.hour

    rows: list[dict[str, Any]] = []
    for zone in cfg.zones:
        zdf = df[df["zone"] == zone].copy()
        if zdf.empty:
            continue
        hours = zdf.drop_duplicates("target_timestamp").set_index("target_timestamp")
        quartile_of_hour = pd.qcut(
            hours["renewable_share_target"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        zdf["renewable_quartile"] = zdf["target_timestamp"].map(quartile_of_hour).astype(str)
        zdf["all"] = "all"

        breakdowns = (
            "all",
            "horizon_bucket",
            "horizon_h",
            "renewable_quartile",
            "month",
            "target_hour_utc",
        )
        for breakdown in breakdowns:
            for level, grp in zdf.groupby(breakdown, sort=True):
                pos = grp["day_pos"].to_numpy()
                covered_by_day = _sum_by_day(grp["covered"].to_numpy(), pos, n_days)
                rows_by_day = _sum_by_day(np.ones(len(grp)), pos, n_days)
                low, high = percentile_interval(
                    bootstrap_ratio(covered_by_day, rows_by_day, samples)
                )
                rows.append(
                    {
                        "zone": zone,
                        "breakdown": breakdown,
                        "level": str(level),
                        "n_rows": len(grp),
                        "coverage": float(grp["covered"].mean()),
                        "coverage_ci_low": low,
                        "coverage_ci_high": high,
                        "share_below_p05": float(grp["below"].mean()),
                        "share_above_p95": float(grp["above"].mean()),
                        "mean_width": float(grp["interval_width"].mean()),
                        "pinball_p05": float(grp["pin_p05"].mean()),
                        "pinball_p50": float(grp["pin_p50"].mean()),
                        "pinball_p95": float(grp["pin_p95"].mean()),
                        "persistence_pinball_p50": float(grp["pin_persistence"].mean()),
                    }
                )
    table = pd.DataFrame(rows)
    low_bound, high_bound = cfg.coverage_bounds
    table["inside_reference_band"] = table["coverage"].between(low_bound, high_bound)
    return table


# --------------------------------------------------------------------------------------
# H1: renewable penetration and volatility (DE-LU)
# --------------------------------------------------------------------------------------


def per_target_hour_table(zone_preds: pd.DataFrame) -> pd.DataFrame:
    """Collapse a zone's forecast rows to one row per target hour (mean over lead times)."""
    y = zone_preds["target_carbon_intensity"].to_numpy()
    work = zone_preds.assign(
        abs_error=np.abs(y - zone_preds["p50"].to_numpy()),
        covered=(y >= zone_preds["p05"].to_numpy()) & (y <= zone_preds["p95"].to_numpy()),
    )
    return (
        work.groupby("target_timestamp", sort=True)
        .agg(
            realized=("target_carbon_intensity", "first"),
            p50=("p50", "mean"),
            width=("interval_width", "mean"),
            abs_error=("abs_error", "mean"),
            coverage=("covered", "mean"),
            renewable_share=("renewable_share_target", "first"),
            n_leads=("horizon_h", "size"),
        )
        .reset_index()
    )


def _values_by_day(values: np.ndarray, positions: np.ndarray, n_days: int) -> np.ndarray:
    """Pack per-day values into an (n_days, max_per_day) matrix padded with NaN."""
    order = np.argsort(positions, kind="stable")
    values, positions = values[order], positions[order]
    counts = np.bincount(positions, minlength=n_days)
    matrix = np.full((n_days, max(int(counts.max()), 1)), np.nan)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    column = np.arange(len(values)) - starts[positions]
    matrix[positions, column] = values
    return matrix


def _bootstrap_median_difference(
    top: np.ndarray, bottom: np.ndarray, samples: np.ndarray, chunk: int = 500
) -> np.ndarray:
    """Resampled (median of top) - (median of bottom) from per-day value matrices."""
    draws = np.empty(samples.shape[0])
    for start in range(0, samples.shape[0], chunk):
        idx = samples[start : start + chunk]
        top_med = np.nanmedian(top[idx].reshape(idx.shape[0], -1), axis=1)
        bottom_med = np.nanmedian(bottom[idx].reshape(idx.shape[0], -1), axis=1)
        draws[start : start + chunk] = top_med - bottom_med
    return draws


def evaluate_h1(
    preds_df: pd.DataFrame,
    bronze_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
    train_thresholds: tuple[float, float] | None = None,
    zone: str = "DE-LU",
    n_resamples: int = N_RESAMPLES,
) -> list[dict[str, Any]]:
    """H1: top renewable-share quartile has lower intensity and a wider interval than bottom."""
    df = attach_target_renewable_share(preds_df[preds_df["zone"] == zone], bronze_df)
    hours = per_target_hour_table(df).dropna(subset=["renewable_share"])
    first_day = hours["target_timestamp"].dt.floor("D").min()
    hours["day_pos"] = day_positions(hours["target_timestamp"], first_day)
    n_days = int(hours["day_pos"].max()) + 1
    samples = moving_block_day_samples(n_days, n_resamples=n_resamples)
    rows: list[dict[str, Any]] = []

    def run(q25: float, q75: float, label: str, decisive: bool) -> None:
        top = hours[hours["renewable_share"] >= q75]
        bottom = hours[hours["renewable_share"] <= q25]
        n = f"{len(top)}/{len(bottom)}"
        note = f"thresholds Q25={q25:.4f} Q75={q75:.4f} ({label}); unit: target hour"

        if decisive:
            mwu_p50 = stats.mannwhitneyu(top["p50"], bottom["p50"], alternative="less")
            mwu_w = stats.mannwhitneyu(top["width"], bottom["width"], alternative="greater")
            rows.append(
                _row(
                    hypothesis="H1",
                    part="a",
                    analysis="registered",
                    zone=zone,
                    quantity="Mann-Whitney U, p50 top < bottom",
                    estimate=float(top["p50"].median() - bottom["p50"].median()),
                    p_value=float(mwu_p50.pvalue),
                    n=n,
                    supported=bool(mwu_p50.pvalue < ALPHA),
                    note=note + "; hourly values treated as independent",
                )
            )
            rows.append(
                _row(
                    hypothesis="H1",
                    part="b",
                    analysis="registered",
                    zone=zone,
                    quantity="Mann-Whitney U, width top > bottom",
                    estimate=float(top["width"].mean() - bottom["width"].mean()),
                    p_value=float(mwu_w.pvalue),
                    n=n,
                    supported=bool(mwu_w.pvalue < ALPHA),
                    note=note + "; hourly values treated as independent",
                )
            )

        median_draws = _bootstrap_median_difference(
            _values_by_day(top["realized"].to_numpy(), top["day_pos"].to_numpy(), n_days),
            _values_by_day(bottom["realized"].to_numpy(), bottom["day_pos"].to_numpy(), n_days),
            samples,
        )
        upper_99 = float(np.nanquantile(median_draws, 1.0 - ALPHA))
        low, high = percentile_interval(median_draws)
        analysis = "corrected" if decisive else "sensitivity"
        rows.append(
            _row(
                hypothesis="H1",
                part="a",
                analysis=analysis,
                zone=zone,
                quantity="median realized intensity, top - bottom",
                estimate=float(top["realized"].median() - bottom["realized"].median()),
                ci_low=low,
                ci_high=high,
                n=n,
                supported=bool(upper_99 < 0.0) if decisive else None,
                note=note + f"; one-sided 99% upper bound={upper_99:.3f}",
            )
        )

        def mean_draws(column: str) -> np.ndarray:
            t_sum = _sum_by_day(top[column].to_numpy(), top["day_pos"].to_numpy(), n_days)
            t_n = _sum_by_day(np.ones(len(top)), top["day_pos"].to_numpy(), n_days)
            b_sum = _sum_by_day(bottom[column].to_numpy(), bottom["day_pos"].to_numpy(), n_days)
            b_n = _sum_by_day(np.ones(len(bottom)), bottom["day_pos"].to_numpy(), n_days)
            return bootstrap_ratio(t_sum, t_n, samples) - bootstrap_ratio(b_sum, b_n, samples)

        width_draws = mean_draws("width")
        lower_99 = float(np.nanquantile(width_draws, ALPHA))
        low, high = percentile_interval(width_draws)
        rows.append(
            _row(
                hypothesis="H1",
                part="b",
                analysis=analysis,
                zone=zone,
                quantity="mean interval width, top - bottom",
                estimate=float(top["width"].mean() - bottom["width"].mean()),
                ci_low=low,
                ci_high=high,
                n=n,
                supported=bool(lower_99 > 0.0) if decisive else None,
                note=note + f"; one-sided 99% lower bound={lower_99:.3f}",
            )
        )

        if decisive:
            relative = hours.assign(relative_width=hours["width"] / hours["p50"])
            for name, group in (("top", top), ("bottom", bottom)):
                rel = relative.loc[group.index, "relative_width"]
                for quantity, value in (
                    ("relative width W/p50", float(rel.mean())),
                    ("mean absolute error", float(group["abs_error"].mean())),
                    ("coverage", float(group["coverage"].mean())),
                    ("mean width", float(group["width"].mean())),
                ):
                    rows.append(
                        _row(
                            hypothesis="H1",
                            part="context",
                            analysis="context",
                            zone=zone,
                            quantity=f"{quantity}, {name} quartile",
                            estimate=value,
                            n=str(len(group)),
                        )
                    )

    q25, q75 = hours["renewable_share"].quantile([0.25, 0.75])
    run(float(q25), float(q75), "evaluation period", decisive=True)
    if train_thresholds is not None:
        run(train_thresholds[0], train_thresholds[1], "Train period", decisive=False)

    by_bucket = df.assign(
        regime=np.where(
            df["renewable_share_target"] >= q75,
            "top",
            np.where(df["renewable_share_target"] <= q25, "bottom", "mid"),
        )
    )
    for bucket, grp in by_bucket.groupby("horizon_bucket", sort=True):
        top_w = grp.loc[grp["regime"] == "top", "interval_width"].mean()
        bottom_w = grp.loc[grp["regime"] == "bottom", "interval_width"].mean()
        rows.append(
            _row(
                hypothesis="H1",
                part="context",
                analysis="context",
                zone=zone,
                quantity=f"mean width top - bottom, {bucket}",
                estimate=float(top_w - bottom_w),
                n=str(len(grp)),
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# H2: midday parity, DE-LU versus FR (descriptive)
# --------------------------------------------------------------------------------------


def midday_daily_table(
    bronze_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Daily midday (local 11:00-15:00) mean intensity per zone and DE-LU solar+wind share.

    A day is kept only if all four local hours are present and trusted in both zones.
    """
    df = bronze_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    start_utc = start if start.tzinfo else start.tz_localize("UTC")
    end_utc = end if end.tzinfo else end.tz_localize("UTC")
    df = df[(df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] <= end_utc)]
    if "is_partial" in df.columns:
        df = df[~df["is_partial"].astype(bool)]
    if "hours_covered" in df.columns:
        df = df[df["hours_covered"] >= 1.0 - 1e-6]

    local = df["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    df = df.assign(local_date=local.dt.date, local_hour=local.dt.hour)
    df = df[df["local_hour"].isin(MIDDAY_LOCAL_HOURS)]
    df = df.assign(vre_share=df["wind_share"] + df["solar_share"])

    daily = df.groupby(["local_date", "zone"]).agg(
        intensity=("carbon_intensity_g_kwh", "mean"),
        vre_share=("vre_share", "mean"),
        n_hours=("local_hour", "nunique"),
    )
    wide = daily.unstack("zone")
    complete = (wide["n_hours"] == len(MIDDAY_LOCAL_HOURS)).all(axis=1)
    wide = wide[complete]
    return pd.DataFrame(
        {
            "de_lu": wide[("intensity", "DE-LU")],
            "fr": wide[("intensity", "FR")],
            "de_lu_vre_share": wide[("vre_share", "DE-LU")],
        }
    ).reset_index()


def evaluate_h2(
    bronze_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prior_year_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """H2 (descriptive): on high solar+wind days, is DE-LU midday intensity below FR?"""
    daily = midday_daily_table(bronze_df, start, end)
    rows: list[dict[str, Any]] = []

    def run(threshold: float, label: str, analysis: str) -> None:
        qualifying = daily[daily["de_lu_vre_share"] > threshold]
        gap = qualifying["de_lu"] - qualifying["fr"]
        if len(gap) >= 1 and (gap != 0).any():
            p_value = float(stats.wilcoxon(gap, alternative="less").pvalue)
        else:
            p_value = float("nan")
        rows.append(
            _row(
                hypothesis="H2",
                part="",
                analysis=analysis,
                zone="DE-LU vs FR",
                quantity="paired Wilcoxon, daily midday mean DE-LU < FR",
                estimate=float(gap.mean()) if len(gap) else float("nan"),
                p_value=p_value,
                n=f"{len(qualifying)} of {len(daily)} days",
                supported=bool(p_value < ALPHA) if analysis == "registered" else None,
                note=(
                    f"threshold={threshold:.4f} ({label}); mean DE-LU="
                    f"{qualifying['de_lu'].mean():.1f}, mean FR={qualifying['fr'].mean():.1f}"
                ),
            )
        )
        if analysis == "registered":
            rows.append(
                _row(
                    hypothesis="H2",
                    part="",
                    analysis="corrected",
                    zone="DE-LU vs FR",
                    quantity="days with DE-LU < FR among qualifying days",
                    estimate=float((gap < 0).sum()),
                    n=f"{len(qualifying)} of {len(daily)} days",
                    supported=bool(p_value < ALPHA),
                    note=f"smallest daily gap DE-LU - FR = {gap.min():.1f} gCO2eq/kWh; descriptive",
                )
            )

    run(float(daily["de_lu_vre_share"].median()), "evaluation-period median", "registered")
    if prior_year_threshold is not None:
        run(prior_year_threshold, "same dates one year earlier", "sensitivity")
    return rows


# --------------------------------------------------------------------------------------
# H3a: importance reversal
# --------------------------------------------------------------------------------------


def permutation_loss_increase(
    model: RawQuantileModel,
    preds_df: pd.DataFrame,
    groups: dict[str, tuple[str, ...]],
    n_repeats: int,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-row increase in median pinball loss when a feature group is shuffled between origins.

    Within a zone every row takes the group's values from a donor origin at the same lead
    time, and all lead times of an origin share one donor. Only origins with every lead time
    are used. Returns the input rows (complete origins only) with one `inc_<group>` column
    per group, averaged over repeats.
    """
    covered = set().union(*groups.values())
    unknown = covered - set(CAUSAL_FEATURE_COLS)
    if unknown or "horizon_h" in covered:
        raise ValueError(f"Invalid permutation groups: {sorted(unknown)} / horizon_h is fixed")

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    n_leads = preds_df["horizon_h"].nunique()
    for _, zdf in preds_df.groupby("zone", sort=True):
        size = zdf.groupby("origin_timestamp")["horizon_h"].transform("size")
        zdf = zdf[size == n_leads].sort_values(["origin_timestamp", "horizon_h"])
        zdf = zdf.reset_index(drop=True)
        n_origins = len(zdf) // n_leads
        if n_origins < 2:
            continue
        y = zdf["target_carbon_intensity"].to_numpy()

        def loss(frame: pd.DataFrame, truth: np.ndarray = y) -> np.ndarray:
            median = np.clip(model.predict_raw_quantiles(frame)["raw_q50"].to_numpy(), 0.0, None)
            return _pinball_rows(truth, median, 0.50)

        base_loss = loss(zdf)
        out = zdf.copy()
        for name, columns in groups.items():
            cols = list(columns)
            block = zdf[cols].to_numpy(dtype=float).reshape(n_origins, n_leads, len(cols))
            increase = np.zeros(len(zdf))
            for _ in range(n_repeats):
                donor = rng.permutation(n_origins)
                shuffled = zdf.copy()
                shuffled[cols] = block[donor].reshape(len(zdf), len(cols))
                increase += loss(shuffled) - base_loss
            out[f"inc_{name}"] = increase / n_repeats
        parts.append(out)
    if not parts:
        raise ValueError("No zone has at least two origins with every lead time")
    return pd.concat(parts, ignore_index=True)


def evaluate_h3a(
    model: RawQuantileModel,
    preds_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
    n_resamples: int = N_RESAMPLES,
    n_repeats: int = N_PERMUTATION_REPEATS,
    n_single_repeats: int = N_SINGLE_FEATURE_REPEATS,
) -> list[dict[str, Any]]:
    """H3a: origin-state importance dominates at 1-6h and is overtaken by calendar at 13-24h."""
    grouped = permutation_loss_increase(model, preds_df, FEATURE_GROUPS, n_repeats)
    singles = permutation_loss_increase(
        model,
        preds_df,
        {f: (f,) for f in (*ORIGIN_STATE_FEATURES, *CALENDAR_FEATURES)},
        n_single_repeats,
        seed=BOOTSTRAP_SEED + 1,
    )
    first_day = grouped["origin_timestamp"].dt.floor("D").min()
    grouped["day_pos"] = day_positions(grouped["origin_timestamp"], first_day)
    n_days = int(grouped["day_pos"].max()) + 1
    samples = moving_block_day_samples(n_days, n_resamples=n_resamples)

    rows: list[dict[str, Any]] = []
    for zone in cfg.zones:
        intervals: dict[str, tuple[float, float]] = {}
        registered_ratio: dict[str, float] = {}
        for bucket in (SHORT_BUCKET, DAY_AHEAD_BUCKET):
            cell = grouped[(grouped["zone"] == zone) & (grouped["horizon_bucket"] == bucket)]
            if cell.empty:
                continue
            pos = cell["day_pos"].to_numpy()
            origin_by_day = _sum_by_day(cell["inc_origin_state"].to_numpy(), pos, n_days)
            calendar_by_day = _sum_by_day(cell["inc_calendar"].to_numpy(), pos, n_days)
            num = origin_by_day[samples].sum(axis=1)
            den = calendar_by_day[samples].sum(axis=1)
            draws = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.inf)
            low, high = percentile_interval(draws)
            intervals[bucket] = (low, high)
            total_calendar = float(cell["inc_calendar"].sum())
            estimate = (
                float(cell["inc_origin_state"].sum()) / total_calendar
                if total_calendar > 0
                else float("inf")
            )
            rows.append(
                _row(
                    hypothesis="H3",
                    part="a",
                    analysis="corrected",
                    zone=zone,
                    quantity=f"importance ratio origin-state / calendar, {bucket}",
                    estimate=estimate,
                    ci_low=low,
                    ci_high=high,
                    n=str(len(cell)),
                    note=(
                        f"mean loss increase: origin-state={cell['inc_origin_state'].mean():.4f},"
                        f" calendar={cell['inc_calendar'].mean():.4f}"
                    ),
                )
            )

            single = singles[(singles["zone"] == zone) & (singles["horizon_bucket"] == bucket)]
            origin_sum = sum(float(single[f"inc_{f}"].mean()) for f in ORIGIN_STATE_FEATURES)
            calendar_sum = sum(float(single[f"inc_{f}"].mean()) for f in CALENDAR_FEATURES)
            registered_ratio[bucket] = (
                origin_sum / calendar_sum if calendar_sum > 0 else float("inf")
            )
            ranked = sorted(
                (
                    (float(single[f"inc_{f}"].mean()), f)
                    for f in (*ORIGIN_STATE_FEATURES, *CALENDAR_FEATURES)
                ),
                reverse=True,
            )[:3]
            rows.append(
                _row(
                    hypothesis="H3",
                    part="a",
                    analysis="registered",
                    zone=zone,
                    quantity=f"single-feature importance ratio, {bucket}",
                    estimate=registered_ratio[bucket],
                    n=str(len(single)),
                    note="top features: "
                    + ", ".join(f"{name}={value:.3f}" for value, name in ranked),
                )
            )

        if SHORT_BUCKET in intervals and DAY_AHEAD_BUCKET in intervals:
            corrected = bool(
                intervals[SHORT_BUCKET][0] > 1.0 and intervals[DAY_AHEAD_BUCKET][1] < 1.0
            )
            registered = bool(
                registered_ratio[SHORT_BUCKET] > 1.0 and registered_ratio[DAY_AHEAD_BUCKET] < 1.0
            )
            rows.append(
                _row(
                    hypothesis="H3",
                    part="a",
                    analysis="corrected",
                    zone=zone,
                    quantity="reversal: ratio interval above 1 at 1-6h and below 1 at 13-24h",
                    supported=corrected,
                )
            )
            rows.append(
                _row(
                    hypothesis="H3",
                    part="a",
                    analysis="registered",
                    zone=zone,
                    quantity="reversal: point ratio above 1 at 1-6h and below 1 at 13-24h",
                    supported=registered,
                    note="approximation: the registered per-horizon models do not exist",
                )
            )
    return rows


# --------------------------------------------------------------------------------------
# H3b: growth of uncertainty with lead time
# --------------------------------------------------------------------------------------


def daily_lead_time_differences(preds_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Per zone, day and origin: mean of `value_column` at 13-24h minus mean at 1-6h.

    Only origins with every lead time are used. Returns one row per (zone, origin).
    """
    n_leads = preds_df["horizon_h"].nunique()
    size = preds_df.groupby(["zone", "origin_timestamp"])["horizon_h"].transform("size")
    full = preds_df[size == n_leads]
    wide = full.pivot_table(
        index=["zone", "origin_timestamp"],
        columns="horizon_bucket",
        values=value_column,
        aggfunc="mean",
    )
    out = (wide[DAY_AHEAD_BUCKET] - wide[SHORT_BUCKET]).rename("difference").reset_index()
    out["day"] = out["origin_timestamp"].dt.floor("D")
    out["origin_hour_utc"] = out["origin_timestamp"].dt.hour
    return out


def _wilcoxon_greater(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0 or not (values != 0).any():
        return float("nan")
    return float(stats.wilcoxon(values, alternative="greater").pvalue)


def evaluate_h3b(preds_df: pd.DataFrame, cfg: CarbonMLOpsConfig) -> list[dict[str, Any]]:
    """H3b: uncertainty is larger at 13-24h than at 1-6h."""
    work = preds_df.assign(
        abs_error=np.abs(
            preds_df["target_carbon_intensity"].to_numpy() - preds_df["p50"].to_numpy()
        )
    )
    rows: list[dict[str, Any]] = []

    width = daily_lead_time_differences(work, "interval_width")
    midnight = width[width["origin_hour_utc"] == 0]
    pooled_p = _wilcoxon_greater(midnight["difference"])
    rows.append(
        _row(
            hypothesis="H3",
            part="b",
            analysis="registered",
            zone="pooled",
            quantity="paired Wilcoxon, width 13-24h > 1-6h, 00:00 UTC origins",
            estimate=float(midnight["difference"].mean()),
            p_value=pooled_p,
            n=str(len(midnight)),
            supported=bool(pooled_p < ALPHA),
            note="zones pooled as registered; pairs from the two zones are not independent",
        )
    )
    for zone in cfg.zones:
        z = midnight[midnight["zone"] == zone]
        rows.append(
            _row(
                hypothesis="H3",
                part="b",
                analysis="registered",
                zone=zone,
                quantity="paired Wilcoxon, width 13-24h > 1-6h, 00:00 UTC origins",
                estimate=float(z["difference"].mean()),
                p_value=_wilcoxon_greater(z["difference"]),
                n=str(len(z)),
                note=f"share of days positive={float((z['difference'] > 0).mean()):.3f}",
            )
        )

    error = daily_lead_time_differences(work, "abs_error")
    daily = error.groupby(["zone", "day"])["difference"].mean().reset_index()
    for zone in cfg.zones:
        z = daily[daily["zone"] == zone]
        p_value = _wilcoxon_greater(z["difference"])
        rows.append(
            _row(
                hypothesis="H3",
                part="b",
                analysis="corrected",
                zone=zone,
                quantity="Wilcoxon, daily abs error 13-24h > 1-6h, all origin hours",
                estimate=float(z["difference"].mean()),
                p_value=p_value,
                n=str(len(z)),
                supported=bool(p_value < ALPHA),
                note=f"share of days positive={float((z['difference'] > 0).mean()):.3f}",
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# Verdicts and orchestration
# --------------------------------------------------------------------------------------


def _all_supported(results: pd.DataFrame, mask: pd.Series) -> bool:
    decided = results[mask & results["supported"].notna()]
    return bool(len(decided) > 0 and decided["supported"].astype(bool).all())


def derive_verdicts(results: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved verdict rule: the corrected analysis decides, disagreement is flagged."""
    rows: list[dict[str, Any]] = []
    for hypothesis in ("H1", "H2", "H3"):
        of_h = results["hypothesis"] == hypothesis
        if not of_h.any():
            continue
        corrected = _all_supported(results, of_h & (results["analysis"] == "corrected"))
        registered = _all_supported(results, of_h & (results["analysis"] == "registered"))
        if corrected == registered:
            verdict = "supported" if corrected else "falsified"
        else:
            verdict = "sensitive to method"
        if hypothesis == "H2":
            verdict += " (descriptive; outcome known in advance)"
        rows.append(
            _row(
                hypothesis=hypothesis,
                part="verdict",
                analysis="verdict",
                zone="",
                quantity=verdict,
                supported=corrected,
                note=f"corrected={corrected}, registered={registered}",
            )
        )
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def run_hypothesis_analysis(
    model: RawQuantileModel,
    preds_df: pd.DataFrame,
    bronze_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    train_renewable_thresholds: tuple[float, float] | None = None,
    prior_year_vre_threshold: float | None = None,
    n_resamples: int = N_RESAMPLES,
    n_repeats: int = N_PERMUTATION_REPEATS,
    n_single_repeats: int = N_SINGLE_FEATURE_REPEATS,
) -> AnalysisOutputs:
    """Run section 4 and H1-H3 of the analysis plan on one set of walk-forward predictions."""
    cfg = cfg or CarbonMLOpsConfig()
    calibration = calibration_table(preds_df, bronze_df, cfg, n_resamples=n_resamples)

    rows: list[dict[str, Any]] = []
    rows += evaluate_h1(
        preds_df,
        bronze_df,
        cfg,
        train_thresholds=train_renewable_thresholds,
        n_resamples=n_resamples,
    )
    rows += evaluate_h2(
        bronze_df,
        start=preds_df["target_timestamp"].min(),
        end=preds_df["target_timestamp"].max(),
        prior_year_threshold=prior_year_vre_threshold,
    )
    rows += evaluate_h3a(
        model,
        preds_df,
        cfg,
        n_resamples=n_resamples,
        n_repeats=n_repeats,
        n_single_repeats=n_single_repeats,
    )
    rows += evaluate_h3b(preds_df, cfg)

    results = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    results = pd.concat([results, derive_verdicts(results)], ignore_index=True)
    logger.info("Hypothesis analysis produced %d result rows", len(results))
    return AnalysisOutputs(calibration=calibration, results=results)
