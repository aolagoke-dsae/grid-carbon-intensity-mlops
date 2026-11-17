"""Feature drift (PSI) and rolling coverage monitoring for the deployed forecaster (US-3.2).

Two questions are answered on a weekly grid of monitoring dates:

1. Have the model's inputs drifted away from what it was trained on? Population stability
   index (PSI) of each origin-state feature over the trailing 30 days, against the whole Train
   period and against the same 30 days one year earlier (which holds the season fixed).
2. Is the 90% interval still covering? Empirical coverage of the forecasts whose targets were
   observed in the trailing 7 days, per zone and per horizon bucket, with interval width and
   skill against the persistence forecast.

The forecasts are the ones the recalibration schedule would have issued: the base models are
fixed and the conformal offsets are replayed week by week from errors observed before each
origin. Nothing here fits, tunes or recalibrates a model. Thresholds are conventions fixed in
advance (PSI 0.10 / 0.25, coverage inside `cfg.coverage_bounds`), not values chosen from data.

A 7-day window holds a few thousand strongly autocorrelated forecasts, so its coverage is
noisy: an occasional value outside the band is expected, a run of them is a signal.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import is_databricks_runtime, write_csv_extract, write_table
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster
from carbon_mlops.models.train import (
    DEFAULT_LGBM_PARAMS,
    chronological_four_way_split,
    evaluate_on_gate_val,
    evaluate_on_locked_test,
    fit_quantile_models,
    pinball_loss,
)

logger = logging.getLogger(__name__)

MONITORING_TABLE = "gold_monitoring"
MONITORING_CSV = "gold_monitoring.csv"
EXPORT_VOLUME = "exports"

PSI_BINS = 10
PSI_WARN = 0.10
PSI_ALERT = 0.25
PSI_FLOOR = 1e-4
COVERAGE_WINDOW_DAYS = 7
PSI_WINDOW_DAYS = 30
SEASONAL_LAG_DAYS = 365
MIN_SEASONAL_COMPLETENESS = 0.8

# One row per forecast origin. Lags are left out: they repeat the origin intensity.
MONITORED_FEATURES: tuple[str, ...] = (
    "carbon_intensity_origin",
    "ci_roll24_mean",
    "ci_roll24_std",
    "wind_share_origin",
    "solar_share_origin",
    "renewable_share_origin",
    "nuclear_share_origin",
    "fossil_share_origin",
    "gen_total_mwh_origin",
)

MONITORING_COLUMNS: tuple[str, ...] = (
    "monitor_date",
    "zone",
    "metric_group",
    "metric",
    "value",
    "n",
    "window_days",
    "lower_bound",
    "upper_bound",
    "status",
)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = PSI_BINS
) -> float:
    """PSI of `current` against `reference`, with bins at the reference quantiles.

    PSI = sum over bins of (current share - reference share) * ln(current / reference).
    Shares are floored at PSI_FLOOR so an empty bin does not give an infinite value. NaN
    inputs are ignored; an empty reference or current sample gives NaN.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    interior = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(ref, interior))
    ref_share = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=edges.size + 1)
    cur_share = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=edges.size + 1)
    ref_share = np.clip(ref_share / ref.size, PSI_FLOOR, None)
    cur_share = np.clip(cur_share / cur.size, PSI_FLOOR, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_status(value: float) -> str:
    """Conventional reading of a PSI value: below 0.10 stable, 0.10 to 0.25 moderate shift."""
    if np.isnan(value):
        return "no_data"
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARN:
        return "warn"
    return "ok"


def coverage_status(value: float, bounds: tuple[float, float]) -> str:
    """Whether a coverage value lies inside the reference band."""
    if np.isnan(value):
        return "no_data"
    low, high = bounds
    if value < low:
        return "below_band"
    if value > high:
        return "above_band"
    return "ok"


def monitoring_dates(start: pd.Timestamp, end: pd.Timestamp, step_days: int) -> pd.DatetimeIndex:
    """Monitoring dates: midnight UTC every `step_days` days, each with a full step behind it."""
    first = start.ceil("D") + pd.Timedelta(days=step_days)
    return pd.date_range(first, end.floor("D") + pd.Timedelta(days=1), freq=f"{step_days}D")


def origin_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (zone, origin): the origin-state features are repeated across lead times."""
    return features_df.drop_duplicates(subset=["zone", "origin_timestamp"])[
        ["zone", "origin_timestamp", *MONITORED_FEATURES]
    ]


def feature_psi_table(
    features_df: pd.DataFrame,
    reference_end: pd.Timestamp,
    dates: pd.DatetimeIndex,
    cfg: CarbonMLOpsConfig,
    window_days: int = PSI_WINDOW_DAYS,
) -> pd.DataFrame:
    """PSI of each monitored feature per zone and monitoring date, against two references.

    Current sample: origins in the `window_days` before the monitoring date. Each origin
    counts once. It is compared with:

    * `feature_drift`: every origin up to `reference_end` (the whole Train period). With a
      30-day window this mostly measures the season, so it is expected to be high.
    * `feature_drift_same_season`: the Train origins of the same `window_days` one year
      earlier. This holds the season fixed and measures year-on-year change. It is reported
      only when that earlier window lies inside Train and is at least 80% complete.
    """
    origins = origin_feature_frame(features_df)
    one_year = pd.Timedelta(days=SEASONAL_LAG_DAYS)
    window = pd.Timedelta(days=window_days)
    min_seasonal_rows = int(MIN_SEASONAL_COMPLETENESS * window_days * 24)
    rows: list[dict[str, object]] = []

    def add(date: pd.Timestamp, zone: str, group: str, metric: str, value: float, n: int) -> None:
        rows.append(
            {
                "monitor_date": date,
                "zone": zone,
                "metric_group": group,
                "metric": metric,
                "value": value,
                "n": n,
                "window_days": window_days,
                "lower_bound": np.nan,
                "upper_bound": PSI_ALERT,
                "status": psi_status(value),
            }
        )

    for zone in cfg.zones:
        zone_origins = origins[origins["zone"] == zone]
        when = zone_origins["origin_timestamp"]
        train = zone_origins[when <= reference_end]
        train_when = train["origin_timestamp"]
        for date in dates:
            current = zone_origins[(when >= date - window) & (when < date)]
            same_season = train[
                (train_when >= date - one_year - window) & (train_when < date - one_year)
            ]
            season_usable = len(same_season) >= min_seasonal_rows
            for feature in MONITORED_FEATURES:
                now = current[feature].to_numpy()
                add(
                    date,
                    zone,
                    "feature_drift",
                    f"psi_{feature}",
                    population_stability_index(train[feature].to_numpy(), now),
                    len(current),
                )
                seasonal_value = (
                    population_stability_index(same_season[feature].to_numpy(), now)
                    if season_usable
                    else float("nan")
                )
                add(
                    date,
                    zone,
                    "feature_drift_same_season",
                    f"psi_same_season_{feature}",
                    seasonal_value,
                    len(current),
                )
    return pd.DataFrame(rows, columns=list(MONITORING_COLUMNS))


def rolling_coverage_table(
    preds_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cfg: CarbonMLOpsConfig,
    window_days: int = COVERAGE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Coverage, miss direction, width and skill of forecasts whose targets fell in each window.

    The window of a monitoring date holds the forecasts with a target in the `window_days`
    before it, so every outcome it uses was observed by that date.
    """
    y = preds_df["target_carbon_intensity"].to_numpy()
    work = preds_df.assign(
        covered=(y >= preds_df["p05"].to_numpy()) & (y <= preds_df["p95"].to_numpy()),
        below=y < preds_df["p05"].to_numpy(),
        above=y > preds_df["p95"].to_numpy(),
    )
    bounds = cfg.coverage_bounds
    rows: list[dict[str, object]] = []

    def add(date: pd.Timestamp, zone: str, metric: str, value: float, n: int, banded: bool) -> None:
        rows.append(
            {
                "monitor_date": date,
                "zone": zone,
                "metric_group": "coverage" if banded else "forecast_quality",
                "metric": metric,
                "value": value,
                "n": n,
                "window_days": window_days,
                "lower_bound": bounds[0] if banded else np.nan,
                "upper_bound": bounds[1] if banded else np.nan,
                "status": coverage_status(value, bounds) if banded else "info",
            }
        )

    for zone in cfg.zones:
        zone_rows = work[work["zone"] == zone]
        for date in dates:
            in_window = (zone_rows["target_timestamp"] >= date - pd.Timedelta(days=window_days)) & (
                zone_rows["target_timestamp"] < date
            )
            window = zone_rows[in_window]
            n = len(window)
            if n == 0:
                add(date, zone, f"coverage_{window_days}d", float("nan"), 0, banded=True)
                continue
            add(date, zone, f"coverage_{window_days}d", float(window["covered"].mean()), n, True)
            for bucket in cfg.horizon_buckets:
                part = window[window["horizon_bucket"] == bucket]
                value = float(part["covered"].mean()) if len(part) else float("nan")
                add(date, zone, f"coverage_{window_days}d_{bucket}", value, len(part), True)
            add(date, zone, "share_below_p05", float(window["below"].mean()), n, False)
            add(date, zone, "share_above_p95", float(window["above"].mean()), n, False)
            add(date, zone, "mean_interval_width", float(window["interval_width"].mean()), n, False)
            truth = window["target_carbon_intensity"].to_numpy()
            model_loss = pinball_loss(truth, window["p50"].to_numpy(), 0.50)
            baseline_loss = pinball_loss(truth, window["carbon_intensity_origin"].to_numpy(), 0.50)
            skill = 1.0 - model_loss / baseline_loss if baseline_loss > 0 else float("nan")
            add(date, zone, "median_skill_vs_persistence", float(skill), n, False)
    return pd.DataFrame(rows, columns=list(MONITORING_COLUMNS))


def monitoring_predictions(
    forecaster: ConformalCarbonForecaster,
    splits: dict[str, pd.DataFrame],
    cfg: CarbonMLOpsConfig,
) -> pd.DataFrame:
    """Walk-forward forecasts over Gate-Val and the period after it, as the schedule issues them.

    The period after Gate-Val is the held-out test period. Reading it here was approved by the
    author on 2026-11-16: it recomputes forecasts already published in
    docs/LOCKED_TEST_RESULTS.md and is used for reporting only.
    """
    _, gate_preds = evaluate_on_gate_val(forecaster, splits, cfg)
    _, later_preds = evaluate_on_locked_test(forecaster, splits, cfg)
    return pd.concat([gate_preds, later_preds], ignore_index=True)


def compute_monitoring(
    forecaster: ConformalCarbonForecaster,
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    coverage_window_days: int = COVERAGE_WINDOW_DAYS,
    psi_window_days: int = PSI_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build the long-format monitoring table: one row per date, zone and metric."""
    cfg = cfg or CarbonMLOpsConfig()
    splits = chronological_four_way_split(features_df, cfg=cfg)
    preds = monitoring_predictions(forecaster, splits, cfg)
    dates = monitoring_dates(
        preds["origin_timestamp"].min(),
        preds["target_timestamp"].max(),
        cfg.recalibration_step_days,
    )
    reference_end = splits["train"]["origin_timestamp"].max()
    table = pd.concat(
        [
            rolling_coverage_table(preds, dates, cfg, window_days=coverage_window_days),
            feature_psi_table(features_df, reference_end, dates, cfg, window_days=psi_window_days),
        ],
        ignore_index=True,
    )
    return table.sort_values(["monitor_date", "zone", "metric_group", "metric"]).reset_index(
        drop=True
    )


def write_monitoring(table: pd.DataFrame, cfg: CarbonMLOpsConfig) -> str:
    """Write the monitoring table and its CSV extract; return the CSV location."""
    write_table(
        table,
        table_name=f"{cfg.schema}.{MONITORING_TABLE}",
        mode="overwrite",
        base_dir=cfg.local_data_dir,
        catalog=cfg.catalog,
    )
    if is_databricks_runtime():
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{EXPORT_VOLUME}")
        csv_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/{EXPORT_VOLUME}/{MONITORING_CSV}"
    else:
        cfg.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(cfg.dashboard_data_dir) / MONITORING_CSV)
    write_csv_extract(table, csv_path, float_format="%.6g")
    logger.info("Wrote %d monitoring rows; extract at %s", len(table), csv_path)
    return csv_path


def summarize_latest(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the most recent monitoring date that are not `ok` or `info`."""
    latest = table[table["monitor_date"] == table["monitor_date"].max()]
    return latest[~latest["status"].isin(["ok", "info"])]


def run_local_monitoring(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Local run: fit the frozen model on Train from the snapshot and write the monitoring CSV."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze = ingest_bronze_snapshot(cfg=cfg)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False)
    splits = chronological_four_way_split(features, cfg=cfg)
    forecaster = ConformalCarbonForecaster(
        models_by_zone=fit_quantile_models(splits["train"], cfg, dict(DEFAULT_LGBM_PARAMS)),
        conformal_offsets={},
        target_mode=cfg.target_mode,
    )
    table = compute_monitoring(forecaster, features, cfg)
    write_monitoring(table, cfg)
    flagged = summarize_latest(table)
    logger.info(
        "Monitoring complete: %d dates, %d rows, %d flagged on the latest date",
        table["monitor_date"].nunique(),
        len(table),
        len(flagged),
    )
    return table


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argparse.ArgumentParser(description="Compute feature drift and rolling coverage.").parse_args()
    run_local_monitoring()


if __name__ == "__main__":
    main()
