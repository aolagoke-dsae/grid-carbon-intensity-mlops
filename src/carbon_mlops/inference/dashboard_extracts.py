"""CSV extracts for the two BI dashboards (US-3.3): forecast history and candidate windows.

Both are built from the forecasts the weekly recalibration schedule would have issued (the
same walk-forward replay the monitor uses), one forecast per zone and day, issued at
00:00 UTC for the following 24 hours. Nothing here fits, tunes or recalibrates a model.

* `forecast_history.csv`: one row per zone, day and forecast hour, with the interval and
  the realized intensity.
* `window_candidates.csv`: one row per zone, day, window length and possible start hour,
  with the forecast and realized mean of that window and flags for the window each strategy
  would have picked.
* `window_strategy_summary.csv`: what each strategy achieved on average.

Strategies compared, all choosing one start hour per day:
  median      lowest mean of the median forecast
  cautious    lowest mean of the upper bound (p95)
  yesterday   the start hour that was cleanest on the previous day (a rule of thumb that
              needs no model; the baseline the forecast has to beat)
  hindsight   the start hour that turned out cleanest (unattainable; the upper limit)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_mlops.analysis.windows import RISK_COLUMNS, best_start, hindsight_ranks, window_means
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import is_databricks_runtime, write_csv_extract, write_table
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster
from carbon_mlops.models.train import (
    DEFAULT_LGBM_PARAMS,
    chronological_four_way_split,
    fit_quantile_models,
)
from carbon_mlops.monitoring.compute_drift import EXPORT_VOLUME, monitoring_predictions

logger = logging.getLogger(__name__)

DAILY_ORIGIN_HOUR_UTC = 0
WINDOW_LENGTHS: tuple[int, ...] = (2, 3, 4, 6)
STRATEGIES: tuple[str, ...] = ("median", "cautious", "yesterday", "hindsight")

HISTORY_TABLE = "gold_forecast_history"
CANDIDATES_TABLE = "gold_window_candidates"
SUMMARY_TABLE = "gold_window_strategy_summary"
VALUE_WEEKLY_TABLE = "gold_window_value_weekly"
HISTORY_CSV = "forecast_history.csv"
CANDIDATES_CSV = "window_candidates.csv"
SUMMARY_CSV = "window_strategy_summary.csv"
VALUE_WEEKLY_CSV = "window_value_weekly.csv"

HISTORY_COLUMNS: tuple[str, ...] = (
    "zone",
    "forecast_date",
    "period",
    "origin_timestamp",
    "horizon_h",
    "target_timestamp",
    "target_hour_utc",
    "p05",
    "p50",
    "p95",
    "interval_width",
    "realized",
    "covered",
    "calibrated_as_of",
)


def forecast_history_table(
    preds_df: pd.DataFrame,
    validation_end: pd.Timestamp,
    cfg: CarbonMLOpsConfig,
    origin_hour_utc: int = DAILY_ORIGIN_HOUR_UTC,
) -> pd.DataFrame:
    """One daily forecast per zone: the origin at `origin_hour_utc` with all its lead times.

    `period` is "validation" for origins up to `validation_end` (the end of Gate-Val) and
    "held_out" afterwards. Origins that lack any lead time are left out.
    """
    daily = preds_df[preds_df["origin_timestamp"].dt.hour == origin_hour_utc]
    size = daily.groupby(["zone", "origin_timestamp"])["horizon_h"].transform("size")
    daily = daily[size == len(cfg.horizons)].copy()

    realized = daily["target_carbon_intensity"].to_numpy()
    out = pd.DataFrame(
        {
            "zone": daily["zone"].to_numpy(),
            "forecast_date": daily["origin_timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
            "period": np.where(
                (daily["origin_timestamp"] <= validation_end).to_numpy(), "validation", "held_out"
            ),
            "origin_timestamp": daily["origin_timestamp"].to_numpy(),
            "horizon_h": daily["horizon_h"].astype(int).to_numpy(),
            "target_timestamp": daily["target_timestamp"].to_numpy(),
            "target_hour_utc": daily["target_timestamp"].dt.hour.to_numpy(),
            "p05": daily["p05"].to_numpy(),
            "p50": daily["p50"].to_numpy(),
            "p95": daily["p95"].to_numpy(),
            "interval_width": daily["interval_width"].to_numpy(),
            "realized": realized,
            "covered": (realized >= daily["p05"].to_numpy())
            & (realized <= daily["p95"].to_numpy()),
            "calibrated_as_of": daily["calibrated_as_of"].to_numpy(),
        }
    )
    out["origin_timestamp"] = daily["origin_timestamp"].to_numpy()
    return out.sort_values(["zone", "forecast_date", "horizon_h"]).reset_index(drop=True)[
        list(HISTORY_COLUMNS)
    ]


def window_candidates_table(
    history_df: pd.DataFrame,
    window_lengths: tuple[int, ...] = WINDOW_LENGTHS,
) -> pd.DataFrame:
    """Every possible window of each daily forecast, scored and flagged per strategy.

    `saving_vs_day_mean` is 1 - (realized mean of the window) / (realized mean of the day):
    positive when the window was cleaner than running at an arbitrary time. `regret` is the
    realized mean of the window minus that of the cleanest window in hindsight.
    `avoided_kg_per_mwh` is the day's realized mean minus the window's: intensity in gCO2eq/kWh
    is the same number as kg CO2eq per MWh, so it is the CO2 avoided by moving 1 MWh of
    consumption from an arbitrary time of day into the window.
    """
    rows: list[pd.DataFrame] = []
    for zone, zone_days in history_df.groupby("zone", sort=True):
        previous_best: dict[int, int | None] = dict.fromkeys(window_lengths)
        previous_date: pd.Timestamp | None = None
        for forecast_date, day in zone_days.groupby("forecast_date", sort=True):
            day = day.sort_values("horizon_h")
            this_date = pd.Timestamp(forecast_date)
            consecutive = previous_date is not None and this_date - previous_date == pd.Timedelta(
                days=1
            )
            targets = pd.DatetimeIndex(day["target_timestamp"])
            realized = day["realized"].to_numpy(dtype=float)
            day_mean = float(realized.mean())
            for length in window_lengths:
                expected = {
                    risk: window_means(day[column].to_numpy(dtype=float), length)
                    for risk, column in RISK_COLUMNS.items()
                }
                realized_mean = window_means(realized, length)
                picks = {
                    "median": best_start(expected["median"]),
                    "cautious": best_start(expected["cautious"]),
                    "hindsight": best_start(realized_mean),
                }
                yesterday = previous_best[length] if consecutive else None
                n_windows = realized_mean.size
                starts = np.arange(n_windows)
                frame = pd.DataFrame(
                    {
                        "zone": zone,
                        "forecast_date": forecast_date,
                        "period": day["period"].iloc[0],
                        "window_hours": length,
                        "start_horizon_h": starts + 1,
                        "start_timestamp": targets[starts],
                        "end_timestamp": targets[starts + length - 1] + pd.Timedelta(hours=1),
                        "expected_p50": expected["median"],
                        "expected_p95": expected["cautious"],
                        "realized_mean": realized_mean,
                        "day_mean_realized": day_mean,
                        "saving_vs_day_mean": 1.0 - realized_mean / day_mean,
                        "avoided_kg_per_mwh": day_mean - realized_mean,
                        "regret": realized_mean - realized_mean[picks["hindsight"]],
                        "hindsight_rank": hindsight_ranks(realized_mean),
                        "n_windows": n_windows,
                        "recommended_median": starts == picks["median"],
                        "recommended_cautious": starts == picks["cautious"],
                        "recommended_yesterday": (
                            starts == yesterday
                            if yesterday is not None
                            else np.zeros(n_windows, bool)
                        ),
                        "best_in_hindsight": starts == picks["hindsight"],
                    }
                )
                rows.append(frame)
                previous_best[length] = picks["hindsight"]
            previous_date = this_date
    return pd.concat(rows, ignore_index=True)


STRATEGY_FLAGS: dict[str, str] = {
    "median": "recommended_median",
    "cautious": "recommended_cautious",
    "yesterday": "recommended_yesterday",
    "hindsight": "best_in_hindsight",
}


def comparable_choices(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """The window each strategy chose, on the days where every strategy made a choice.

    One row per zone, period, window length, day and strategy. Restricting to days with a
    "yesterday" choice judges the model-based strategies and the rule of thumb on the same days.
    """
    keys = ["zone", "period", "window_hours", "forecast_date"]
    has_yesterday = candidates_df.groupby(keys)["recommended_yesterday"].transform("any")
    comparable = candidates_df[has_yesterday]
    chosen = [
        comparable[comparable[STRATEGY_FLAGS[strategy]]].assign(strategy=strategy)
        for strategy in STRATEGIES
    ]
    return pd.concat(chosen, ignore_index=True)


def strategy_summary_table(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Average outcome of each strategy per zone, period and window length.

    Besides the relative saving, this states the outcome in the units a decision needs:
    `mean_avoided_kg_per_mwh` is the CO2 avoided per MWh moved into the chosen window instead
    of running at an arbitrary time of day, and `total_avoided_kg_per_daily_mwh` is what a load
    of 1 MWh per day would have avoided over the period. `share_of_hindsight_value_captured`
    is the strategy's total as a share of what choosing with hindsight would have avoided.
    """
    choices = comparable_choices(candidates_df)
    group_keys = ["zone", "period", "window_hours"]
    rows: list[dict[str, object]] = []
    for (zone, period, length, strategy), grp in choices.groupby([*group_keys, "strategy"]):
        rows.append(
            {
                "zone": zone,
                "period": period,
                "window_hours": int(length),
                "strategy": strategy,
                "n_days": len(grp),
                "mean_saving_vs_day_mean": float(grp["saving_vs_day_mean"].mean()),
                "share_of_days_cleaner_than_day_mean": float(
                    (grp["saving_vs_day_mean"] > 0).mean()
                ),
                "mean_realized_intensity": float(grp["realized_mean"].mean()),
                "mean_regret": float(grp["regret"].mean()),
                "share_of_days_best_window": float((grp["hindsight_rank"] == 1).mean()),
                "median_hindsight_rank": float(grp["hindsight_rank"].median()),
                "mean_avoided_kg_per_mwh": float(grp["avoided_kg_per_mwh"].mean()),
                "total_avoided_kg_per_daily_mwh": float(grp["avoided_kg_per_mwh"].sum()),
                "share_of_days_dirtier_than_day_mean": float(
                    (grp["saving_vs_day_mean"] < 0).mean()
                ),
                "worst_day_saving_vs_day_mean": float(grp["saving_vs_day_mean"].min()),
            }
        )
    summary = pd.DataFrame(rows)
    hindsight_total = (
        summary[summary["strategy"] == "hindsight"]
        .set_index(group_keys)["total_avoided_kg_per_daily_mwh"]
        .rename("hindsight_total")
    )
    summary = summary.join(hindsight_total, on=group_keys)
    summary["share_of_hindsight_value_captured"] = (
        summary["total_avoided_kg_per_daily_mwh"] / summary["hindsight_total"]
    )
    return (
        summary.drop(columns="hindsight_total")
        .sort_values(["zone", "period", "window_hours", "strategy"])
        .reset_index(drop=True)
    )


def window_value_weekly_table(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Weekly value of each strategy: CO2 avoided per MWh moved, and how often it backfired.

    Weeks start on Monday (`week_start`) and hold only the days on which every strategy made a
    choice, so a week can have fewer than seven days; `n_days` says how many.
    """
    choices = comparable_choices(candidates_df)
    day = pd.to_datetime(choices["forecast_date"])
    choices = choices.assign(
        week_start=(day - pd.to_timedelta(day.dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d")
    )
    rows: list[dict[str, object]] = []
    for (zone, length, week, strategy), grp in choices.groupby(
        ["zone", "window_hours", "week_start", "strategy"]
    ):
        rows.append(
            {
                "zone": zone,
                "window_hours": int(length),
                "week_start": week,
                "strategy": strategy,
                "n_days": len(grp),
                "mean_avoided_kg_per_mwh": float(grp["avoided_kg_per_mwh"].mean()),
                "mean_saving_vs_day_mean": float(grp["saving_vs_day_mean"].mean()),
                "share_of_days_dirtier_than_day_mean": float(
                    (grp["saving_vs_day_mean"] < 0).mean()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["zone", "window_hours", "week_start", "strategy"])
        .reset_index(drop=True)
    )


def build_dashboard_extracts(
    forecaster: ConformalCarbonForecaster,
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Replay the schedule over Gate-Val and the held-out period and build all four tables."""
    cfg = cfg or CarbonMLOpsConfig()
    splits = chronological_four_way_split(features_df, cfg=cfg)
    preds = monitoring_predictions(forecaster, splits, cfg)
    validation_end = splits["gate_val"]["origin_timestamp"].max()
    history = forecast_history_table(preds, validation_end, cfg)
    candidates = window_candidates_table(history)
    return {
        HISTORY_CSV: history,
        CANDIDATES_CSV: candidates,
        SUMMARY_CSV: strategy_summary_table(candidates),
        VALUE_WEEKLY_CSV: window_value_weekly_table(candidates),
    }


def write_dashboard_extracts(tables: dict[str, pd.DataFrame], cfg: CarbonMLOpsConfig) -> list[str]:
    """Write each table to the lakehouse and as a CSV; return the CSV locations."""
    table_names = {
        HISTORY_CSV: HISTORY_TABLE,
        CANDIDATES_CSV: CANDIDATES_TABLE,
        SUMMARY_CSV: SUMMARY_TABLE,
        VALUE_WEEKLY_CSV: VALUE_WEEKLY_TABLE,
    }
    if is_databricks_runtime():
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{EXPORT_VOLUME}")
        csv_dir = f"/Volumes/{cfg.catalog}/{cfg.schema}/{EXPORT_VOLUME}"
    else:
        cfg.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = str(Path(cfg.dashboard_data_dir))

    written: list[str] = []
    for csv_name, table in tables.items():
        write_table(
            table,
            table_name=f"{cfg.schema}.{table_names[csv_name]}",
            mode="overwrite",
            base_dir=cfg.local_data_dir,
            catalog=cfg.catalog,
        )
        csv_path = f"{csv_dir}/{csv_name}"
        write_csv_extract(table, csv_path, float_format="%.6g")
        written.append(csv_path)
        logger.info("Wrote %d rows to %s", len(table), csv_path)
    return written


def run_local_extracts(cfg: CarbonMLOpsConfig | None = None) -> dict[str, pd.DataFrame]:
    """Local run: fit the frozen model on Train from the snapshot and write the extracts."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze = ingest_bronze_snapshot(cfg=cfg)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False)
    splits = chronological_four_way_split(features, cfg=cfg)
    forecaster = ConformalCarbonForecaster(
        models_by_zone=fit_quantile_models(splits["train"], cfg, dict(DEFAULT_LGBM_PARAMS)),
        conformal_offsets={},
        target_mode=cfg.target_mode,
    )
    tables = build_dashboard_extracts(forecaster, features, cfg)
    write_dashboard_extracts(tables, cfg)
    return tables


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argparse.ArgumentParser(
        description="Write the CSV extracts for the BI dashboards."
    ).parse_args()
    run_local_extracts()


if __name__ == "__main__":
    main()
