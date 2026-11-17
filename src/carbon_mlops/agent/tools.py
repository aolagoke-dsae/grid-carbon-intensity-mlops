"""Deterministic tools behind the QA agent (gate G4).

Each tool reads a Gold table (on Databricks) or its committed snapshot (locally), computes its
answer with ordinary code, and returns a plain dictionary. The agent never does arithmetic and
never writes a number of its own: every figure in an answer is one of the `facts` returned here,
and `qa_agent.check_grounding` verifies that after the answer is written.

A tool result has three parts:

* `facts`: named numbers, each a number the answer may cite.
* `times`: named timestamps (UTC), formatted by the agent, never computed by it.
* `source`: which table was read and how fresh it is, so an answer can say what it rests on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from carbon_mlops.analysis.windows import RISK_COLUMNS, best_start, window_means
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.inference.batch_score import GOLD_TABLE
from carbon_mlops.io.store import is_databricks_runtime, read_table
from carbon_mlops.monitoring.compute_drift import MONITORING_CSV, MONITORING_TABLE

logger = logging.getLogger(__name__)

FORECAST_CSV = "latest_24h_forecasts.csv"
CHOICES: tuple[str, ...] = tuple(RISK_COLUMNS)  # "median", "cautious" (the upper bound)
DRIFT_GROUP = "feature_drift_same_season"
ToolResult = dict[str, Any]


class ToolInputError(ValueError):
    """The caller asked for something the tool cannot answer (unknown zone, bad window)."""


def load_forecasts(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """The latest 24-hour forecast per zone: the Gold table on Databricks, its snapshot locally."""
    cfg = cfg or CarbonMLOpsConfig()
    if is_databricks_runtime():
        df = read_table(f"{cfg.schema}.{GOLD_TABLE}", catalog=cfg.catalog)
    else:
        df = pd.read_csv(Path(cfg.dashboard_data_dir) / FORECAST_CSV)
    return _with_utc(df, ("origin_timestamp", "target_timestamp"))


def load_monitoring(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Weekly monitoring rows: the Gold table on Databricks, its snapshot locally."""
    cfg = cfg or CarbonMLOpsConfig()
    if is_databricks_runtime():
        df = read_table(f"{cfg.schema}.{MONITORING_TABLE}", catalog=cfg.catalog)
    else:
        df = pd.read_csv(Path(cfg.dashboard_data_dir) / MONITORING_CSV)
    return _with_utc(df, ("monitor_date",))


def _with_utc(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_datetime(out[column], utc=True)
    return out


def _zone_forecast(forecasts: pd.DataFrame, zone: str, cfg: CarbonMLOpsConfig) -> pd.DataFrame:
    """The 24 rows of one zone's latest forecast, in lead-time order, or a clear error."""
    if zone not in cfg.zones:
        raise ToolInputError(f"Unknown zone {zone!r}; expected one of {list(cfg.zones)}")
    rows = forecasts[forecasts["zone"] == zone]
    if rows.empty:
        raise ToolInputError(f"No forecast rows for zone {zone!r}")
    latest = rows[rows["origin_timestamp"] == rows["origin_timestamp"].max()].sort_values(
        "horizon_h"
    )
    if latest["horizon_h"].tolist() != list(cfg.horizons):
        raise ToolInputError(f"The latest forecast for {zone!r} does not cover every lead time")
    return latest.reset_index(drop=True)


def _source(table: str, day: pd.DataFrame) -> dict[str, str]:
    source = {"table": table, "origin_timestamp": day["origin_timestamp"].iloc[0].isoformat()}
    if "calibrated_as_of" in day.columns:
        source["calibrated_as_of"] = str(day["calibrated_as_of"].iloc[0])
    return source


@mlflow.trace(name="find_greenest_window", span_type="TOOL")
def find_greenest_window(
    zone: str,
    window_hours: int = 3,
    choose_by: str = "median",
    forecasts: pd.DataFrame | None = None,
    cfg: CarbonMLOpsConfig | None = None,
) -> ToolResult:
    """The cleanest run of `window_hours` consecutive hours in a zone's latest 24-hour forecast.

    `choose_by` is "median" (lowest median forecast) or "cautious" (lowest upper bound). The
    saving is the forecast's own expectation against the mean of the forecast day; it is not a
    realized saving.
    """
    cfg = cfg or CarbonMLOpsConfig()
    if choose_by not in CHOICES:
        raise ToolInputError(f"choose_by must be one of {list(CHOICES)}, got {choose_by!r}")
    day = _zone_forecast(forecasts if forecasts is not None else load_forecasts(cfg), zone, cfg)
    if not 1 <= int(window_hours) <= len(day):
        raise ToolInputError(f"window_hours must be between 1 and {len(day)}, got {window_hours}")
    window_hours = int(window_hours)

    scores = window_means(day[RISK_COLUMNS[choose_by]].to_numpy(dtype=float), window_hours)
    start = best_start(scores)
    chosen = day.iloc[start : start + window_hours]
    expected = float(chosen["p50"].mean())
    day_mean = float(day["p50"].mean())
    return {
        "tool": "find_greenest_window",
        "args": {"zone": zone, "window_hours": window_hours, "choose_by": choose_by},
        "facts": {
            "window_hours": window_hours,
            "forecast_horizon_h": len(day),
            "interval_level_pct": 100.0 * cfg.nominal_coverage,
            "start_lead_time_h": int(chosen["horizon_h"].iloc[0]),
            "expected_intensity": expected,
            "expected_lower_bound": float(chosen["p05"].mean()),
            "expected_upper_bound": float(chosen["p95"].mean()),
            "day_mean_expected_intensity": day_mean,
            "expected_saving_pct": 100.0 * (1.0 - expected / day_mean),
            "expected_avoided_kg_per_mwh": day_mean - expected,
        },
        "times": {
            "window_start": chosen["target_timestamp"].iloc[0].isoformat(),
            "window_end": (chosen["target_timestamp"].iloc[-1] + pd.Timedelta(hours=1)).isoformat(),
            "forecast_issued": day["origin_timestamp"].iloc[0].isoformat(),
        },
        "source": _source(GOLD_TABLE, day),
    }


@mlflow.trace(name="compare_zones_24h", span_type="TOOL")
def compare_zones_24h(
    forecasts: pd.DataFrame | None = None,
    cfg: CarbonMLOpsConfig | None = None,
) -> ToolResult:
    """The latest 24-hour forecast of every zone side by side: level, range and uncertainty."""
    cfg = cfg or CarbonMLOpsConfig()
    forecasts = forecasts if forecasts is not None else load_forecasts(cfg)
    facts: dict[str, float | int] = {}
    times: dict[str, str] = {}
    means: dict[str, float] = {}
    first: pd.DataFrame | None = None
    for zone in cfg.zones:
        day = _zone_forecast(forecasts, zone, cfg)
        first = day if first is None else first
        key = zone.lower().replace("-", "_")
        cleanest = day.loc[day["p50"].idxmin()]
        dirtiest = day.loc[day["p50"].idxmax()]
        means[zone] = float(day["p50"].mean())
        facts[f"{key}_mean_expected_intensity"] = means[zone]
        facts[f"{key}_cleanest_hour_intensity"] = float(cleanest["p50"])
        facts[f"{key}_dirtiest_hour_intensity"] = float(dirtiest["p50"])
        facts[f"{key}_mean_interval_width"] = float((day["p95"] - day["p05"]).mean())
        times[f"{key}_cleanest_hour"] = cleanest["target_timestamp"].isoformat()
        times[f"{key}_dirtiest_hour"] = dirtiest["target_timestamp"].isoformat()
    cleaner = min(means, key=means.get)
    dirtier = max(means, key=means.get)
    facts["ratio_dirtier_to_cleaner"] = means[dirtier] / means[cleaner]
    facts["forecast_horizon_h"] = len(cfg.horizons)
    assert first is not None
    times["forecast_issued"] = first["origin_timestamp"].iloc[0].isoformat()
    return {
        "tool": "compare_zones_24h",
        "args": {},
        "facts": facts,
        "times": times,
        "labels": {"cleaner_zone": cleaner, "dirtier_zone": dirtier},
        "source": _source(GOLD_TABLE, first),
    }


@mlflow.trace(name="check_model_health", span_type="TOOL")
def check_model_health(
    zone: str,
    monitoring: pd.DataFrame | None = None,
    cfg: CarbonMLOpsConfig | None = None,
) -> ToolResult:
    """The latest week of monitoring for one zone: coverage, skill and feature drift.

    The verdict is a rule, not a judgement: "healthy" when the latest weekly coverage is inside
    the accepted band and the median forecast beats "no change"; otherwise "needs attention".
    """
    cfg = cfg or CarbonMLOpsConfig()
    if zone not in cfg.zones:
        raise ToolInputError(f"Unknown zone {zone!r}; expected one of {list(cfg.zones)}")
    monitoring = monitoring if monitoring is not None else load_monitoring(cfg)
    rows = monitoring[monitoring["zone"] == zone]
    if rows.empty:
        raise ToolInputError(f"No monitoring rows for zone {zone!r}")
    latest_date = rows["monitor_date"].max()
    latest = rows[rows["monitor_date"] == latest_date].set_index("metric")

    coverage_row = latest.loc["coverage_7d"]
    coverage = float(coverage_row["value"])
    skill = float(latest.loc["median_skill_vs_persistence", "value"])
    drift = latest[latest["metric_group"] == DRIFT_GROUP]
    history = rows[rows["metric"] == "coverage_7d"]
    in_band = float(coverage_row["lower_bound"]) <= coverage <= float(coverage_row["upper_bound"])
    healthy = bool(in_band and skill > 0)
    return {
        "tool": "check_model_health",
        "args": {"zone": zone},
        "facts": {
            "weekly_coverage_pct": 100.0 * coverage,
            "band_lower_pct": 100.0 * float(coverage_row["lower_bound"]),
            "band_upper_pct": 100.0 * float(coverage_row["upper_bound"]),
            "target_coverage_pct": 100.0 * cfg.nominal_coverage,
            "skill_vs_no_change_pct": 100.0 * skill,
            "drift_features_on_alert": int((drift["status"] == "alert").sum()),
            "drift_features_checked": int(len(drift)),
            "weeks_monitored": int(len(history)),
            "weeks_outside_band": int((history["status"] != "ok").sum()),
            "forecasts_in_week": int(coverage_row["n"]),
        },
        "times": {"week_ending": latest_date.isoformat()},
        "labels": {
            "coverage_status": str(coverage_row["status"]),
            "verdict": "healthy" if healthy else "needs attention",
        },
        "source": {"table": MONITORING_TABLE, "monitor_date": latest_date.isoformat()},
    }


TOOLS = {
    "find_greenest_window": find_greenest_window,
    "compare_zones_24h": compare_zones_24h,
    "check_model_health": check_model_health,
}


def fact_values(result: ToolResult) -> list[float]:
    """Every number a tool result allows an answer to cite."""
    return [float(v) for v in result["facts"].values() if isinstance(v, int | float | np.number)]
