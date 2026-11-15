"""Leakage-free horizon feature engineering for 24h carbon forecasting (Gate G2)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import read_table, write_table

logger = logging.getLogger(__name__)

CAUSAL_FEATURE_COLS: tuple[str, ...] = (
    "horizon_h",
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
    "target_hour_sin",
    "target_hour_cos",
    "target_dow",
    "target_month",
    "is_weekend",
)


def _build_zone_origin_features(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal origin-state and backward-looking lag/rolling features per zone."""
    df = zone_df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    df["origin_timestamp"] = pd.to_datetime(df["timestamp_utc"])
    # Every backward-looking feature uses observations at or before origin_timestamp
    df["max_feature_timestamp"] = df["origin_timestamp"]

    ci = df["carbon_intensity_g_kwh"]
    df["carbon_intensity_origin"] = ci
    df["ci_lag_1h"] = ci.shift(1)
    df["ci_lag_3h"] = ci.shift(3)
    df["ci_lag_6h"] = ci.shift(6)
    df["ci_lag_24h"] = ci.shift(24)
    df["ci_roll24_mean"] = ci.rolling(window=24, min_periods=12).mean()
    df["ci_roll24_std"] = ci.rolling(window=24, min_periods=12).std().fillna(0.0)

    df["wind_share_origin"] = df["wind_share"]
    df["solar_share_origin"] = df["solar_share"]
    df["renewable_share_origin"] = df["renewable_share"]
    df["nuclear_share_origin"] = df["nuclear_share"]
    df["fossil_share_origin"] = df["fossil_share"]
    df["gen_total_mwh_origin"] = df["gen_total_mwh"]

    # Drop initial warmup rows lacking a full 24h lag
    df = df.dropna(subset=["ci_lag_24h", "carbon_intensity_origin"]).reset_index(drop=True)
    return df


def build_horizon_features(
    bronze_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    write_output: bool = True,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """Expand hourly zone telemetry into (zone, origin_timestamp, horizon_h) supervised rows."""
    cfg = cfg or CarbonMLOpsConfig()
    target_base = base_dir or cfg.local_data_dir

    zone_frames: list[pd.DataFrame] = []
    for zone in cfg.zones:
        sub = bronze_df[bronze_df["zone"] == zone].copy()
        if sub.empty:
            continue
        origin_df = _build_zone_origin_features(sub)
        target_lookup = (
            sub[["timestamp_utc", "carbon_intensity_g_kwh"]]
            .assign(timestamp_utc=lambda d: pd.to_datetime(d["timestamp_utc"]))
            .rename(
                columns={
                    "timestamp_utc": "target_timestamp",
                    "carbon_intensity_g_kwh": "target_carbon_intensity",
                }
            )
        )

        for h in cfg.horizons:
            h_df = origin_df[
                [
                    "zone",
                    "origin_timestamp",
                    "max_feature_timestamp",
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
                ]
            ].copy()
            h_df["horizon_h"] = int(h)
            h_df["horizon_bucket"] = cfg.bucket_for_horizon(int(h))
            h_df["target_timestamp"] = h_df["origin_timestamp"] + pd.Timedelta(hours=int(h))

            # Deterministic calendar encodings for target_timestamp (known at origin)
            target_hour = h_df["target_timestamp"].dt.hour
            h_df["target_hour_sin"] = np.sin(2.0 * np.pi * target_hour / 24.0)
            h_df["target_hour_cos"] = np.cos(2.0 * np.pi * target_hour / 24.0)
            h_df["target_dow"] = h_df["target_timestamp"].dt.day_of_week
            h_df["target_month"] = h_df["target_timestamp"].dt.month
            h_df["is_weekend"] = (h_df["target_dow"] >= 5).astype(int)

            # Left join realized future target (NaN for live inference horizons beyond data end)
            h_df = h_df.merge(target_lookup, on="target_timestamp", how="left")
            zone_frames.append(h_df)

    features_df = (
        pd.concat(zone_frames, ignore_index=True)
        .sort_values(["zone", "origin_timestamp", "horizon_h"])
        .reset_index(drop=True)
    )

    # Enforce Gate G2 temporal invariants before persisting
    if (features_df["max_feature_timestamp"] > features_df["origin_timestamp"]).any():
        raise AssertionError("Gate G2 violation: max_feature_timestamp exceeds origin_timestamp")
    expected_target = features_df["origin_timestamp"] + pd.to_timedelta(
        features_df["horizon_h"], unit="h"
    )
    if (features_df["target_timestamp"] != expected_target).any():
        raise AssertionError("Gate G2 violation: target_timestamp != origin_timestamp + horizon_h")

    if write_output:
        table_name = f"{cfg.schema}.grid_horizon_features"
        write_table(features_df, table_name=table_name, mode="overwrite", base_dir=target_base)
        logger.info("Materialized %d horizon feature rows into %s", len(features_df), table_name)

    return features_df


def run_feature_pipeline(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Read Bronze table and materialize Silver horizon features."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze_df = read_table(f"{cfg.schema}.bronze_entsoe_hourly", base_dir=cfg.local_data_dir)
    return build_horizon_features(bronze_df, cfg=cfg, write_output=True)
