"""Leakage-free horizon feature engineering for 24h carbon forecasting (Gate G2)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import is_databricks_runtime, read_table, write_table

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

FEATURE_PRIMARY_KEYS: list[str] = ["zone", "origin_timestamp", "horizon_h"]


def regularize_zone_to_hourly_grid(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Put one zone on a gap-free hourly grid and blank out intensities that cannot be trusted.

    Row-based shifts are only lags in hours when every hour has a row, so missing hours are
    inserted as all-NaN rows. Hours flagged upstream as partial (a published series was
    missing) or not fully covered (`hours_covered` < 1) keep their row but lose their
    intensity, so they are used neither as forecast origins, nor as lag values, nor as targets.
    """
    df = zone_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    ts = df["timestamp_utc"]
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamp_utc within a zone")
    if (ts != ts.dt.floor("h")).any():
        raise ValueError("timestamp_utc must be aligned to whole hours")

    untrusted = pd.Series(False, index=df.index)
    if "is_partial" in df.columns:
        untrusted |= df["is_partial"].astype(bool)
    if "hours_covered" in df.columns:
        untrusted |= df["hours_covered"] < 1.0 - 1e-6
    df.loc[untrusted, "carbon_intensity_g_kwh"] = np.nan

    grid = pd.date_range(ts.min(), ts.max(), freq="h").as_unit(ts.dt.unit)
    n_missing = len(grid) - len(df)
    if n_missing or untrusted.any():
        logger.info(
            "Hourly grid: inserted %d missing hours, masked %d partial/incomplete hours",
            n_missing,
            int(untrusted.sum()),
        )
    zone = df["zone"].iloc[0]
    df = df.set_index("timestamp_utc").reindex(grid).rename_axis("timestamp_utc").reset_index()
    df["zone"] = zone
    return df


def _build_zone_origin_features(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal origin-state and backward-looking lag/rolling features per zone.

    Expects the gap-free hourly frame from regularize_zone_to_hourly_grid, so shift(k) is
    exactly the value k hours before the origin (NaN where that hour is missing or masked).
    """
    df = grid_df.copy()
    df["origin_timestamp"] = df["timestamp_utc"]
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
        grid_df = regularize_zone_to_hourly_grid(sub)
        origin_df = _build_zone_origin_features(grid_df)
        target_lookup = (
            grid_df[["timestamp_utc", "carbon_intensity_g_kwh"]]
            .dropna(subset=["carbon_intensity_g_kwh"])
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

            target_hour = h_df["target_timestamp"].dt.hour
            h_df["target_hour_sin"] = np.sin(2.0 * np.pi * target_hour / 24.0)
            h_df["target_hour_cos"] = np.cos(2.0 * np.pi * target_hour / 24.0)
            h_df["target_dow"] = h_df["target_timestamp"].dt.day_of_week
            h_df["target_month"] = h_df["target_timestamp"].dt.month
            h_df["is_weekend"] = (h_df["target_dow"] >= 5).astype(int)

            h_df = h_df.merge(target_lookup, on="target_timestamp", how="left")
            zone_frames.append(h_df)

    features_df = (
        pd.concat(zone_frames, ignore_index=True)
        .sort_values(["zone", "origin_timestamp", "horizon_h"])
        .reset_index(drop=True)
    )

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


def materialize_uc_feature_table(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    base_dir: Path | None = None,
) -> str:
    """Register & merge causal features into Unity Catalog Feature Store (or local Parquet)."""
    cfg = cfg or CarbonMLOpsConfig()
    feature_only_df = features_df.drop(columns=["target_carbon_intensity"], errors="ignore").copy()

    if is_databricks_runtime():
        from databricks.feature_engineering import FeatureEngineeringClient
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        fe = FeatureEngineeringClient()
        uc_table = f"{cfg.catalog}.{cfg.schema}.grid_horizon_features"
        spark_df = spark.createDataFrame(feature_only_df)

        if not spark.catalog.tableExists(uc_table):
            fe.create_table(
                name=uc_table,
                primary_keys=FEATURE_PRIMARY_KEYS,
                df=spark_df,
                description="Causal 24h grid carbon intensity horizon features (DE-LU & FR).",
            )
        else:
            fe.write_table(name=uc_table, df=spark_df, mode="merge")
        logger.info(
            "Merged %d causal feature rows into UC Feature Store %s", len(feature_only_df), uc_table
        )
        return uc_table

    target_base = base_dir or cfg.local_data_dir
    local_table = f"{cfg.schema}.grid_horizon_features_uc_offline"
    written = write_table(
        feature_only_df, table_name=local_table, mode="overwrite", base_dir=target_base
    )
    logger.info("Local offline mode: wrote UC feature snapshot to %s", written)
    return written


def run_feature_pipeline(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Read Bronze table and materialize Silver & UC horizon features."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze_df = read_table(f"{cfg.schema}.bronze_entsoe_hourly", base_dir=cfg.local_data_dir)
    features_df = build_horizon_features(bronze_df, cfg=cfg, write_output=True)
    materialize_uc_feature_table(features_df, cfg=cfg)
    return features_df
