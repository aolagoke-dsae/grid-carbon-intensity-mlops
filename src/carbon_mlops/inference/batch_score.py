"""End-to-end local and Databricks Spark UDF batch scoring runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import (
    CAUSAL_FEATURE_COLS,
    FEATURE_TABLE,
    build_horizon_features,
)
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import is_databricks_runtime, write_table
from carbon_mlops.models.train import resolve_model_name, train_split_cqr
from carbon_mlops.viz.diagnostics import plot_probabilistic_forecast

logger = logging.getLogger(__name__)

GOLD_TABLE = "gold_forecasts"
EXPORT_VOLUME = "exports"


def run_local_pipeline(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Execute Bronze -> Silver -> Split-CQR -> Gold forecasts & fan chart PNG locally."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze_df = ingest_bronze_snapshot(cfg=cfg)
    features_df = build_horizon_features(bronze_df, cfg=cfg, write_output=True)
    artifacts = train_split_cqr(features_df, cfg=cfg)

    # Show the latest Gate-Val origin that still has every horizon (splits are purged at their
    # boundaries). Its forecast comes from the walk-forward replay, so it was calibrated
    # only on errors observed before that origin: an out-of-sample picture.
    gate_preds = artifacts.gate_val_predictions
    n_horizons = gate_preds.groupby(["zone", "origin_timestamp"])["horizon_h"].transform("size")
    complete = gate_preds[n_horizons == len(cfg.horizons)]
    latest_origins = complete.groupby("zone")["origin_timestamp"].transform("max")
    gold_df = (
        complete[complete["origin_timestamp"] == latest_origins]
        .drop(columns=["low_score", "high_score", "raw_q05", "raw_q50", "raw_q95"])
        .sort_values(["zone", "horizon_h"])
        .reset_index(drop=True)
    )

    write_table(
        gold_df,
        table_name=f"{cfg.schema}.{GOLD_TABLE}",
        mode="overwrite",
        base_dir=cfg.local_data_dir,
        catalog=cfg.catalog,
    )

    cfg.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    csv_out = cfg.dashboard_data_dir / "latest_24h_forecasts.csv"
    gold_df.to_csv(csv_out, index=False)

    fan_path = Path(cfg.images_dir) / "cqr_24h_fan_chart.png"
    plot_probabilistic_forecast(gold_df, output_path=fan_path)
    logger.info(
        "Local pipeline complete | Gate-Val coverage=%.3f | mean_width=%.1f | saved %s",
        artifacts.gate_val_metrics["gate_val_coverage"],
        artifacts.gate_val_metrics["gate_val_mean_width"],
        fan_path,
    )
    return gold_df


def select_latest_complete_origins(features_df: pd.DataFrame, n_horizons: int) -> pd.DataFrame:
    """Rows of the newest origin per zone that has every lead time.

    Zones can end at different hours, and an origin with missing lead times would give a
    forecast with holes, so the choice is made per zone among complete origins only.
    """
    size = features_df.groupby(["zone", "origin_timestamp"])["horizon_h"].transform("size")
    complete = features_df[size == n_horizons]
    latest = complete.groupby("zone")["origin_timestamp"].transform("max")
    return (
        complete[complete["origin_timestamp"] == latest]
        .sort_values(["zone", "horizon_h"])
        .reset_index(drop=True)
    )


def run_databricks_batch_scoring(cfg: CarbonMLOpsConfig | None = None) -> str:
    """Score the newest complete origin per zone via @champion spark_udf -> gold_forecasts."""
    if not is_databricks_runtime():
        raise RuntimeError("run_databricks_batch_scoring requires a Databricks runtime.")

    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    cfg = cfg or CarbonMLOpsConfig()
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    model_uri = f"models:/{resolve_model_name(cfg)}@champion"
    feature_table = f"{cfg.catalog}.{cfg.schema}.{FEATURE_TABLE}"
    gold_table = f"{cfg.catalog}.{cfg.schema}.{GOLD_TABLE}"

    sdf = spark.table(feature_table)
    complete_origins = (
        sdf.groupBy("zone", "origin_timestamp").count().filter(F.col("count") == len(cfg.horizons))
    )
    latest_origins = complete_origins.groupBy("zone").agg(
        F.max("origin_timestamp").alias("origin_timestamp")
    )
    latest_sdf = sdf.join(latest_origins, on=["zone", "origin_timestamp"], how="inner")

    udf_cols = ["zone", *CAUSAL_FEATURE_COLS]
    cast_exprs = [
        F.col(c).cast("double").alias(c)
        if c in ("horizon_h", "target_dow", "target_month", "is_weekend")
        else F.col(c)
        for c in udf_cols
    ]

    predict_udf = mlflow.pyfunc.spark_udf(
        spark=spark,
        model_uri=model_uri,
        result_type="p05 double, p50 double, p95 double, interval_width double",
    )

    scored_sdf = (
        latest_sdf.withColumn("pred", predict_udf(F.struct(*cast_exprs)))
        .select(
            "*",
            F.col("pred.p05").alias("p05"),
            F.col("pred.p50").alias("p50"),
            F.col("pred.p95").alias("p95"),
            F.col("pred.interval_width").alias("interval_width"),
        )
        .drop("pred")
    )

    scored_sdf.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(gold_table)

    # The job's working directory is ephemeral, so the BI extract goes to a Unity Catalog
    # volume, where it can be downloaded for the Tableau packaged workbook.
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{EXPORT_VOLUME}")
    export_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/{EXPORT_VOLUME}/latest_24h_forecasts.csv"
    pdf = spark.table(gold_table).orderBy("zone", "horizon_h").toPandas()
    pdf.to_csv(export_path, index=False)
    logger.info(
        "Scored %d rows into %s via %s; extract at %s", len(pdf), gold_table, model_uri, export_path
    )
    return gold_table


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run carbon intensity probabilistic pipeline.")
    parser.add_argument("--mode", choices=["local", "databricks"], default="local")
    args = parser.parse_args()
    if args.mode == "local":
        run_local_pipeline()
    else:
        run_databricks_batch_scoring()


if __name__ == "__main__":
    main()
