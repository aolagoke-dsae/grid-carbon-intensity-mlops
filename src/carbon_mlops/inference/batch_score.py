"""End-to-end local and Databricks Spark UDF batch scoring runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS, build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import is_databricks_runtime, write_table
from carbon_mlops.models.train import train_split_cqr
from carbon_mlops.viz.diagnostics import plot_probabilistic_forecast

logger = logging.getLogger(__name__)


def run_local_pipeline(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Execute Bronze -> Silver -> Split-CQR -> Gold forecasts & fan chart PNG locally."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze_df = ingest_bronze_snapshot(cfg=cfg)
    features_df = build_horizon_features(bronze_df, cfg=cfg, write_output=True)
    artifacts = train_split_cqr(features_df, cfg=cfg)

    # Splits are purged at their boundaries, so the last origins of Gate-Val keep only their
    # shortest horizons. Use the latest origin per zone that still has every horizon.
    gate_df = artifacts.splits["gate_val"]
    n_horizons = gate_df.groupby(["zone", "origin_timestamp"])["horizon_h"].transform("size")
    complete = gate_df[n_horizons == len(cfg.horizons)]
    latest_origins = complete.groupby("zone")["origin_timestamp"].transform("max")
    latest_24h = complete[complete["origin_timestamp"] == latest_origins].reset_index(drop=True)

    preds = artifacts.forecaster.predict(context=None, model_input=latest_24h)
    gold_df = pd.concat([latest_24h, preds], axis=1)

    write_table(gold_df, table_name=f"{cfg.schema}.gold_forecasts", mode="overwrite")

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


def run_databricks_batch_scoring(cfg: CarbonMLOpsConfig | None = None) -> str:
    """Score latest 24h origins on Databricks via @champion spark_udf -> gold_forecasts."""
    if not is_databricks_runtime():
        raise RuntimeError("run_databricks_batch_scoring requires a Databricks runtime.")

    import mlflow
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    cfg = cfg or CarbonMLOpsConfig()
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    model_uri = f"models:/{cfg.catalog}.{cfg.schema}.conformal_carbon_forecaster@champion"
    feature_table = f"{cfg.catalog}.{cfg.schema}.grid_horizon_features"
    gold_table = f"{cfg.catalog}.{cfg.schema}.gold_forecasts"

    sdf = spark.table(feature_table)
    latest_ts = sdf.agg(F.max("origin_timestamp")).collect()
    latest_sdf = sdf.filter(F.col("origin_timestamp") == latest_ts)

    udf_cols = ["zone", "horizon_bucket", *CAUSAL_FEATURE_COLS]
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

    scored_sdf.write.format("delta").mode("overwrite").saveAsTable(gold_table)

    pdf = scored_sdf.toPandas()
    cfg.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(cfg.dashboard_data_dir / "latest_24h_forecasts.csv", index=False)
    logger.info("Scored %d rows into %s via %s", len(pdf), gold_table, model_uri)
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
