"""Unified Delta (Databricks) and Parquet (local/CI) storage adapter."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def is_databricks_runtime() -> bool:
    """Detect whether execution is occurring inside a Databricks cluster/serverless runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _local_parquet_path(table_name: str, base_dir: Path) -> Path:
    """Map a logical table name (e.g. 'silver.features') to a deterministic local Parquet path."""
    sanitized = table_name.replace(".", "_")
    return base_dir / "lakehouse_local" / f"{sanitized}.parquet"


def write_table(
    df: pd.DataFrame,
    table_name: str,
    mode: str = "overwrite",
    base_dir: Path = Path("data"),
) -> str:
    """Persist a DataFrame to Unity Catalog Delta on Databricks or local Parquet offline."""
    if is_databricks_runtime():
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(df)
        spark_df.write.format("delta").mode(mode).saveAsTable(table_name)
        logger.info("Wrote %d rows to Delta table %s", len(df), table_name)
        return table_name

    target_path = _local_parquet_path(table_name, base_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append" and target_path.exists():
        existing = pd.read_parquet(target_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(target_path, index=False)
    logger.info("Wrote %d rows to local Parquet %s", len(df), target_path)
    return str(target_path)


def read_table(table_name: str, base_dir: Path = Path("data")) -> pd.DataFrame:
    """Read a logical table from Unity Catalog Delta on Databricks or local Parquet offline."""
    if is_databricks_runtime():
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        logger.info("Reading Delta table %s", table_name)
        return spark.table(table_name).toPandas()

    target_path = _local_parquet_path(table_name, base_dir)
    if not target_path.exists():
        raise FileNotFoundError(f"Local Parquet table not found: {target_path}")
    logger.info("Reading local Parquet %s", target_path)
    return pd.read_parquet(target_path)
