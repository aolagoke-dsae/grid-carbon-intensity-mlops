"""End-to-end local and batch scoring runner for 24-hour probabilistic carbon forecasts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import write_table
from carbon_mlops.models.train import train_split_cqr
from carbon_mlops.viz.diagnostics import plot_probabilistic_forecast

logger = logging.getLogger(__name__)


def run_local_pipeline(cfg: CarbonMLOpsConfig | None = None) -> pd.DataFrame:
    """Execute Bronze -> Silver -> Split-CQR -> Gold forecasts & fan chart PNG locally."""
    cfg = cfg or CarbonMLOpsConfig()
    bronze_df = ingest_bronze_snapshot(cfg=cfg)
    features_df = build_horizon_features(bronze_df, cfg=cfg, write_output=True)
    artifacts = train_split_cqr(features_df, cfg=cfg)

    gate_df = artifacts.splits["gate_val"]
    latest_origins = gate_df.groupby("zone")["origin_timestamp"].transform("max")
    latest_24h = (
        gate_df[gate_df["origin_timestamp"] == latest_origins].copy().reset_index(drop=True)
    )

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


if __name__ == "__main__":
    main()
