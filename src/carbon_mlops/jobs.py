"""Job entry points: one function per Databricks task, each also runnable locally.

The `run_*_job` functions hold the logic and take optional arguments so tests can drive them
with synthetic data and a temporary registry. The `*_main` functions are the wheel entry
points named in pyproject.toml and databricks.yml; they take no arguments and return None,
because a console entry point's return value is used as the process exit status.

Task order on Databricks:
    carbon_training_pipeline:  ingest -> features -> train -> gate
    carbon_batch_inference:    recalibrate -> score, monitor, dashboard extracts
    carbon_qa_agent_check:     agent check (reads the Gold tables the inference job wrote)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

from carbon_mlops.agent import qa_agent
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import TRAINING_TABLE, run_feature_pipeline
from carbon_mlops.inference import batch_score
from carbon_mlops.inference.dashboard_extracts import (
    build_dashboard_extracts,
    write_dashboard_extracts,
)
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.io.store import is_databricks_runtime, read_table
from carbon_mlops.models.evaluate_gate import GateDecision, evaluate_and_promote_champion
from carbon_mlops.models.train import (
    TrainingArtifacts,
    ensure_mlflow_experiment,
    load_registered_forecaster,
    recalibrate_forecaster,
    register_forecaster,
    resolve_model_name,
    train_split_cqr,
)
from carbon_mlops.monitoring.compute_drift import (
    EXPORT_VOLUME,
    compute_monitoring,
    write_monitoring,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def read_training_table(cfg: CarbonMLOpsConfig) -> pd.DataFrame:
    """Read the horizon rows (features plus realized targets) written by the features task."""
    return read_table(
        f"{cfg.schema}.{TRAINING_TABLE}", base_dir=cfg.local_data_dir, catalog=cfg.catalog
    )


def run_training_job(
    cfg: CarbonMLOpsConfig | None = None,
    features_df: pd.DataFrame | None = None,
    n_estimators: int | None = None,
    tracking_uri: str | None = None,
) -> TrainingArtifacts:
    """Fit the model and register it as @challenger with its Gate-Val metrics."""
    cfg = cfg or CarbonMLOpsConfig()
    features = features_df if features_df is not None else read_training_table(cfg)
    return train_split_cqr(
        features,
        cfg=cfg,
        n_estimators=n_estimators,
        register_mlflow=True,
        tracking_uri=tracking_uri,
    )


def run_gate_job(
    cfg: CarbonMLOpsConfig | None = None,
    features_df: pd.DataFrame | None = None,
    client: MlflowClient | None = None,
    model_name: str | None = None,
) -> GateDecision:
    """Re-score @challenger (and @champion) on Gate-Val and promote if Gate G3 passes."""
    cfg = cfg or CarbonMLOpsConfig()
    features = features_df if features_df is not None else read_training_table(cfg)
    return evaluate_and_promote_champion(features, cfg=cfg, model_name=model_name, client=client)


def run_recalibration_job(
    cfg: CarbonMLOpsConfig | None = None,
    features_df: pd.DataFrame | None = None,
    client: MlflowClient | None = None,
) -> str:
    """Recompute @champion's conformal offsets from the latest realized errors.

    This is the scheduled recalibration the evaluation assumes. The base models are not
    refitted, so the new version predicts the same raw quantiles as the gated champion; only
    the offsets move, using targets observed up to the newest hour in the training table.
    The new version becomes @champion and records the version it was derived from.
    Returns the new model version.
    """
    cfg = cfg or CarbonMLOpsConfig()
    client = client or MlflowClient()
    features = features_df if features_df is not None else read_training_table(cfg)
    model_name = resolve_model_name(cfg)

    previous = client.get_model_version_by_alias(name=model_name, alias="champion")
    champion = load_registered_forecaster(model_name, "champion")

    labelled = features.dropna(subset=["target_carbon_intensity"])
    as_of = labelled["target_timestamp"].max() + pd.Timedelta(hours=1)
    window_start = as_of - pd.Timedelta(days=cfg.recalibration_window_days)
    recent = labelled[labelled["origin_timestamp"] >= window_start]
    recalibrated = recalibrate_forecaster(champion, recent, as_of=as_of, cfg=cfg)

    expected_cells = len(cfg.zones) * len(cfg.horizons)
    if len(recalibrated.conformal_offsets) != expected_cells:
        raise ValueError(
            f"Recalibration produced {len(recalibrated.conformal_offsets)} offset cells, "
            f"expected {expected_cells}; the window does not cover every zone and lead time"
        )

    _, new_version = register_forecaster(
        recalibrated,
        sample_df=recent,
        cfg=cfg,
        run_name="scheduled_recalibration",
        alias="champion",
        params={"calibrated_through": str(as_of)},
        tags={
            "recalibrated_from_version": str(previous.version),
            "calibrated_through": str(as_of),
            "gate_decision": "recalibration of a gated champion",
        },
    )
    logger.info(
        "Recalibrated %s v%s -> v%s through %s", model_name, previous.version, new_version, as_of
    )
    return new_version


def run_monitoring_job(
    cfg: CarbonMLOpsConfig | None = None,
    features_df: pd.DataFrame | None = None,
) -> str:
    """Compute feature drift and rolling coverage for @champion and write `gold_monitoring`.

    Uses the champion's base models and replays the recalibration schedule, so the coverage
    it reports is that of the forecasts the schedule would have issued. Returns the location
    of the CSV extract.
    """
    cfg = cfg or CarbonMLOpsConfig()
    features = features_df if features_df is not None else read_training_table(cfg)
    champion = load_registered_forecaster(resolve_model_name(cfg), "champion")
    return write_monitoring(compute_monitoring(champion, features, cfg), cfg)


def run_dashboard_extracts_job(
    cfg: CarbonMLOpsConfig | None = None,
    features_df: pd.DataFrame | None = None,
) -> list[str]:
    """Write the forecast-history and window extracts for the BI dashboards from @champion.

    Returns the locations of the CSV files.
    """
    cfg = cfg or CarbonMLOpsConfig()
    features = features_df if features_df is not None else read_training_table(cfg)
    champion = load_registered_forecaster(resolve_model_name(cfg), "champion")
    return write_dashboard_extracts(build_dashboard_extracts(champion, features, cfg), cfg)


AGENT_ANSWERS_FILE = "qa_agent_answers.json"


def run_agent_check_job(
    cfg: CarbonMLOpsConfig | None = None, out_dir: str | Path | None = None
) -> str:
    """Ask the QA agent its evaluation questions against the current tables and save the answers.

    On Databricks the agent reads the Gold tables and the answers go to the export volume, where
    they can be compared with the tables. Raises if any answer is not the expected one or
    contains a number its tool did not return. Returns the location of the answers file.
    """
    cfg = cfg or CarbonMLOpsConfig()
    ensure_mlflow_experiment()
    records = qa_agent.evaluate(cfg)
    if out_dir is None:
        if is_databricks_runtime():
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{EXPORT_VOLUME}")
            out_dir = f"/Volumes/{cfg.catalog}/{cfg.schema}/{EXPORT_VOLUME}"
        else:
            out_dir = cfg.local_data_dir
    path = Path(out_dir) / AGENT_ANSWERS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1, default=str), encoding="utf-8")
    failed = [record["id"] for record in records if not record["ok"]]
    logger.info(
        "QA agent: %d of %d answers as expected; written to %s",
        len(records) - len(failed),
        len(records),
        path,
    )
    if failed:
        raise RuntimeError(f"QA agent answers not as expected: {failed}")
    return str(path)


def ingest_main() -> None:
    _configure_logging()
    ingest_bronze_snapshot()


def features_main() -> None:
    _configure_logging()
    run_feature_pipeline()


def train_main() -> None:
    _configure_logging()
    run_training_job()


def gate_main() -> None:
    _configure_logging()
    decision = run_gate_job()
    logger.info("Gate decision: %s", decision.reason)


def recalibrate_main() -> None:
    _configure_logging()
    run_recalibration_job()


def monitor_main() -> None:
    _configure_logging()
    run_monitoring_job()


def extracts_main() -> None:
    _configure_logging()
    run_dashboard_extracts_job()


def agent_check_main() -> None:
    _configure_logging()
    run_agent_check_job()


def score_main() -> None:
    batch_score.main()
