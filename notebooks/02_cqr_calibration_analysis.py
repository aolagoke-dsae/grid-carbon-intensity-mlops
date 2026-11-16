# Databricks notebook source
"""Sprint 3 (US-3.1): locked-test calibration and H1-H3, per docs/ANALYSIS_PLAN_H1_H3.md.

READ THE PLAN FIRST. This notebook is the only place that scores the Locked Test. The plan
allows one execution; `--check` recomputes the same numbers to prove they are reproducible
and changes nothing.

    python notebooks/02_cqr_calibration_analysis.py --run      # the single execution
    python notebooks/02_cqr_calibration_analysis.py --check    # recompute and compare
    python notebooks/02_cqr_calibration_analysis.py --dry-run-gate-val --output-dir <dir>

`--dry-run-gate-val` runs the identical code on Gate-Val and writes to a directory of your
choice. It exists so the code can be exercised without touching the Locked Test.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from carbon_mlops.analysis.hypotheses import (
    ALPHA,
    BLOCK_DAYS,
    BOOTSTRAP_SEED,
    N_PERMUTATION_REPEATS,
    N_RESAMPLES,
    AnalysisOutputs,
    midday_daily_table,
    run_hypothesis_analysis,
)
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import compute_file_sha256, normalize_timestamps_to_utc
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster
from carbon_mlops.models.train import (
    DEFAULT_LGBM_PARAMS,
    chronological_four_way_split,
    evaluate_on_gate_val,
    evaluate_on_locked_test,
    fit_quantile_models,
)
from carbon_mlops.viz.diagnostics import plot_coverage_by_horizon

logger = logging.getLogger("locked_test_analysis")

# Commit of docs/ANALYSIS_PLAN_H1_H3.md that was approved before this notebook existed.
ANALYSIS_PLAN_COMMIT = "6c8e73032a5107c61ae5d6cbf06069b22e27fdb9"

CALIBRATION_CSV = "locked_test_calibration.csv"
RESULTS_CSV = "hypothesis_results.csv"
RUN_RECORD_JSON = "locked_test_run.json"
COVERAGE_FIGURE = "locked_test_coverage_by_horizon.png"
FLOAT_FORMAT = "%.6g"


def compute_analysis(
    cfg: CarbonMLOpsConfig, evaluation_split: str
) -> tuple[AnalysisOutputs, dict[str, object]]:
    """Fit the frozen model on Train, walk forward over one split, and run the analysis."""
    if evaluation_split not in ("locked_test", "gate_val"):
        raise ValueError(f"Unknown evaluation split: {evaluation_split}")

    snapshot_sha = compute_file_sha256(cfg.bronze_snapshot_path)
    if snapshot_sha != cfg.bronze_snapshot_sha256:
        raise ValueError("Bronze snapshot does not match the pinned SHA-256 (Gate G1)")
    bronze = normalize_timestamps_to_utc(pd.read_parquet(cfg.bronze_snapshot_path))
    features = build_horizon_features(bronze, cfg=cfg, write_output=False)
    splits = chronological_four_way_split(features, cfg=cfg)

    # Section 2 of the plan: base models are fitted on Train only and never refitted.
    lgbm_params = dict(DEFAULT_LGBM_PARAMS)
    forecaster = ConformalCarbonForecaster(
        models_by_zone=fit_quantile_models(splits["train"], cfg, lgbm_params),
        conformal_offsets={},
        target_mode=cfg.target_mode,
    )
    replay = evaluate_on_locked_test if evaluation_split == "locked_test" else evaluate_on_gate_val
    _, predictions = replay(forecaster, splits, cfg)

    # Sensitivity thresholds are fixed from data that precedes the evaluated period.
    train_end = splits["train"]["target_timestamp"].max()
    train_de = bronze[(bronze["zone"] == "DE-LU") & (bronze["timestamp_utc"] <= train_end)]
    renewable_q25, renewable_q75 = train_de["renewable_share"].quantile([0.25, 0.75])
    period_start = predictions["target_timestamp"].min()
    period_end = predictions["target_timestamp"].max()
    one_year = pd.DateOffset(years=1)
    prior_year = midday_daily_table(bronze, period_start - one_year, period_end - one_year)
    prior_year_vre = float(prior_year["de_lu_vre_share"].median())

    outputs = run_hypothesis_analysis(
        forecaster,
        predictions,
        bronze,
        cfg,
        train_renewable_thresholds=(float(renewable_q25), float(renewable_q75)),
        prior_year_vre_threshold=prior_year_vre,
    )
    record: dict[str, object] = {
        "analysis_plan_commit": ANALYSIS_PLAN_COMMIT,
        "evaluation_split": evaluation_split,
        "bronze_snapshot_sha256": snapshot_sha,
        "first_origin_utc": str(predictions["origin_timestamp"].min()),
        "last_origin_utc": str(predictions["origin_timestamp"].max()),
        "n_forecast_rows": int(len(predictions)),
        "n_recalibrations": int(predictions["calibrated_as_of"].nunique()),
        "model": {
            "target_mode": cfg.target_mode,
            "lgbm_params": lgbm_params,
            "fitted_on": "train",
            "recalibration_window_days": cfg.recalibration_window_days,
            "recalibration_step_days": cfg.recalibration_step_days,
            "nominal_coverage": cfg.nominal_coverage,
        },
        "statistics": {
            "alpha": ALPHA,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_block_days": BLOCK_DAYS,
            "bootstrap_resamples": N_RESAMPLES,
            "permutation_repeats": N_PERMUTATION_REPEATS,
            "train_renewable_q25_q75": [float(renewable_q25), float(renewable_q75)],
            "prior_year_midday_vre_median": prior_year_vre,
        },
    }
    return outputs, record


def write_outputs(
    outputs: AnalysisOutputs, record: dict[str, object], data_dir: Path, images_dir: Path
) -> list[Path]:
    """Write both CSV files, the run record and the coverage figure."""
    data_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = data_dir / CALIBRATION_CSV
    results_path = data_dir / RESULTS_CSV
    record_path = data_dir / RUN_RECORD_JSON
    outputs.calibration.to_csv(calibration_path, index=False, float_format=FLOAT_FORMAT)
    outputs.results.to_csv(results_path, index=False, float_format=FLOAT_FORMAT)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_path = plot_coverage_by_horizon(outputs.calibration, images_dir / COVERAGE_FIGURE)
    return [calibration_path, results_path, record_path, figure_path]


def check_outputs(outputs: AnalysisOutputs, record: dict[str, object], data_dir: Path) -> None:
    """Fail unless a fresh computation reproduces the committed files exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        write_outputs(outputs, record, fresh_dir, fresh_dir)
        for name in (CALIBRATION_CSV, RESULTS_CSV, RUN_RECORD_JSON):
            committed = (data_dir / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            fresh = (fresh_dir / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            if committed != fresh:
                raise AssertionError(f"{name} is not reproduced by a fresh computation")
            logger.info("Reproduced %s", name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Locked-test calibration and H1-H3 analysis.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Score the Locked Test and write results")
    mode.add_argument("--check", action="store_true", help="Recompute and compare with the files")
    mode.add_argument(
        "--dry-run-gate-val", action="store_true", help="Run the same code on Gate-Val instead"
    )
    parser.add_argument("--output-dir", type=Path, help="Required with --dry-run-gate-val")
    args = parser.parse_args()
    cfg = CarbonMLOpsConfig()

    if args.dry_run_gate_val:
        if args.output_dir is None:
            parser.error("--dry-run-gate-val needs --output-dir (never the dashboards folder)")
        if args.output_dir.resolve() == cfg.dashboard_data_dir.resolve():
            parser.error("--output-dir must not be the dashboards data folder")
        outputs, record = compute_analysis(cfg, "gate_val")
        for path in write_outputs(outputs, record, args.output_dir, args.output_dir):
            logger.info("Dry run wrote %s", path)
        check_outputs(outputs, record, args.output_dir)
        return

    outputs, record = compute_analysis(cfg, "locked_test")
    if args.check:
        check_outputs(outputs, record, cfg.dashboard_data_dir)
        logger.info("Committed locked-test results are reproduced exactly")
        return
    for path in write_outputs(outputs, record, cfg.dashboard_data_dir, cfg.images_dir):
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
