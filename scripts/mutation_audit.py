"""Breakage audit: undo one review fix at a time and confirm the test suite notices.

A fix is only protected if some test fails when the fix is broken. For every entry in
MUTATIONS this script copies `src/`, `tests/` and the Bronze snapshot into a temporary
directory, applies one deliberate breakage there, and runs pytest. The repository itself is
never modified.

Exit code 0 means every breakage was caught. Exit code 1 means at least one survived (a
missing test) or a breakage no longer applies because the code it targets has changed (the
audit needs updating). Exit code 2 means the untouched scratch copy itself fails the suite,
in which case every breakage would falsely look "caught" and nothing is audited.

Run it at the end of every review phase:

    python scripts/mutation_audit.py
    python scripts/mutation_audit.py --only C1 D2      # only issues whose ID matches
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("mutation_audit")

FEATURES = "src/carbon_mlops/features/build_features.py"
INGEST = "src/carbon_mlops/ingestion/ingest_entsoe.py"
CONFIG = "src/carbon_mlops/config.py"
TRAIN = "src/carbon_mlops/models/train.py"
CONFORMAL = "src/carbon_mlops/models/conformal.py"
WRAPPER = "src/carbon_mlops/models/pyfunc_wrapper.py"
GATE = "src/carbon_mlops/models/evaluate_gate.py"
ANALYSIS = "src/carbon_mlops/analysis/hypotheses.py"
STORE = "src/carbon_mlops/io/store.py"
JOBS = "src/carbon_mlops/jobs.py"
SCORING = "src/carbon_mlops/inference/batch_score.py"
BUNDLE = "databricks.yml"
MONITORING = "src/carbon_mlops/monitoring/compute_drift.py"
WINDOWS = "src/carbon_mlops/analysis/windows.py"
EXTRACTS = "src/carbon_mlops/inference/dashboard_extracts.py"

COPIED_PATHS: tuple[str, ...] = (
    "src",
    "tests",
    "scripts",
    "notebooks",
    "docs",
    "data/bronze_snapshot",
    "dashboards",
)
COPIED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "databricks.yml",
    "README.md",
    "Makefile",
    "LICENSE",
)


@dataclass(frozen=True)
class Mutation:
    """One deliberate breakage: replace `old` with `new` in `path` (must match exactly once)."""

    issue: str
    name: str
    path: str
    old: str
    new: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "B3",
        "lags computed on rows instead of the hourly grid",
        FEATURES,
        'df = df.set_index("timestamp_utc").reindex(grid)'
        '.rename_axis("timestamp_utc").reset_index()',
        "df = df.reset_index(drop=True)",
    ),
    Mutation(
        "B2/B4",
        "partial and incomplete hours not masked",
        FEATURES,
        'df.loc[untrusted, "carbon_intensity_g_kwh"] = np.nan',
        "pass",
    ),
    Mutation(
        "B6",
        "default ingestion skips the pinned hash",
        INGEST,
        "expected_sha256 = cfg.bronze_snapshot_sha256",
        "expected_sha256 = None",
    ),
    Mutation(
        "B6",
        "pinned hash differs from the snapshot file",
        CONFIG,
        "c3dc7796be7589eb72c5363277057b164493f909f81500f982d4f3160eea34a9",
        "c3dc7796be7589eb72c5363277057b164493f909f81500f982d4f3160eea34a0",
    ),
    Mutation(
        "B8",
        "timestamps left timezone-naive",
        INGEST,
        'out["timestamp_utc"] = ts.dt.tz_localize("UTC")',
        'out["timestamp_utc"] = ts',
    ),
    Mutation(
        "B5",
        "intensity consistency check disabled",
        INGEST,
        'if not np.allclose(recomputed, df.loc[priced, "carbon_intensity_g_kwh"], rtol=1e-9):',
        "if False:",
    ),
    Mutation(
        "C1",
        "train split not purged",
        TRAIN,
        '"train": df[target <= ends["train"]],',
        '"train": df[origin <= ends["train"]],',
    ),
    Mutation(
        "C1",
        "gate_val split not purged",
        TRAIN,
        '"gate_val": df[(origin > ends["calib"]) & (target <= ends["gate_val"])],',
        '"gate_val": df[(origin > ends["calib"]) & (origin <= ends["gate_val"])],',
    ),
    Mutation(
        "C1",
        "split overlap assertion disabled",
        TRAIN,
        "if last_target >= first_origin:",
        "if False:",
    ),
    Mutation(
        "C2",
        "G2 audit lag check disabled",
        FEATURES,
        "if not np.allclose(zone_features[col].to_numpy(), expected, equal_nan=True):\n"
        '            raise AssertionError(f"Gate G2 violation: {col} is not the value at origin',
        "if False:\n"
        '            raise AssertionError(f"Gate G2 violation: {col} is not the value at origin',
    ),
    Mutation(
        "C2",
        "G2 audit target check disabled",
        FEATURES,
        "if not np.allclose(actual_target, expected_target, equal_nan=True):",
        "if False:",
    ),
    Mutation(
        "C2",
        "1h lag built with a 2h shift",
        FEATURES,
        'df["ci_lag_1h"] = ci.shift(1)',
        'df["ci_lag_1h"] = ci.shift(2)',
    ),
    Mutation(
        "C2",
        "lag feature reads one hour into the future",
        FEATURES,
        'df["ci_lag_3h"] = ci.shift(3)',
        'df["ci_lag_3h"] = ci.shift(-1)',
    ),
    Mutation(
        "C2",
        "rolling mean uses a centred window",
        FEATURES,
        'df["ci_roll24_mean"] = ci.rolling(window=24, min_periods=12).mean()',
        'df["ci_roll24_mean"] = ci.rolling(window=24, min_periods=12, center=True).mean()',
    ),
    Mutation(
        "C2",
        "rolling std uses a centred window",
        FEATURES,
        'df["ci_roll24_std"] = ci.rolling(window=24, min_periods=12).std().fillna(0.0)',
        'df["ci_roll24_std"] = ci.rolling(window=24, min_periods=12, center=True)'
        ".std().fillna(0.0)",
    ),
    Mutation(
        "C2",
        "generation-mix feature reads three hours into the future",
        FEATURES,
        'df["wind_share_origin"] = df["wind_share"]',
        'df["wind_share_origin"] = df["wind_share"].shift(-3)',
    ),
    Mutation(
        "D10",
        "conformal quantile one rank too high",
        CONFORMAL,
        "rank = min(n, int(np.ceil((n + 1) * level - 1e-9)))",
        "rank = min(n, int(np.ceil((n + 1) * level - 1e-9)) + 1)",
    ),
    Mutation(
        "D2",
        "calibration window includes targets not yet observed",
        CONFORMAL,
        'usable = (scored_df["target_timestamp"] < as_of) & (',
        'usable = (scored_df["origin_timestamp"] < as_of) & (',
    ),
    Mutation(
        "D2",
        "calibration window has no lower bound",
        CONFORMAL,
        'scored_df["origin_timestamp"] >= window_start',
        'scored_df["origin_timestamp"] >= window_start - pd.Timedelta(days=3650)',
    ),
    Mutation(
        "D2",
        "walk-forward calibrates on the block it predicts",
        CONFORMAL,
        "scored_df, as_of=block_start, window_days=cfg.recalibration_window_days",
        "scored_df, as_of=block_end, window_days=cfg.recalibration_window_days",
    ),
    Mutation(
        "D3",
        "each tail calibrated at 90% instead of 95%",
        CONFORMAL,
        "tail_level = 1.0 - (1.0 - nominal_coverage) / 2.0",
        "tail_level = nominal_coverage",
    ),
    Mutation(
        "D3",
        "lower and upper offsets swapped",
        CONFORMAL,
        'conformal_quantile(grp["low_score"].to_numpy(), tail_level),\n'
        '            conformal_quantile(grp["high_score"].to_numpy(), tail_level),',
        'conformal_quantile(grp["high_score"].to_numpy(), tail_level),\n'
        '            conformal_quantile(grp["low_score"].to_numpy(), tail_level),',
    ),
    Mutation(
        "D2",
        "conformity score has the wrong sign",
        CONFORMAL,
        'out["low_score"] = out["raw_q05"].to_numpy() - y',
        'out["low_score"] = y - out["raw_q05"].to_numpy()',
    ),
    Mutation(
        "D5",
        "missing offset silently accepted",
        WRAPPER,
        "if missing:\n"
        '            raise ValueError(f"No conformal offsets for zone|horizon_h: {missing[:5]}")',
        "if False:\n            pass",
    ),
    Mutation(
        "D1",
        "prediction does not add the origin intensity",
        WRAPPER,
        'out.loc[grp.index, f"raw_{key}"] = pred + base',
        'out.loc[grp.index, f"raw_{key}"] = pred',
    ),
    Mutation(
        "D1",
        "training fits the level while prediction assumes a change",
        TRAIN,
        'y_tr = y_tr - z_train["carbon_intensity_origin"].to_numpy()',
        "y_tr = y_tr",
    ),
    Mutation(
        "D2",
        "rows needed by the first calibration window are not scored",
        TRAIN,
        "window_start = eval_start - pd.Timedelta(days=cfg.recalibration_window_days)",
        "window_start = eval_start - pd.Timedelta(days=cfg.recalibration_window_days - 1)",
    ),
    Mutation(
        "G3",
        "quantile crossing not repaired",
        WRAPPER,
        "stacked = np.sort(np.column_stack([p05, p50, p95]), axis=1)",
        "stacked = np.column_stack([p05, p50, p95])",
    ),
    Mutation(
        "D2",
        "gate metrics computed from other intervals than the predictions",
        TRAIN,
        'metrics = compute_interval_metrics(gate_preds, cfg, prefix="gate_val")',
        "metrics = compute_interval_metrics(\n"
        '        gate_preds.assign(p05=0.0, p95=1e9), cfg, prefix="gate_val"\n'
        "    )",
    ),
    Mutation(
        "E1",
        "per-zone coverage bounds not enforced",
        GATE,
        "if not (low <= cov <= high):",
        "if False:",
    ),
    Mutation(
        "E1",
        "gate coverage ignores the upper interval bound",
        GATE,
        "covered = (y >= p05) & (y <= p95)",
        "covered = y >= p05",
    ),
    Mutation(
        "E3",
        "persistence check not enforced",
        GATE,
        "for cell in challenger.cells_losing_to_persistence:",
        "for cell in ():",
    ),
    Mutation(
        "E3",
        "persistence comparison inverted",
        GATE,
        "if model_loss >= baseline_loss:",
        "if model_loss < baseline_loss:",
    ),
    Mutation(
        "E2",
        "champion is not re-scored on the challenger's window",
        GATE,
        "if champion_mv is not None and champion_version != challenger_version:",
        "if False:",
    ),
    Mutation(
        "E2",
        "pinball comparison against the champion not enforced",
        GATE,
        "if ratio > 1.0:",
        "if False:",
    ),
    Mutation(
        "E2",
        "pinball comparison pooled across zones instead of scale-free",
        GATE,
        "return float(np.mean(ratios))",
        "return float(\n"
        "        sum(challenger.mean_pinball_by_zone.values())\n"
        "        / sum(champion.mean_pinball_by_zone.values())\n"
        "    )",
    ),
    Mutation(
        "E4",
        "challenger promoted despite failed checks",
        GATE,
        "promoted = not failed",
        "promoted = True",
    ),
    Mutation(
        "F",
        "locked-test replay seeded from Calib instead of Gate-Val",
        TRAIN,
        'forecaster, splits["gate_val"], splits["locked_test"], cfg)',
        'forecaster, splits["calib"], splits["locked_test"], cfg)',
    ),
    Mutation(
        "F",
        "bootstrap resamples single days instead of 7-day blocks",
        ANALYSIS,
        "block = min(block_days, n_days)",
        "block = 1",
    ),
    Mutation(
        "F",
        "calibration coverage ignores the upper interval bound",
        ANALYSIS,
        'df["covered"] = (y >= df["p05"].to_numpy()) & (y <= df["p95"].to_numpy())',
        'df["covered"] = y >= df["p05"].to_numpy()',
    ),
    Mutation(
        "F",
        "H1a decision uses the wrong direction",
        ANALYSIS,
        "supported=bool(upper_99 < 0.0) if decisive else None,",
        "supported=bool(upper_99 > 0.0) if decisive else None,",
    ),
    Mutation(
        "F",
        "H1b decision uses the wrong direction",
        ANALYSIS,
        "supported=bool(lower_99 > 0.0) if decisive else None,",
        "supported=bool(lower_99 < 0.0) if decisive else None,",
    ),
    Mutation(
        "F",
        "H2 midday window taken in UTC instead of local time",
        ANALYSIS,
        'local = df["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)',
        'local = df["timestamp_utc"]',
    ),
    Mutation(
        "F",
        "H2 keeps days with missing midday hours",
        ANALYSIS,
        "wide = wide[complete]",
        "wide = wide[complete | True]",
    ),
    Mutation(
        "F",
        "H3a permutation does not shuffle",
        ANALYSIS,
        "shuffled[cols] = block[donor].reshape(len(zdf), len(cols))",
        "shuffled[cols] = block.reshape(len(zdf), len(cols))",
    ),
    Mutation(
        "F",
        "H3a reversal needs only one end",
        ANALYSIS,
        "intervals[SHORT_BUCKET][0] > 1.0 and intervals[DAY_AHEAD_BUCKET][1] < 1.0",
        "intervals[SHORT_BUCKET][0] > 1.0 or intervals[DAY_AHEAD_BUCKET][1] < 1.0",
    ),
    Mutation(
        "F",
        "H3b difference taken as short minus day-ahead",
        ANALYSIS,
        'out = (wide[DAY_AHEAD_BUCKET] - wide[SHORT_BUCKET]).rename("difference").reset_index()',
        'out = (wide[SHORT_BUCKET] - wide[DAY_AHEAD_BUCKET]).rename("difference").reset_index()',
    ),
    Mutation(
        "F",
        "H3b registered test uses all origins instead of 00:00 UTC",
        ANALYSIS,
        'midnight = width[width["origin_hour_utc"] == 0]',
        "midnight = width",
    ),
    Mutation(
        "F",
        "verdict hides disagreement between analyses",
        ANALYSIS,
        "if corrected == registered:",
        "if True:",
    ),
    Mutation(
        "G7",
        "catalog prefixed onto an already three-level table name",
        STORE,
        'if catalog and table_name.count(".") == 1:',
        "if catalog:",
    ),
    Mutation(
        "G4",
        "no fallback to the snapshot packaged in the wheel",
        INGEST,
        "if is_pinned_snapshot and not src_path.exists():",
        "if False:",
    ),
    Mutation(
        "G3",
        "recalibration starts from the challenger instead of the champion",
        JOBS,
        'champion = load_registered_forecaster(model_name, "champion")',
        'champion = load_registered_forecaster(model_name, "challenger")',
    ),
    Mutation(
        "G3",
        "recalibration stops five days before the newest observation",
        JOBS,
        'as_of = labelled["target_timestamp"].max() + pd.Timedelta(hours=1)',
        'as_of = labelled["target_timestamp"].max() - pd.Timedelta(days=5)',
    ),
    Mutation(
        "G3",
        "recalibrated model is not made the champion",
        JOBS,
        'alias="champion",',
        'alias="challenger",',
    ),
    Mutation(
        "G11",
        "scoring picks an origin with missing lead times",
        SCORING,
        "complete = features_df[size == n_horizons]",
        "complete = features_df",
    ),
    Mutation(
        "G2",
        "bundle parameters placed beside python_wheel_task",
        BUNDLE,
        '            parameters: ["--mode", "databricks"]',
        '          parameters: ["--mode", "databricks"]',
    ),
    Mutation(
        "G1",
        "a bundle task has no serverless environment",
        BUNDLE,
        "          environment_key: default\n"
        "          python_wheel_task:\n"
        "            package_name: grid_carbon_intensity_mlops\n"
        "            entry_point: carbon-train",
        "          python_wheel_task:\n"
        "            package_name: grid_carbon_intensity_mlops\n"
        "            entry_point: carbon-train",
    ),
    Mutation(
        "G3",
        "bundle names an entry point the wheel does not have",
        BUNDLE,
        "entry_point: carbon-gate",
        "entry_point: carbon-evaluate",
    ),
    Mutation(
        "G15",
        "no MLflow experiment selected when running as a Databricks job",
        TRAIN,
        "    mlflow.set_experiment(name)\n    return name",
        "    return name",
    ),
    Mutation(
        "M1",
        "PSI log ratio inverted",
        MONITORING,
        "np.log(cur_share / ref_share)",
        "np.log(ref_share / cur_share)",
    ),
    Mutation(
        "M1",
        "PSI reference includes data after the Train period",
        MONITORING,
        "train = zone_origins[when <= reference_end]",
        "train = zone_origins",
    ),
    Mutation(
        "M1",
        "PSI counts each origin once per lead time",
        MONITORING,
        'return features_df.drop_duplicates(subset=["zone", "origin_timestamp"])[',
        "return features_df[",
    ),
    Mutation(
        "M1",
        "same-season PSI reported from an incomplete reference window",
        MONITORING,
        "season_usable = len(same_season) >= min_seasonal_rows",
        "season_usable = True",
    ),
    Mutation(
        "M2",
        "coverage window selected by origin instead of observed target",
        MONITORING,
        'in_window = (zone_rows["target_timestamp"] >= date - pd.Timedelta(days=window_days)) & (\n'
        '                zone_rows["target_timestamp"] < date\n'
        "            )",
        'in_window = (zone_rows["origin_timestamp"] >= date - pd.Timedelta(days=window_days)) & (\n'
        '                zone_rows["origin_timestamp"] < date\n'
        "            )",
    ),
    Mutation(
        "M2",
        "coverage ignores the upper interval bound",
        MONITORING,
        'covered=(y >= preds_df["p05"].to_numpy()) & (y <= preds_df["p95"].to_numpy()),',
        'covered=(y >= preds_df["p05"].to_numpy()),',
    ),
    Mutation(
        "M2",
        "coverage below the band reported as ok",
        MONITORING,
        '        return "below_band"',
        '        return "ok"',
    ),
    Mutation(
        "M2",
        "skill against persistence inverted",
        MONITORING,
        "skill = 1.0 - model_loss / baseline_loss if baseline_loss > 0",
        "skill = 1.0 - baseline_loss / model_loss if baseline_loss > 0",
    ),
    Mutation(
        "W1",
        "window mean divided by the wrong length",
        WINDOWS,
        "return (totals[window_hours:] - totals[:-window_hours]) / window_hours",
        "return (totals[window_hours:] - totals[:-window_hours]) / (window_hours + 1)",
    ),
    Mutation(
        "W1",
        "the dirtiest window is recommended instead of the cleanest",
        WINDOWS,
        "return int(np.argmin(scores))",
        "return int(np.argmax(scores))",
    ),
    Mutation(
        "W1",
        "cautious ranking uses the median instead of the upper bound",
        WINDOWS,
        '{"median": "p50", "cautious": "p95"}',
        '{"median": "p50", "cautious": "p50"}',
    ),
    Mutation(
        "W2",
        "forecast history keeps every origin hour, not the daily one",
        EXTRACTS,
        'daily = preds_df[preds_df["origin_timestamp"].dt.hour == origin_hour_utc]',
        "daily = preds_df",
    ),
    Mutation(
        "W2",
        "saving against the day mean has the wrong sign",
        EXTRACTS,
        '"saving_vs_day_mean": 1.0 - realized_mean / day_mean,',
        '"saving_vs_day_mean": realized_mean / day_mean - 1.0,',
    ),
    Mutation(
        "W2",
        "the yesterday strategy reuses a day that is not the previous one",
        EXTRACTS,
        "consecutive = previous_date is not None and this_date - previous_date == pd.Timedelta(\n"
        "                days=1\n"
        "            )",
        "consecutive = previous_date is not None",
    ),
    Mutation(
        "W2",
        "strategies are compared on different sets of days",
        EXTRACTS,
        "comparable = candidates_df[has_yesterday]",
        "comparable = candidates_df",
    ),
    Mutation(
        "W5",
        "timestamps in a time zone other than UTC are written without converting",
        STORE,
        'values.dt.tz_convert("UTC")',
        "values",
    ),
    Mutation(
        "W5",
        "CSV timestamps written without the UTC offset",
        STORE,
        'UTC_TEXT_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"',
        'UTC_TEXT_FORMAT = "%Y-%m-%d %H:%M:%S"',
    ),
    Mutation(
        "W5",
        "the monitoring CSV bypasses the shared writer",
        MONITORING,
        'write_csv_extract(table, csv_path, float_format="%.6g")',
        'table.to_csv(csv_path, index=False, float_format="%.6g")',
    ),
    Mutation(
        "W6",
        "avoided emissions have the wrong sign",
        EXTRACTS,
        '"avoided_kg_per_mwh": day_mean - realized_mean,',
        '"avoided_kg_per_mwh": realized_mean - day_mean,',
    ),
    Mutation(
        "W6",
        "bad days counted as the days that were cleaner",
        EXTRACTS,
        '(grp["saving_vs_day_mean"] < 0).mean()\n                ),\n'
        '                "worst_day_saving_vs_day_mean"',
        '(grp["saving_vs_day_mean"] > 0).mean()\n                ),\n'
        '                "worst_day_saving_vs_day_mean"',
    ),
    Mutation(
        "W6",
        "worst day reported as the best day",
        EXTRACTS,
        '"worst_day_saving_vs_day_mean": float(grp["saving_vs_day_mean"].min()),',
        '"worst_day_saving_vs_day_mean": float(grp["saving_vs_day_mean"].max()),',
    ),
    Mutation(
        "W6",
        "value captured measured against the strategy itself, not hindsight",
        EXTRACTS,
        'summary["total_avoided_kg_per_daily_mwh"] / summary["hindsight_total"]',
        'summary["total_avoided_kg_per_daily_mwh"] / summary["total_avoided_kg_per_daily_mwh"]',
    ),
    Mutation(
        "W6",
        "weekly value grouped by calendar day instead of Monday week",
        EXTRACTS,
        '(day - pd.to_timedelta(day.dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d")',
        'day.dt.strftime("%Y-%m-%d")',
    ),
)


def _copy_project(repo: Path, work: Path) -> None:
    """Copy the code, tests and snapshot needed to run the suite into a scratch directory."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in COPIED_PATHS:
        shutil.copytree(repo / rel, work / rel, ignore=ignore)
    for rel in COPIED_FILES:
        shutil.copy(repo / rel, work / rel)


def baseline_passes(repo: Path) -> bool:
    """Run the suite on an unmodified scratch copy; the audit is meaningless if it fails."""
    with tempfile.TemporaryDirectory(prefix="mutation_audit_") as tmp:
        work = Path(tmp)
        _copy_project(repo, work)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        logger.error("Baseline failed in the scratch copy:\n%s", proc.stdout[-1500:])
    return proc.returncode == 0


def run_mutation(repo: Path, mutation: Mutation) -> str:
    """Apply one breakage in a scratch copy and return 'caught', 'survived' or 'stale'."""
    with tempfile.TemporaryDirectory(prefix="mutation_audit_") as tmp:
        work = Path(tmp)
        _copy_project(repo, work)
        target = work / mutation.path
        text = target.read_text(encoding="utf-8")
        if text.count(mutation.old) != 1:
            return "stale"
        target.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
    return "survived" if proc.returncode == 0 else "caught"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--only", nargs="*", default=[], help="Run only mutations whose issue ID contains a value"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()

    if not baseline_passes(repo):
        logger.error("The untouched copy does not pass the suite; fix that before auditing.")
        return 2
    logger.info("Baseline: untouched scratch copy passes the suite")

    selected = [
        m for m in MUTATIONS if not args.only or any(token in m.issue for token in args.only)
    ]
    outcomes: dict[str, list[Mutation]] = {"caught": [], "survived": [], "stale": []}
    for mutation in selected:
        outcome = run_mutation(repo, mutation)
        outcomes[outcome].append(mutation)
        logger.info("%-9s %-6s %s", outcome.upper(), mutation.issue, mutation.name)

    logger.info(
        "\n%d of %d breakages caught | %d survived | %d stale",
        len(outcomes["caught"]),
        len(selected),
        len(outcomes["survived"]),
        len(outcomes["stale"]),
    )
    for mutation in outcomes["survived"]:
        logger.error("SURVIVED (no test fails): %s %s", mutation.issue, mutation.name)
    for mutation in outcomes["stale"]:
        logger.error("STALE (pattern not found once): %s %s", mutation.issue, mutation.name)
    return 0 if not outcomes["survived"] and not outcomes["stale"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
