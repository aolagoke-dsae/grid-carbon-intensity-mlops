"""Breakage audit: undo one review fix at a time and confirm the test suite notices.

A fix is only protected if some test fails when the fix is broken. For every entry in
MUTATIONS this script copies `src/`, `tests/` and the Bronze snapshot into a temporary
directory, applies one deliberate breakage there, and runs pytest. The repository itself is
never modified.

Exit code 0 means every breakage was caught. Exit code 1 means at least one survived (a
missing test) or a breakage no longer applies because the code it targets has changed (the
audit needs updating). Run it at the end of every review phase:

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

COPIED_PATHS: tuple[str, ...] = ("src", "tests", "data/bronze_snapshot")
COPIED_FILES: tuple[str, ...] = ("pyproject.toml",)


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
        "Calib rows needed by the first calibration window are not scored",
        TRAIN,
        "window_start = gate_start - pd.Timedelta(days=cfg.recalibration_window_days)",
        "window_start = gate_start - pd.Timedelta(days=cfg.recalibration_window_days - 1)",
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
)


def _copy_project(repo: Path, work: Path) -> None:
    """Copy the code, tests and snapshot needed to run the suite into a scratch directory."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in COPIED_PATHS:
        shutil.copytree(repo / rel, work / rel, ignore=ignore)
    for rel in COPIED_FILES:
        shutil.copy(repo / rel, work / rel)


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
