"""The README must not drift from the committed results or point at files that do not exist."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def test_readme_headline_numbers_match_the_result_files() -> None:
    calibration = pd.read_csv(
        REPO_ROOT / "dashboards/data/locked_test_calibration.csv", dtype={"level": str}
    )
    for zone in ("DE-LU", "FR"):
        overall = calibration[(calibration["zone"] == zone) & (calibration["breakdown"] == "all")]
        row = overall.iloc[0]
        assert _percent(row["coverage"]) in README, zone
        interval = f"{_percent(row['coverage_ci_low'])} to {_percent(row['coverage_ci_high'])}"
        assert interval in README, zone

    def cell(zone: str, breakdown: str, level: str) -> float:
        match = calibration[
            (calibration["zone"] == zone)
            & (calibration["breakdown"] == breakdown)
            & (calibration["level"] == level)
        ]
        return float(match["coverage"].iloc[0])

    assert _percent(cell("FR", "horizon_bucket", "day_ahead_13_24h")) in README
    assert _percent(cell("DE-LU", "renewable_quartile", "Q4")) in README


def test_readme_reports_every_hypothesis_verdict_as_recorded() -> None:
    results = pd.read_csv(REPO_ROOT / "dashboards/data/hypothesis_results.csv")
    verdicts = results[results["analysis"] == "verdict"].set_index("hypothesis")["quantity"]
    assert set(verdicts.index) == {"H1", "H2", "H3"}
    for hypothesis, verdict in verdicts.items():
        expected = "Falsified" if verdict.startswith("falsified") else "Supported"
        line = next(ln for ln in README.splitlines() if ln.startswith(f"| {hypothesis} |"))
        assert f"**{expected}." in line, hypothesis


def test_readme_links_and_commands_point_at_things_that_exist() -> None:
    for target in re.findall(r"\]\(([^)#]+)\)", README):
        if target.startswith("http"):
            continue
        assert (REPO_ROOT / target).exists(), f"README links to a missing file: {target}"

    for script in re.findall(r"python ((?:notebooks|scripts)/\S+\.py)", README):
        assert (REPO_ROOT / script).exists(), script

    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert "dev" in extras and '".[dev]"' in README

    runner = (REPO_ROOT / "scripts/dev.ps1").read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for task in ("test", "run-local", "audit", "monitor", "dashboards", "slides-data", "slides"):
        assert f'"{task}"' in runner, task
        assert f"{task}:" in makefile, task


def test_recorded_plan_commit_matches_the_notebook_constant() -> None:
    import json

    record = json.loads(
        (REPO_ROOT / "dashboards/data/locked_test_run.json").read_text(encoding="utf-8")
    )
    notebook = (REPO_ROOT / "notebooks/02_cqr_calibration_analysis.py").read_text(encoding="utf-8")
    constant = re.search(r'ANALYSIS_PLAN_COMMIT = "([0-9a-f]{40})"', notebook)
    assert constant is not None
    assert record["analysis_plan_commit"] == constant.group(1)
    # The write-up and the rewrite note cite the same commit by its short ID.
    short = constant.group(1)[:7]
    assert f"`{short}`" in (REPO_ROOT / "docs/LOCKED_TEST_RESULTS.md").read_text(encoding="utf-8")
    assert f"`{short}`" in (REPO_ROOT / "docs/HISTORY_REWRITE.md").read_text(encoding="utf-8")


def test_readme_window_savings_match_the_strategy_summary() -> None:
    summary = pd.read_csv(REPO_ROOT / "dashboards/data/window_strategy_summary.csv")
    held_out = summary[(summary["period"] == "held_out") & (summary["window_hours"] == 3)]
    saving = held_out.pivot(index="zone", columns="strategy", values="mean_saving_vs_day_mean")
    for zone in ("DE-LU", "FR"):
        row = next(ln for ln in README.splitlines() if ln.startswith(f"| {zone} | 3 h |"))
        for strategy in ("median", "yesterday", "hindsight"):
            assert f"{100 * saving.loc[zone, strategy]:.1f}%" in row, (zone, strategy)
    labels = {"median": "Forecast (median)", "yesterday": "Repeat yesterday"}
    by_key = held_out.set_index(["zone", "strategy"])
    for (zone, strategy), label in ((k, labels[k[1]]) for k in by_key.index if k[1] in labels):
        row = next(ln for ln in README.splitlines() if ln.startswith(f"| {zone} | {label} |"))
        values = by_key.loc[(zone, strategy)]
        assert f"{values['mean_avoided_kg_per_mwh']:.1f} kg" in row, (zone, strategy)
        assert f"{100 * values['share_of_days_dirtier_than_day_mean']:.1f}%" in row, (
            zone,
            strategy,
        )
        assert f"{-100 * values['worst_day_saving_vs_day_mean']:.1f}% dirtier" in row, (
            zone,
            strategy,
        )
    cautious = by_key.loc[("DE-LU", "cautious")]
    assert f"({cautious['mean_avoided_kg_per_mwh']:.1f} kg)" in README
    assert f"({100 * cautious['share_of_days_dirtier_than_day_mean']:.1f}% of days)" in README
    total = by_key.loc[("DE-LU", "median"), "total_avoided_kg_per_daily_mwh"]
    assert f"about {total / 1000:.1f} t CO2" in README
    build_notes = (REPO_ROOT / "dashboards/README.md").read_text(encoding="utf-8")
    for name in (
        "forecast_history.csv",
        "window_candidates.csv",
        "window_value_weekly.csv",
        "gold_monitoring.csv",
    ):
        assert name in build_notes
        assert (REPO_ROOT / "dashboards/data" / name).exists()


def test_readme_agent_example_is_what_the_agent_says() -> None:
    import mlflow

    from carbon_mlops.agent.qa_agent import answer

    mlflow.tracing.disable()
    question = "What is the cleanest 4-hour window in Germany?"
    reply = answer(question)
    assert reply.grounded
    assert question in README
    assert reply.text in README, "the README quotes an answer the agent no longer gives"
