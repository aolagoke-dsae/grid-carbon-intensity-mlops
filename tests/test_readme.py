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
    for task in ("test", "run-local", "audit"):
        assert f'"{task}"' in runner, task
        assert f"{task}:" in makefile, task
