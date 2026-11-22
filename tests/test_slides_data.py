"""The slide data in dashboards/slides must match the project's current results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDES = REPO_ROOT / "dashboards" / "slides"
REFRESH = "run `make slides-data` (on Windows: `scripts/dev.ps1 slides-data`) and commit the result"


def _export_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_deck_data", SLIDES / "export_deck_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = _export_module()


def _load() -> dict:
    return json.loads((SLIDES / "deck_data.json").read_text(encoding="utf-8"))


def _whole_suite(request: pytest.FixtureRequest) -> bool:
    """True when pytest is running every test, so that its count is the project's test count."""
    config = request.config
    selected = (
        config.getoption("keyword") or config.getoption("markexpr") or config.getoption("deselect")
    )
    return not selected and all(Path(arg).is_dir() for arg in config.args)


def test_slide_data_matches_the_current_results(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    count = request.session.testscollected if _whole_suite(request) else None
    reasons = EXPORT.stale_reasons(_load(), tests_collected=count)
    assert not reasons, (
        f"dashboards/slides/deck_data.json is out of date; {REFRESH}:\n" + "\n".join(reasons)
    )


def test_a_changed_result_table_is_noticed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(REPO_ROOT)
    data = _load()
    copy = tmp_path / "data"
    copy.mkdir()
    for name in EXPORT.TABLES:
        (copy / name).write_bytes((EXPORT.DATA / name).read_bytes())
    monkeypatch.setattr(EXPORT, "DATA", copy)
    # The same tables in another folder are a different source list, so every table is reported.
    assert len(EXPORT.stale_reasons(data)) == 2 * len(EXPORT.TABLES)

    data = _load()
    monkeypatch.setattr(EXPORT, "DATA", Path("dashboards/data"))
    assert EXPORT.stale_reasons(data) == []
    name = "dashboards/data/window_strategy_summary.csv"
    data["sources"][name] = "0" * 64
    assert EXPORT.stale_reasons(data) == [f"{name} has changed"]


def test_a_changed_counter_or_register_figure_is_noticed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    data = _load()
    data["mlops"]["breakages"] -= 1
    data["facts"]["configurations"] += 1
    data["backup"]["review_open"] = data["backup"]["review_open"][:-1]
    reasons = EXPORT.stale_reasons(data, tests_collected=data["mlops"]["tests"] + 1)
    assert len(reasons) == 4
    assert any(reason.startswith("the test count") for reason in reasons)


def test_line_endings_do_not_change_a_table_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    unix, windows = tmp_path / "unix", tmp_path / "windows"
    for folder, ending in ((unix, b"\n"), (windows, b"\r\n")):
        folder.mkdir()
        for name in EXPORT.TABLES:
            raw = (EXPORT.DATA / name).read_bytes().replace(b"\r\n", b"\n")
            (folder / name).write_bytes(raw.replace(b"\n", ending))
    prints = []
    for folder in (unix, windows):
        monkeypatch.setattr(EXPORT, "DATA", folder)
        prints.append(
            sorted(v for k, v in EXPORT.source_fingerprint().items() if k.endswith(".csv"))
        )
    assert prints[0] == prints[1]


def test_local_speaker_notes_quote_the_current_figures() -> None:
    """Guards a local, untracked notes file; skipped wherever that file does not exist."""
    notes_path = REPO_ROOT / "dashboards" / "design" / "followup_answers.md"
    if not notes_path.exists():
        pytest.skip("the local speaker notes are not part of the repository")
    notes = " ".join(notes_path.read_text(encoding="utf-8").split())
    data = _load()
    tests, breakages = data["mlops"]["tests"], data["mlops"]["breakages"]
    items, n_open = data["backup"]["review_items"], len(data["backup"]["review_open"])
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()

    expected = [
        f"{tests} tests, {breakages} deliberate breakages, {items} register items, {n_open} open",
        f"now all {breakages} are caught",
        f"from 28 cases to {breakages}",
        f"{words[n_open].capitalize()} findings are open",
    ]
    calibration = pd.read_csv(REPO_ROOT / "dashboards/data/locked_test_calibration.csv", dtype=str)
    for zone, breakdown, level in (
        ("FR", "horizon_bucket", "day_ahead_13_24h"),
        ("DE-LU", "renewable_quartile", "Q4"),
    ):
        row = calibration[
            (calibration["zone"] == zone)
            & (calibration["breakdown"] == breakdown)
            & (calibration["level"] == level)
        ]
        expected.append(f"{100 * float(row['coverage'].iloc[0]):.1f}%")
    missing = [text for text in expected if text not in notes]
    assert not missing, f"the speaker notes no longer match the project: {missing}"
