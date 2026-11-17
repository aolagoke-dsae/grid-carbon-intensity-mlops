"""Unit tests for the dashboard extracts and the window ranking (US-3.3), synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carbon_mlops.analysis.windows import best_start, hindsight_ranks, window_means
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.inference.dashboard_extracts import (
    HISTORY_COLUMNS,
    forecast_history_table,
    strategy_summary_table,
    window_candidates_table,
    window_value_weekly_table,
    write_dashboard_extracts,
)

CFG = CarbonMLOpsConfig()
UTC = "UTC"


# ----------------------------------------------------------------------------- window ranking


def test_window_means_and_best_start_match_hand_computed_values() -> None:
    values = np.array([10.0, 2.0, 4.0, 6.0, 1.0, 1.0])
    assert window_means(values, 1).tolist() == values.tolist()
    assert window_means(values, 2).tolist() == [6.0, 3.0, 5.0, 3.5, 1.0]
    assert window_means(values, 3).tolist() == pytest.approx([16 / 3, 4.0, 11 / 3, 8 / 3])
    assert window_means(values, 6).tolist() == [4.0]
    assert best_start(window_means(values, 2)) == 4
    # A tie goes to the earliest start.
    assert best_start(np.array([3.0, 1.0, 1.0, 2.0])) == 1

    with pytest.raises(ValueError, match="window_hours"):
        window_means(values, 7)
    with pytest.raises(ValueError, match="window_hours"):
        window_means(values, 0)
    with pytest.raises(ValueError, match="missing scores"):
        best_start(np.array([1.0, np.nan]))


def test_hindsight_ranks_start_at_one_and_share_ties() -> None:
    assert hindsight_ranks(np.array([5.0, 1.0, 3.0, 1.0])).tolist() == [4, 1, 3, 1]


# ----------------------------------------------------------------------------- forecast history


def _predictions(days: int = 3, zones: tuple[str, ...] = ("FR",)) -> pd.DataFrame:
    """Walk-forward style rows for hourly origins; p50 equals the lead time."""
    origins = pd.date_range("2025-09-10", periods=24 * days, freq="h", tz=UTC)
    frame = pd.MultiIndex.from_product(
        [zones, origins, np.arange(1, 25)], names=["zone", "origin_timestamp", "horizon_h"]
    ).to_frame(index=False)
    frame["target_timestamp"] = frame["origin_timestamp"] + pd.to_timedelta(
        frame["horizon_h"], unit="h"
    )
    frame["p50"] = frame["horizon_h"].astype(float)
    frame["p05"], frame["p95"] = frame["p50"] - 2.0, frame["p50"] + 2.0
    frame["interval_width"] = 4.0
    frame["target_carbon_intensity"] = frame["p50"] + 1.0
    frame["calibrated_as_of"] = pd.Timestamp("2025-09-08", tz=UTC)
    return frame


def test_forecast_history_keeps_one_complete_midnight_forecast_per_day() -> None:
    preds = _predictions(days=3)
    # Remove one lead time of the second day's midnight origin: that day must be dropped.
    second_midnight = pd.Timestamp("2025-09-11 00:00", tz=UTC)
    preds = preds[~((preds["origin_timestamp"] == second_midnight) & (preds["horizon_h"] == 5))]
    validation_end = pd.Timestamp("2025-09-10 23:00", tz=UTC)

    history = forecast_history_table(preds, validation_end, CFG)

    assert list(history.columns) == list(HISTORY_COLUMNS)
    assert history["forecast_date"].unique().tolist() == ["2025-09-10", "2025-09-12"]
    assert history.groupby("forecast_date").size().eq(24).all()
    assert (history["origin_timestamp"].dt.hour == 0).all()
    by_date = history.groupby("forecast_date")["period"].first()
    assert by_date["2025-09-10"] == "validation"
    assert by_date["2025-09-12"] == "held_out"
    # The first forecast hour of a midnight origin is 01:00, the last is 00:00 next day.
    first_day = history[history["forecast_date"] == "2025-09-10"]
    assert first_day["target_hour_utc"].tolist() == [*range(1, 24), 0]
    assert first_day["realized"].tolist() == [h + 1.0 for h in range(1, 25)]
    assert first_day["covered"].all()

    outside = preds.assign(target_carbon_intensity=preds["p95"] + 0.5)
    assert not forecast_history_table(outside, validation_end, CFG)["covered"].any()


# ----------------------------------------------------------------------------- candidates


def _history(day_profiles: dict[str, tuple[list[float], list[float], list[float]]]) -> pd.DataFrame:
    """History rows for one zone from {date: (p50, p95, realized)}, each a 24-hour profile."""
    rows = []
    for date, (p50, p95, realized) in day_profiles.items():
        origin = pd.Timestamp(date, tz=UTC)
        for h in range(1, 25):
            rows.append(
                {
                    "zone": "FR",
                    "forecast_date": date,
                    "period": "held_out",
                    "origin_timestamp": origin,
                    "horizon_h": h,
                    "target_timestamp": origin + pd.Timedelta(hours=h),
                    "p50": p50[h - 1],
                    "p95": p95[h - 1],
                    "realized": realized[h - 1],
                }
            )
    return pd.DataFrame(rows)


def _dip(at: int, depth: float = 60.0, base: float = 100.0) -> list[float]:
    """A flat 24-hour profile with a 3-hour dip starting at position `at` (0-based)."""
    profile = [base] * 24
    for i in range(at, at + 3):
        profile[i] = base - depth
    return profile


def test_window_candidates_flag_each_strategy_and_score_it() -> None:
    # Day 1: forecast dip at 10, truth dips at 10 too (forecast is right).
    # Day 2: median says 4, upper bound says 15, truth dips at 15 (the cautious view is right).
    day1 = (_dip(10), _dip(10), _dip(10))
    day2 = (_dip(4), _dip(15), _dip(15))
    candidates = window_candidates_table(_history({"2025-10-01": day1, "2025-10-02": day2}), (3,))

    assert len(candidates) == 2 * 22
    assert (candidates["n_windows"] == 22).all()
    day = candidates.set_index(["forecast_date", "start_horizon_h"])

    def chosen(date: str, flag: str) -> int:
        rows = candidates[(candidates["forecast_date"] == date) & candidates[flag]]
        assert len(rows) == 1
        return int(rows["start_horizon_h"].iloc[0])

    # start_horizon_h is 1-based: a dip at position 10 starts at lead time 11.
    assert chosen("2025-10-01", "recommended_median") == 11
    assert chosen("2025-10-01", "best_in_hindsight") == 11
    assert chosen("2025-10-02", "recommended_median") == 5
    assert chosen("2025-10-02", "recommended_cautious") == 16
    assert chosen("2025-10-02", "best_in_hindsight") == 16
    # "Yesterday" repeats day 1's cleanest start on day 2 and has nothing to go on for day 1.
    assert chosen("2025-10-02", "recommended_yesterday") == 11
    assert not candidates[candidates["forecast_date"] == "2025-10-01"][
        "recommended_yesterday"
    ].any()

    # Day mean of a flat-100 profile with a 3-hour dip to 40 is (21 * 100 + 3 * 40) / 24 = 92.5.
    best = day.loc[("2025-10-01", 11)]
    assert best["day_mean_realized"] == pytest.approx(92.5)
    assert best["realized_mean"] == pytest.approx(40.0)
    assert best["saving_vs_day_mean"] == pytest.approx(1 - 40.0 / 92.5)
    assert best["regret"] == 0.0
    # Intensity in g/kWh equals kg/MWh: moving 1 MWh into the dip avoids 92.5 - 40 = 52.5 kg.
    assert best["avoided_kg_per_mwh"] == pytest.approx(52.5)
    assert best["hindsight_rank"] == 1
    assert best["start_timestamp"] == pd.Timestamp("2025-10-01 11:00", tz=UTC)
    assert best["end_timestamp"] == pd.Timestamp("2025-10-01 14:00", tz=UTC)
    # On day 2 the median pick misses the dip entirely: flat 100 against a best of 40.
    wrong = day.loc[("2025-10-02", 5)]
    assert wrong["realized_mean"] == pytest.approx(100.0)
    assert wrong["regret"] == pytest.approx(60.0)
    assert wrong["saving_vs_day_mean"] == pytest.approx(1 - 100.0 / 92.5)
    assert wrong["saving_vs_day_mean"] < 0
    assert wrong["avoided_kg_per_mwh"] == pytest.approx(-7.5)


def test_yesterday_strategy_needs_the_previous_calendar_day() -> None:
    profile = (_dip(6), _dip(6), _dip(6))
    gap = window_candidates_table(_history({"2025-10-01": profile, "2025-10-03": profile}), (3,))
    assert not gap["recommended_yesterday"].any()


def test_strategy_summary_compares_strategies_on_the_same_days() -> None:
    day1 = (_dip(10), _dip(10), _dip(10))
    day2 = (_dip(4), _dip(15), _dip(15))
    day3 = (_dip(15), _dip(15), _dip(15))
    candidates = window_candidates_table(
        _history({"2025-10-01": day1, "2025-10-02": day2, "2025-10-03": day3}), (3,)
    )
    summary = strategy_summary_table(candidates).set_index("strategy")

    # Day 1 has no "yesterday" choice, so every strategy is judged on days 2 and 3 only.
    assert (summary["n_days"] == 2).all()
    saving_hit = 1 - 40.0 / 92.5
    saving_miss = 1 - 100.0 / 92.5
    assert summary.loc["hindsight", "mean_saving_vs_day_mean"] == pytest.approx(saving_hit)
    assert summary.loc["hindsight", "mean_regret"] == 0.0
    assert summary.loc["hindsight", "share_of_days_best_window"] == 1.0
    # Median: misses day 2, hits day 3. Cautious: hits both.
    assert summary.loc["median", "mean_saving_vs_day_mean"] == pytest.approx(
        (saving_miss + saving_hit) / 2
    )
    assert summary.loc["median", "share_of_days_cleaner_than_day_mean"] == 0.5
    assert summary.loc["median", "mean_regret"] == pytest.approx(30.0)
    assert summary.loc["cautious", "mean_regret"] == 0.0
    # Yesterday: day 2 repeats day 1's start (10, a miss); day 3 repeats day 2's (15, a hit).
    assert summary.loc["yesterday", "share_of_days_best_window"] == 0.5

    # Business value. A hit avoids 92.5 - 40 = 52.5 kg per MWh; a miss costs 100 - 92.5 = 7.5.
    assert summary.loc["hindsight", "mean_avoided_kg_per_mwh"] == pytest.approx(52.5)
    assert summary.loc["hindsight", "total_avoided_kg_per_daily_mwh"] == pytest.approx(105.0)
    assert summary.loc["median", "mean_avoided_kg_per_mwh"] == pytest.approx(22.5)
    assert summary.loc["median", "total_avoided_kg_per_daily_mwh"] == pytest.approx(45.0)
    assert summary.loc["median", "share_of_days_dirtier_than_day_mean"] == 0.5
    assert summary.loc["median", "worst_day_saving_vs_day_mean"] == pytest.approx(saving_miss)
    assert summary.loc["cautious", "share_of_days_dirtier_than_day_mean"] == 0.0
    assert summary.loc["cautious", "worst_day_saving_vs_day_mean"] == pytest.approx(saving_hit)
    assert summary.loc["hindsight", "share_of_hindsight_value_captured"] == pytest.approx(1.0)
    assert summary.loc["median", "share_of_hindsight_value_captured"] == pytest.approx(45 / 105)
    assert summary.loc["cautious", "share_of_hindsight_value_captured"] == pytest.approx(1.0)


def test_weekly_value_groups_comparable_days_by_monday_week() -> None:
    hit = (_dip(10), _dip(10), _dip(10))
    miss = (_dip(4), _dip(4), _dip(10))  # the forecast points at the wrong hours
    # Saturday 4 Oct has no "yesterday"; Sunday 5 Oct closes the week of 29 Sep; then Mon, Tue.
    candidates = window_candidates_table(
        _history({"2025-10-04": hit, "2025-10-05": hit, "2025-10-06": miss, "2025-10-07": hit}),
        (3,),
    )
    weekly = window_value_weekly_table(candidates).set_index(["week_start", "strategy"])

    assert sorted(weekly.index.get_level_values("week_start").unique()) == [
        "2025-09-29",
        "2025-10-06",
    ]
    assert (weekly.loc["2025-09-29", "n_days"] == 1).all()
    assert (weekly.loc["2025-10-06", "n_days"] == 2).all()
    assert weekly.loc[("2025-09-29", "median"), "mean_avoided_kg_per_mwh"] == pytest.approx(52.5)
    # Week of 6 Oct: the median misses on Monday (-7.5) and hits on Tuesday (52.5).
    assert weekly.loc[("2025-10-06", "median"), "mean_avoided_kg_per_mwh"] == pytest.approx(22.5)
    assert weekly.loc[("2025-10-06", "median"), "share_of_days_dirtier_than_day_mean"] == 0.5
    assert weekly.loc[("2025-10-06", "hindsight"), "mean_avoided_kg_per_mwh"] == pytest.approx(52.5)
    assert weekly.loc[("2025-10-06", "yesterday"), "share_of_days_dirtier_than_day_mean"] == 0.0


def test_write_dashboard_extracts_writes_csv_and_tables(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig(local_data_dir=tmp_path, dashboard_data_dir=tmp_path / "dash")
    profile = (_dip(6), _dip(6), _dip(6))
    candidates = window_candidates_table(_history({"2025-10-01": profile, "2025-10-02": profile}))
    tables = {
        "forecast_history.csv": _history({"2025-10-01": profile}),
        "window_candidates.csv": candidates,
        "window_strategy_summary.csv": strategy_summary_table(candidates),
        "window_value_weekly.csv": window_value_weekly_table(candidates),
    }
    written = write_dashboard_extracts(tables, cfg)

    assert [Path(p).name for p in written] == list(tables)
    for path, table in zip(written, tables.values(), strict=True):
        assert len(pd.read_csv(path)) == len(table)
    assert (tmp_path / "lakehouse_local" / "carbon_mlops_gold_window_candidates.parquet").exists()
    # All four window lengths are present by default.
    assert sorted(candidates["window_hours"].unique().tolist()) == [2, 3, 4, 6]
