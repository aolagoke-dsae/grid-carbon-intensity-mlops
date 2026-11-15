"""Unit tests for the H1-H3 analysis (docs/ANALYSIS_PLAN_H1_H3.md), on synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carbon_mlops.analysis.hypotheses import (
    CALENDAR_FEATURES,
    DAY_AHEAD_BUCKET,
    FEATURE_GROUPS,
    ORIGIN_STATE_FEATURES,
    SHORT_BUCKET,
    bootstrap_ratio,
    calibration_table,
    daily_lead_time_differences,
    derive_verdicts,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3a,
    evaluate_h3b,
    midday_daily_table,
    moving_block_day_samples,
    percentile_interval,
    permutation_loss_increase,
)
from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS, build_horizon_features
from carbon_mlops.models.conformal import (
    add_conformity_scores,
    walk_forward_calibrated_predictions,
)
from carbon_mlops.models.train import evaluate_on_locked_test, train_split_cqr
from tests.test_no_leakage import _make_synthetic_bronze

CFG = CarbonMLOpsConfig()


def _prediction_frame(
    n_days: int = 12,
    zones: tuple[str, ...] = ("DE-LU", "FR"),
    start: str = "2025-03-01",
) -> pd.DataFrame:
    """One row per (zone, hourly origin, lead time) with neutral placeholder values."""
    origins = pd.date_range(start, periods=24 * n_days, freq="h", tz="UTC")
    horizons = np.arange(1, 25)
    frame = pd.MultiIndex.from_product(
        [zones, origins, horizons], names=["zone", "origin_timestamp", "horizon_h"]
    ).to_frame(index=False)
    frame["target_timestamp"] = frame["origin_timestamp"] + pd.to_timedelta(
        frame["horizon_h"], unit="h"
    )
    frame["horizon_bucket"] = frame["horizon_h"].map(CFG.bucket_for_horizon)
    for col in CAUSAL_FEATURE_COLS:
        if col != "horizon_h":
            frame[col] = 0.0
    hour = frame["target_timestamp"].dt.hour
    frame["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["target_carbon_intensity"] = 100.0
    frame["carbon_intensity_origin"] = 100.0
    frame["p05"], frame["p50"], frame["p95"] = 90.0, 100.0, 110.0
    frame["interval_width"] = frame["p95"] - frame["p05"]
    return frame


def _bronze_for(frame: pd.DataFrame, share_by_hour: pd.Series | None = None) -> pd.DataFrame:
    """Minimal Bronze rows covering every target hour of a prediction frame."""
    hours = frame[["zone", "target_timestamp"]].drop_duplicates()
    bronze = hours.rename(columns={"target_timestamp": "timestamp_utc"}).reset_index(drop=True)
    if share_by_hour is None:
        position = bronze.groupby("zone").cumcount()
        bronze["renewable_share"] = 0.2 + 0.6 * (position % 97) / 96.0
    else:
        bronze["renewable_share"] = bronze["timestamp_utc"].map(share_by_hour).to_numpy()
    return bronze


# ----------------------------------------------------------------------------- bootstrap


def test_block_bootstrap_resamples_whole_runs_of_days() -> None:
    samples = moving_block_day_samples(n_days=30, n_resamples=50, block_days=7, seed=1)
    assert samples.shape == (50, 30)
    assert samples.min() >= 0 and samples.max() <= 29
    # Inside each 7-day block the days are consecutive.
    steps = np.diff(samples, axis=1)
    inside_block = np.ones(29, dtype=bool)
    inside_block[6::7] = False
    assert (steps[:, inside_block] == 1).all()
    # Deterministic for a fixed seed, different for another seed.
    assert np.array_equal(samples, moving_block_day_samples(30, 50, 7, seed=1))
    assert not np.array_equal(samples, moving_block_day_samples(30, 50, 7, seed=2))
    # Fewer days than one block: every resample is the whole period.
    short = moving_block_day_samples(n_days=3, n_resamples=4, block_days=7, seed=1)
    assert (short == np.array([0, 1, 2])).all()
    with pytest.raises(ValueError, match="at least one day"):
        moving_block_day_samples(0)


def test_bootstrap_ratio_and_interval() -> None:
    samples = moving_block_day_samples(n_days=10, n_resamples=200, block_days=3, seed=0)
    covered = np.full(10, 9.0)
    total = np.full(10, 10.0)
    draws = bootstrap_ratio(covered, total, samples)
    assert np.allclose(draws, 0.9)
    assert percentile_interval(draws) == pytest.approx((0.9, 0.9))
    # A day without rows contributes nothing; an all-empty resample is undefined, not 0.
    assert np.isnan(bootstrap_ratio(np.zeros(10), np.zeros(10), samples)).all()
    low, high = percentile_interval(np.array([np.nan, 1.0, 3.0]))
    assert (low, high) == pytest.approx((1.05, 2.95))
    # Infinite draws (a ratio with a zero denominator) give infinite bounds, not NaN.
    assert percentile_interval(np.full(50, np.inf)) == (np.inf, np.inf)
    mixed = percentile_interval(np.array([1.0] * 10 + [np.inf] * 90))
    assert mixed == (1.0, np.inf)


# ----------------------------------------------------------------------------- calibration


def test_calibration_table_matches_hand_computed_values() -> None:
    frame = _prediction_frame(n_days=6, zones=("FR",))
    frame["target_carbon_intensity"] = 100.0
    below = frame["horizon_h"] == 1
    above = frame["horizon_h"].isin([23, 24])
    frame.loc[below, "target_carbon_intensity"] = 80.0
    frame.loc[above, "target_carbon_intensity"] = 125.0
    table = calibration_table(frame, _bronze_for(frame), CFG, n_resamples=100)

    overall = table[(table["breakdown"] == "all")].iloc[0]
    assert overall["zone"] == "FR"
    assert overall["n_rows"] == len(frame)
    assert overall["coverage"] == pytest.approx(21 / 24)
    assert overall["share_below_p05"] == pytest.approx(1 / 24)
    assert overall["share_above_p95"] == pytest.approx(2 / 24)
    assert overall["mean_width"] == pytest.approx(20.0)
    assert overall["coverage_ci_low"] <= overall["coverage"] <= overall["coverage_ci_high"]
    assert bool(overall["inside_reference_band"]) is True

    by_lead = table[table["breakdown"] == "horizon_h"].set_index("level")
    assert by_lead.loc["1", "coverage"] == 0.0
    assert by_lead.loc["2", "coverage"] == 1.0
    assert bool(by_lead.loc["1", "inside_reference_band"]) is False
    # Median pinball: |100 - 80| * 0.5 at lead 1; persistence equals the median here.
    assert by_lead.loc["1", "pinball_p50"] == pytest.approx(10.0)
    assert by_lead.loc["1", "persistence_pinball_p50"] == pytest.approx(10.0)

    buckets = table[table["breakdown"] == "horizon_bucket"].set_index("level")
    assert buckets.loc[SHORT_BUCKET, "coverage"] == pytest.approx(5 / 6)
    assert buckets.loc[DAY_AHEAD_BUCKET, "coverage"] == pytest.approx(10 / 12)
    assert set(table["breakdown"]) == {
        "all",
        "horizon_bucket",
        "horizon_h",
        "renewable_quartile",
        "month",
        "target_hour_utc",
    }


# ----------------------------------------------------------------------------- H1


def _h1_frame(width_top: float, width_bottom: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _prediction_frame(n_days=16, zones=("DE-LU",))
    hours = pd.Series(frame["target_timestamp"].unique()).sort_values().reset_index(drop=True)
    # Alternate days of high and low renewable share so both regimes span the period.
    high_day = (hours.dt.dayofyear % 2 == 0).to_numpy()
    share = pd.Series(np.where(high_day, 0.85, 0.15), index=hours.to_numpy())
    share.iloc[::5] = 0.5  # a middle group so the quartiles are well defined
    bronze = _bronze_for(frame, share_by_hour=share)

    target_share = frame["target_timestamp"].map(share).to_numpy()
    top, bottom = target_share >= 0.85, target_share <= 0.15
    frame.loc[top, "target_carbon_intensity"] = 150.0
    frame.loc[bottom, "target_carbon_intensity"] = 500.0
    frame["p50"] = frame["target_carbon_intensity"]
    half = np.where(top, width_top, np.where(bottom, width_bottom, 50.0)) / 2
    frame["p05"], frame["p95"] = frame["p50"] - half, frame["p50"] + half
    frame["interval_width"] = frame["p95"] - frame["p05"]
    return frame, bronze


def _decisions(rows: list[dict], hypothesis: str, analysis: str) -> dict[str, bool]:
    return {
        r["part"]: r["supported"]
        for r in rows
        if r["hypothesis"] == hypothesis
        and r["analysis"] == analysis
        and r["supported"] is not None
    }


def test_h1_supported_when_high_renewable_hours_are_cleaner_and_wider() -> None:
    frame, bronze = _h1_frame(width_top=120.0, width_bottom=60.0)
    rows = evaluate_h1(frame, bronze, CFG, n_resamples=300)

    assert _decisions(rows, "H1", "corrected") == {"a": True, "b": True}
    assert _decisions(rows, "H1", "registered") == {"a": True, "b": True}
    median_row = next(r for r in rows if r["analysis"] == "corrected" and r["part"] == "a")
    assert median_row["estimate"] == pytest.approx(150.0 - 500.0)
    width_row = next(r for r in rows if r["analysis"] == "corrected" and r["part"] == "b")
    assert width_row["estimate"] == pytest.approx(120.0 - 60.0)

    # Each target hour is counted once, not once per lead time.
    share = bronze.set_index("timestamp_utc")["renewable_share"]
    n_top, n_bottom = (int(x) for x in median_row["n"].split("/"))
    assert n_top == int((share >= 0.85).sum())
    assert n_bottom == int((share <= 0.15).sum())
    assert n_top + n_bottom < frame["target_timestamp"].nunique()


def test_h1_width_claim_fails_when_high_renewable_hours_are_narrower() -> None:
    frame, bronze = _h1_frame(width_top=60.0, width_bottom=120.0)
    rows = evaluate_h1(frame, bronze, CFG, train_thresholds=(0.15, 0.85), n_resamples=300)

    assert _decisions(rows, "H1", "corrected") == {"a": True, "b": False}
    assert _decisions(rows, "H1", "registered") == {"a": True, "b": False}
    # Sensitivity rows are reported but never decide anything.
    sensitivity = [r for r in rows if r["analysis"] == "sensitivity"]
    assert len(sensitivity) == 2
    assert all(r["supported"] is None for r in sensitivity)
    context = {r["quantity"]: r["estimate"] for r in rows if r["analysis"] == "context"}
    assert context["relative width W/p50, top quartile"] == pytest.approx(60.0 / 150.0)
    assert context["relative width W/p50, bottom quartile"] == pytest.approx(120.0 / 500.0)


# ----------------------------------------------------------------------------- H2


def _midday_bronze(de_level: float, fr_level: float) -> pd.DataFrame:
    hours = pd.date_range("2025-06-25", "2025-07-10 23:00", freq="h", tz="UTC")
    rows = []
    for zone, level in (("DE-LU", de_level), ("FR", fr_level)):
        for ts in hours:
            windy = ts.day % 2 == 0
            rows.append(
                {
                    "zone": zone,
                    "timestamp_utc": ts,
                    "carbon_intensity_g_kwh": level + ts.hour,
                    "wind_share": 0.5 if windy else 0.1,
                    "solar_share": 0.2,
                    "is_partial": False,
                    "hours_covered": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_midday_window_follows_local_time_across_daylight_saving() -> None:
    summer = pd.date_range("2025-07-01", "2025-07-01 23:00", freq="h", tz="UTC")
    winter = pd.date_range("2025-12-01", "2025-12-01 23:00", freq="h", tz="UTC")
    rows = [
        {
            "zone": zone,
            "timestamp_utc": ts,
            "carbon_intensity_g_kwh": float(ts.hour),
            "wind_share": 0.3,
            "solar_share": 0.1,
        }
        for zone in ("DE-LU", "FR")
        for ts in summer.append(winter)
    ]
    bronze = pd.DataFrame(rows)
    daily = midday_daily_table(bronze, summer.min(), winter.max()).set_index("local_date")

    # Local 11:00-14:00 is 09:00-12:00 UTC in summer and 10:00-13:00 UTC in winter.
    assert daily.loc[pd.Timestamp("2025-07-01").date(), "de_lu"] == pytest.approx(10.5)
    assert daily.loc[pd.Timestamp("2025-12-01").date(), "de_lu"] == pytest.approx(11.5)

    # A day missing one of its four hours in either zone is dropped.
    missing = bronze[
        ~(
            (bronze["zone"] == "FR")
            & (bronze["timestamp_utc"] == pd.Timestamp("2025-07-01 10:00", tz="UTC"))
        )
    ]
    kept = midday_daily_table(missing, summer.min(), winter.max())
    assert list(kept["local_date"]) == [pd.Timestamp("2025-12-01").date()]


def test_h2_is_decided_by_the_daily_midday_gap() -> None:
    bronze = _midday_bronze(de_level=20.0, fr_level=60.0)
    start, end = bronze["timestamp_utc"].min(), bronze["timestamp_utc"].max()
    rows = evaluate_h2(bronze, start, end, prior_year_threshold=0.0)
    registered = next(r for r in rows if r["analysis"] == "registered")
    count = next(r for r in rows if r["analysis"] == "corrected")
    assert registered["supported"] is True
    assert registered["estimate"] == pytest.approx(-40.0)
    assert count["estimate"] == 8.0  # the 8 windy (even) days of the 16
    assert count["n"] == "8 of 16 days"
    assert next(r for r in rows if r["analysis"] == "sensitivity")["supported"] is None

    reverse = evaluate_h2(_midday_bronze(de_level=300.0, fr_level=30.0), start, end)
    assert next(r for r in reverse if r["analysis"] == "registered")["supported"] is False
    assert next(r for r in reverse if r["analysis"] == "corrected")["estimate"] == 0.0


# ----------------------------------------------------------------------------- H3a


class _HorizonSwitchModel:
    """Median follows the origin intensity up to `switch_h`, the target hour beyond it."""

    def __init__(self, switch_h: int) -> None:
        self.switch_h = switch_h

    def predict_raw_quantiles(self, model_input: pd.DataFrame) -> pd.DataFrame:
        from_origin = model_input["carbon_intensity_origin"].to_numpy(dtype=float)
        from_calendar = 100.0 + 50.0 * model_input["target_hour_sin"].to_numpy(dtype=float)
        short = model_input["horizon_h"].to_numpy() <= self.switch_h
        median = np.where(short, from_origin, from_calendar)
        return pd.DataFrame(
            {"raw_q05": median, "raw_q50": median, "raw_q95": median}, index=model_input.index
        )


def _h3a_frame(model: _HorizonSwitchModel) -> pd.DataFrame:
    frame = _prediction_frame(n_days=10)
    origin_hour = frame["origin_timestamp"].dt.hour + 24 * frame["origin_timestamp"].dt.day
    frame["carbon_intensity_origin"] = 200.0 + 3.0 * origin_hour
    frame["target_carbon_intensity"] = model.predict_raw_quantiles(frame)["raw_q50"].to_numpy()
    frame["p50"] = frame["target_carbon_intensity"]
    # Drop one lead time of one origin: incomplete origins must be excluded entirely.
    first = frame["origin_timestamp"].min()
    return frame[~((frame["origin_timestamp"] == first) & (frame["horizon_h"] == 7))]


def test_feature_groups_partition_the_model_features() -> None:
    grouped = set(ORIGIN_STATE_FEATURES) | set(CALENDAR_FEATURES)
    assert set(ORIGIN_STATE_FEATURES).isdisjoint(CALENDAR_FEATURES)
    assert grouped == set(CAUSAL_FEATURE_COLS) - {"horizon_h"}
    with pytest.raises(ValueError, match="horizon_h is fixed"):
        permutation_loss_increase(
            _HorizonSwitchModel(6), _prediction_frame(n_days=2), {"bad": ("horizon_h",)}, 1
        )


def test_permutation_importance_attributes_loss_to_the_group_that_matters() -> None:
    model = _HorizonSwitchModel(switch_h=12)
    frame = _h3a_frame(model)
    out = permutation_loss_increase(model, frame, FEATURE_GROUPS, n_repeats=3)

    assert out.groupby(["zone", "origin_timestamp"]).size().eq(24).all()
    assert frame["origin_timestamp"].min() not in set(out["origin_timestamp"])
    short, long = out["horizon_h"] <= 12, out["horizon_h"] > 12
    # Up to the switch only origin-state matters; beyond it only the calendar.
    assert out.loc[short, "inc_origin_state"].mean() > 1.0
    assert np.allclose(out.loc[short, "inc_calendar"], 0.0)
    assert out.loc[long, "inc_calendar"].mean() > 1.0
    assert np.allclose(out.loc[long, "inc_origin_state"], 0.0)
    # The original predictions are perfect, so every increase is a pure loss.
    assert (out[["inc_origin_state", "inc_calendar"]] >= 0).all().all()


def test_h3a_reversal_needs_both_ends() -> None:
    reversing = _HorizonSwitchModel(switch_h=12)
    rows = evaluate_h3a(
        reversing, _h3a_frame(reversing), CFG, n_resamples=200, n_repeats=2, n_single_repeats=1
    )
    reversal = [r for r in rows if r["analysis"] == "corrected" and r["supported"] is not None]
    assert len(reversal) == 2
    assert all(r["supported"] is True for r in reversal)

    # Origin-state dominates at every lead time: no reversal, so not supported.
    persistent = _HorizonSwitchModel(switch_h=24)
    rows = evaluate_h3a(
        persistent, _h3a_frame(persistent), CFG, n_resamples=200, n_repeats=2, n_single_repeats=1
    )
    reversal = [r for r in rows if r["analysis"] == "corrected" and r["supported"] is not None]
    assert all(r["supported"] is False for r in reversal)
    registered = [r for r in rows if r["analysis"] == "registered" and r["supported"] is not None]
    assert all(r["supported"] is False for r in registered)


# ----------------------------------------------------------------------------- H3b


def test_daily_lead_time_difference_is_day_ahead_minus_short() -> None:
    frame = _prediction_frame(n_days=2, zones=("FR",))
    frame["value"] = frame["horizon_h"].astype(float)
    first = frame["origin_timestamp"].min()
    frame = frame[~((frame["origin_timestamp"] == first) & (frame["horizon_h"] == 24))]
    out = daily_lead_time_differences(frame, "value")

    # mean(13..24) - mean(1..6) = 18.5 - 3.5
    assert np.allclose(out["difference"], 15.0)
    assert first not in set(out["origin_timestamp"])
    assert len(out) == 47


def test_h3b_corrected_uses_error_over_all_origins_not_width_at_midnight() -> None:
    frame = _prediction_frame(n_days=14)
    rng = np.random.default_rng(3)
    growth = 1.0 + frame["horizon_h"].to_numpy() / 4.0
    noise = rng.normal(0.0, 1.0, len(frame)) * growth
    frame["target_carbon_intensity"] = frame["p50"] + noise
    # Width shrinks with lead time, the opposite of the realized error.
    frame["interval_width"] = 60.0 - frame["horizon_h"]
    rows = evaluate_h3b(frame, CFG)

    corrected = [r for r in rows if r["analysis"] == "corrected"]
    assert {r["zone"] for r in corrected} == {"DE-LU", "FR"}
    assert all(r["supported"] is True for r in corrected)
    assert all(int(r["n"]) == 14 for r in corrected)
    pooled = next(r for r in rows if r["analysis"] == "registered" and r["zone"] == "pooled")
    assert pooled["supported"] is False
    assert pooled["estimate"] == pytest.approx(-15.0)
    assert int(pooled["n"]) == 28  # 14 midnight origins in each of two zones


# ----------------------------------------------------------------------------- verdicts


def _decision_rows(hypothesis: str, corrected: bool, registered: bool) -> list[dict]:
    return [
        {"hypothesis": hypothesis, "analysis": "corrected", "supported": corrected},
        {"hypothesis": hypothesis, "analysis": "registered", "supported": registered},
        {"hypothesis": hypothesis, "analysis": "context", "supported": None},
    ]


def test_verdict_rule_flags_disagreement_between_analyses() -> None:
    rows = (
        _decision_rows("H1", corrected=True, registered=True)
        + _decision_rows("H2", corrected=False, registered=False)
        + _decision_rows("H3", corrected=True, registered=False)
        + [{"hypothesis": "H1", "analysis": "corrected", "supported": True}]
    )
    verdicts = derive_verdicts(pd.DataFrame(rows)).set_index("hypothesis")["quantity"]
    assert verdicts["H1"] == "supported"
    assert verdicts["H2"].startswith("falsified (descriptive")
    assert verdicts["H3"] == "sensitive to method"

    # One failed part of a two-part hypothesis falsifies it.
    partial = _decision_rows("H1", corrected=True, registered=True) + [
        {"hypothesis": "H1", "analysis": "corrected", "supported": False},
        {"hypothesis": "H1", "analysis": "registered", "supported": False},
    ]
    assert derive_verdicts(pd.DataFrame(partial))["quantity"].iloc[0] == "falsified"


# ----------------------------------------------------------------------------- locked test


def test_locked_test_replay_is_seeded_from_gate_val_and_stays_causal(tmp_path: Path) -> None:
    # The window must reach back past the purged day at the Gate-Val/Test boundary.
    cfg = CarbonMLOpsConfig(recalibration_window_days=4, recalibration_step_days=1)
    bronze = _make_synthetic_bronze(hours=480)
    feats = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    artifacts = train_split_cqr(feats, cfg=cfg, n_estimators=5)
    splits = artifacts.splits
    test_df, gate_df = splits["locked_test"], splits["gate_val"]

    scored, preds = evaluate_on_locked_test(artifacts.forecaster, splits, cfg)

    assert len(preds) == len(test_df)
    assert preds["origin_timestamp"].min() == test_df["origin_timestamp"].min()
    assert (preds["calibrated_as_of"] <= preds["origin_timestamp"]).all()
    # History before the test comes from Gate-Val only, never from Train or Calib.
    before = scored[scored["origin_timestamp"] < test_df["origin_timestamp"].min()]
    assert before["origin_timestamp"].min() >= gate_df["origin_timestamp"].min()
    assert before["origin_timestamp"].max() <= gate_df["origin_timestamp"].max()
    assert len(before) > 0

    reference_history = add_conformity_scores(
        artifacts.forecaster, pd.concat([gate_df, test_df], ignore_index=True)
    )
    reference = walk_forward_calibrated_predictions(
        reference_history,
        test_df["origin_timestamp"].min(),
        test_df["origin_timestamp"].max(),
        cfg=cfg,
    )
    keys = ["zone", "origin_timestamp", "horizon_h"]
    left = preds.sort_values(keys).reset_index(drop=True)
    right = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[["p05", "p50", "p95"]], right[["p05", "p50", "p95"]])
