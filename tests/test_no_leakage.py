"""Unit tests verifying zero temporal leakage across horizon features (Sprint 1 | Gate G2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import (
    CAUSAL_FEATURE_COLS,
    audit_temporal_alignment,
    build_horizon_features,
    materialize_uc_feature_table,
    regularize_zone_to_hourly_grid,
)
from carbon_mlops.models.train import (
    assert_splits_do_not_overlap,
    chronological_four_way_split,
)


def _make_synthetic_bronze(hours: int = 72) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01 00:00:00", periods=hours, freq="h")
    rows = []
    for zone, base_ci in [("DE-LU", 350.0), ("FR", 30.0)]:
        for i, t in enumerate(ts):
            rows.append(
                {
                    "zone": zone,
                    "timestamp_utc": t,
                    "carbon_intensity_g_kwh": base_ci
                    + 25.0 * np.sin(2 * np.pi * i / 24.0)
                    + i * 0.1,
                    "wind_share": 0.30,
                    "solar_share": 0.15,
                    "renewable_share": 0.50,
                    "nuclear_share": 0.0 if zone == "DE-LU" else 0.70,
                    "fossil_share": 0.45 if zone == "DE-LU" else 0.05,
                    "gen_total_mwh": 50000.0,
                }
            )
    return pd.DataFrame(rows)


def test_gate_g2_temporal_invariants_and_lag_causality(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=72)
    features = build_horizon_features(bronze, cfg=cfg, write_output=True, base_dir=tmp_path)

    # 1. Verify Gate G2 timestamp equations
    assert (features["max_feature_timestamp"] <= features["origin_timestamp"]).all()
    expected_target = features["origin_timestamp"] + pd.to_timedelta(
        features["horizon_h"], unit="h"
    )
    assert (features["target_timestamp"] == expected_target).all()

    # 2. Verify all 24 horizons & expected feature columns exist
    assert set(features["horizon_h"].unique()) == set(range(1, 25))
    for col in CAUSAL_FEATURE_COLS:
        assert col in features.columns, f"Missing feature column {col}"

    # 3. Perturbing future targets after origin t0 must NOT change features at or before t0
    t0 = pd.Timestamp("2025-01-02 12:00:00")
    bronze_perturbed = bronze.copy()
    future_mask = bronze_perturbed["timestamp_utc"] > t0
    bronze_perturbed.loc[future_mask, "carbon_intensity_g_kwh"] = 1999.0

    features_perturbed = build_horizon_features(
        bronze_perturbed, cfg=cfg, write_output=False, base_dir=tmp_path
    )

    orig_slice = (
        features[features["origin_timestamp"] <= t0][list(CAUSAL_FEATURE_COLS)]
        .sort_index(axis=1)
        .reset_index(drop=True)
    )
    pert_slice = (
        features_perturbed[features_perturbed["origin_timestamp"] <= t0][list(CAUSAL_FEATURE_COLS)]
        .sort_index(axis=1)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(orig_slice, pert_slice)


def test_us21_materialize_uc_feature_table_offline_drops_target(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=72)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    out_path = materialize_uc_feature_table(features, cfg=cfg, base_dir=tmp_path)

    saved = pd.read_parquet(out_path)
    assert "target_carbon_intensity" not in saved.columns
    for pk in ("zone", "origin_timestamp", "horizon_h"):
        assert pk in saved.columns


def test_lags_are_hours_not_rows_when_hours_are_missing(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=96)
    gap = [pd.Timestamp("2025-01-03 05:00:00"), pd.Timestamp("2025-01-03 06:00:00")]
    gapped = bronze[~((bronze["zone"] == "FR") & bronze["timestamp_utc"].isin(gap))]

    features = build_horizon_features(gapped, cfg=cfg, write_output=False, base_dir=tmp_path)
    truth = bronze[bronze["zone"] == "FR"].set_index("timestamp_utc")["carbon_intensity_g_kwh"]
    truth = truth.drop(index=gap)

    fr = features[(features["zone"] == "FR") & (features["horizon_h"] == 1)]
    assert not fr["origin_timestamp"].isin(gap).any(), "A missing hour cannot be an origin"
    for k in (1, 3, 6, 24):
        expected = truth.reindex(fr["origin_timestamp"] - pd.Timedelta(hours=k)).to_numpy()
        actual = fr[f"ci_lag_{k}h"].to_numpy()
        assert np.allclose(actual, expected, equal_nan=True), f"ci_lag_{k}h is not origin-{k}h"

    # The hour right after the gap has no 1h-old observation, so its lag must be NaN.
    after_gap = fr[fr["origin_timestamp"] == pd.Timestamp("2025-01-03 07:00:00")]
    assert after_gap["ci_lag_1h"].isna().all()

    # Targets that fall in the gap are absent, never filled from a neighbouring hour.
    in_gap = features[(features["zone"] == "FR") & features["target_timestamp"].isin(gap)]
    assert in_gap["target_carbon_intensity"].isna().all()


def test_partial_and_incomplete_hours_are_masked(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=96)
    bronze["is_partial"] = False
    bronze["hours_covered"] = 1.0
    partial_ts = pd.Timestamp("2025-01-03 10:00:00")
    short_ts = pd.Timestamp("2025-01-03 15:00:00")
    de = bronze["zone"] == "DE-LU"
    bronze.loc[de & (bronze["timestamp_utc"] == partial_ts), "is_partial"] = True
    bronze.loc[de & (bronze["timestamp_utc"] == short_ts), "hours_covered"] = 0.5

    features = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    de_feats = features[features["zone"] == "DE-LU"]
    flagged = [partial_ts, short_ts]

    assert not de_feats["origin_timestamp"].isin(flagged).any()
    as_target = de_feats[de_feats["target_timestamp"].isin(flagged)]
    assert as_target["target_carbon_intensity"].isna().all()
    one_hour_later = de_feats[
        de_feats["origin_timestamp"].isin([t + pd.Timedelta(hours=1) for t in flagged])
    ]
    assert one_hour_later["ci_lag_1h"].isna().all()

    # The other zone is untouched.
    fr_feats = features[features["zone"] == "FR"]
    assert fr_feats["origin_timestamp"].isin(flagged).any()


def _one_zone_features(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=96)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    grid = regularize_zone_to_hourly_grid(bronze[bronze["zone"] == "DE-LU"])
    return features[features["zone"] == "DE-LU"].reset_index(drop=True), grid


def test_gate_g2_audit_detects_misaligned_lag(tmp_path: Path) -> None:
    features, grid = _one_zone_features(tmp_path)
    audit_temporal_alignment(features, grid)

    shifted = features.copy()
    shifted["ci_lag_3h"] = shifted["ci_lag_6h"]
    with pytest.raises(AssertionError, match="ci_lag_3h"):
        audit_temporal_alignment(shifted, grid)


def test_gate_g2_audit_detects_future_value_as_feature(tmp_path: Path) -> None:
    features, grid = _one_zone_features(tmp_path)
    leaked = features.copy()
    leaked["carbon_intensity_origin"] = leaked["target_carbon_intensity"]
    with pytest.raises(AssertionError, match="carbon_intensity_origin"):
        audit_temporal_alignment(leaked, grid)


def test_gate_g2_audit_detects_wrong_target(tmp_path: Path) -> None:
    features, grid = _one_zone_features(tmp_path)
    wrong = features.copy()
    wrong["target_carbon_intensity"] = wrong["carbon_intensity_origin"]
    with pytest.raises(AssertionError, match="target is not the value"):
        audit_temporal_alignment(wrong, grid)


def test_splits_are_purged_at_every_boundary(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=480)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    splits = chronological_four_way_split(features, cfg=cfg)

    names = ["train", "calib", "gate_val", "locked_test"]
    assert all(len(splits[n]) > 0 for n in names)
    for earlier, later in zip(names[:-1], names[1:], strict=True):
        assert splits[earlier]["target_timestamp"].max() < splits[later]["origin_timestamp"].min()

    # No (zone, target hour) label is shared between any two splits.
    seen: set[tuple[str, pd.Timestamp]] = set()
    for n in names:
        labels = set(zip(splits[n]["zone"], splits[n]["target_timestamp"], strict=True))
        assert not (labels & seen), f"{n} shares target hours with an earlier split"
        seen |= labels

    # The locked test keeps every horizon of every one of its origins.
    per_origin = splits["locked_test"].groupby(["zone", "origin_timestamp"]).size()
    assert per_origin.max() == len(cfg.horizons)


def test_split_overlap_check_fails_on_unpurged_splits(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bronze = _make_synthetic_bronze(hours=480)
    features = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    labelled = features.dropna(subset=["target_carbon_intensity"])
    cut = labelled["origin_timestamp"].sort_values().iloc[len(labelled) // 2]
    unpurged = {
        "train": labelled[labelled["origin_timestamp"] <= cut],
        "calib": labelled[labelled["origin_timestamp"] > cut],
    }
    with pytest.raises(AssertionError, match="Split leakage"):
        assert_splits_do_not_overlap(unpurged)
