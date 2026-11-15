"""Unit tests for Bronze ingestion schema & SHA-256 provenance (Sprint 1 | Gate G1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.ingestion.ingest_entsoe import compute_file_sha256, ingest_bronze_snapshot


def test_gate_g1_bronze_snapshot_provenance(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    assert cfg.bronze_snapshot_path.exists(), "Bronze Parquet snapshot must exist in repo"

    # The pinned hash is a literal in the config, not derived from the file under test.
    pinned_hash = cfg.bronze_snapshot_sha256
    assert compute_file_sha256(cfg.bronze_snapshot_path) == pinned_hash
    readme = (cfg.bronze_snapshot_path.parent / "README.md").read_text(encoding="utf-8")
    assert pinned_hash in readme, "Snapshot README must record the pinned SHA-256"

    # No expected_sha256 argument: the default path must verify against the pin by itself.
    df = ingest_bronze_snapshot(cfg=cfg, base_dir=tmp_path)

    assert len(df) == 35082
    assert set(df["zone"].unique()) == {"DE-LU", "FR"}
    assert (df["source_file_sha256"] == pinned_hash).all()
    assert df["ingested_at_utc"].notna().all()
    assert df["upstream_payload_sha256"].str.len().eq(64).all()
    assert str(df["timestamp_utc"].dt.tz) == "UTC"


def test_gate_g1_default_ingest_rejects_wrong_pin(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig(bronze_snapshot_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_bronze_snapshot(cfg=cfg, base_dir=tmp_path)


def test_gate_g1_rejects_tampered_checksum(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_bronze_snapshot(
            cfg=cfg,
            snapshot_path=cfg.bronze_snapshot_path,
            expected_sha256="0" * 64,
            base_dir=tmp_path,
        )


def test_gate_g1_rejects_unphysical_intensity(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bad_df = pd.read_parquet(cfg.bronze_snapshot_path).head(10).copy()
    bad_df.loc[0, "carbon_intensity_g_kwh"] = 9999.0
    bad_path = tmp_path / "bad_snapshot.parquet"
    bad_df.to_parquet(bad_path, index=False)

    with pytest.raises(ValueError, match="physical bounds"):
        ingest_bronze_snapshot(cfg=cfg, snapshot_path=bad_path, base_dir=tmp_path)


def test_gate_g1_rejects_inconsistent_intensity(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig()
    bad_df = pd.read_parquet(cfg.bronze_snapshot_path).head(10).copy()
    bad_df.loc[0, "carbon_intensity_g_kwh"] *= 1.05
    bad_path = tmp_path / "inconsistent_snapshot.parquet"
    bad_df.to_parquet(bad_path, index=False)

    with pytest.raises(ValueError, match="emissions_kg / priced_mwh"):
        ingest_bronze_snapshot(cfg=cfg, snapshot_path=bad_path, base_dir=tmp_path)
