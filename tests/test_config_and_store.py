"""Unit tests for CarbonMLOpsConfig and the dual Delta/Parquet storage adapter (US-1.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import read_table, write_csv_extract, write_table


def test_config_split_ratios_and_buckets() -> None:
    cfg = CarbonMLOpsConfig()
    assert pytest.approx(sum(cfg.split_ratios), rel=1e-6) == 1.0
    assert cfg.zones == ("DE-LU", "FR")
    assert cfg.bucket_for_horizon(1) == "short_1_6h"
    assert cfg.bucket_for_horizon(10) == "mid_7_12h"
    assert cfg.bucket_for_horizon(24) == "day_ahead_13_24h"


def test_local_parquet_roundtrip(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "zone": ["DE-LU", "FR"],
            "horizon_h": list(range(1, 3)),
            "carbon_intensity": [312.5, 48.2],
        }
    )
    written = write_table(sample, "bronze.test_carbon", mode="overwrite", base_dir=tmp_path)
    assert Path(written).exists()

    loaded = read_table("bronze.test_carbon", base_dir=tmp_path)
    pd.testing.assert_frame_equal(sample, loaded)


def test_csv_extract_writes_the_same_utc_text_with_or_without_a_time_zone(tmp_path: Path) -> None:
    """Delta hands timestamps back without a zone, Parquet with one; the CSV must not differ."""
    naive = pd.DataFrame(
        {
            "zone": ["FR", "FR"],
            "origin_timestamp": pd.to_datetime(["2025-07-03 00:00:00", "2025-07-03 13:00:00"]),
            "p50": [30.123456789, 31.0],
        }
    )
    aware = naive.assign(origin_timestamp=naive["origin_timestamp"].dt.tz_localize("UTC"))
    berlin = aware.assign(origin_timestamp=aware["origin_timestamp"].dt.tz_convert("Europe/Berlin"))

    texts = []
    for name, frame in (("naive", naive), ("aware", aware), ("berlin", berlin)):
        path = write_csv_extract(frame, tmp_path / f"{name}.csv", float_format="%.6g")
        texts.append(Path(path).read_text(encoding="utf-8"))

    assert texts[0] == texts[1] == texts[2]
    lines = texts[0].splitlines()
    assert lines[1] == "FR,2025-07-03 00:00:00+00:00,30.1235"
    assert lines[2] == "FR,2025-07-03 13:00:00+00:00,31"
    assert str(naive["origin_timestamp"].dt.tz) == "None", "the input frame must not be modified"


def test_every_csv_extract_goes_through_the_shared_writer() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "carbon_mlops"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if ".to_csv(" in path.read_text(encoding="utf-8") and path.name != "store.py"
    ]
    assert offenders == []
