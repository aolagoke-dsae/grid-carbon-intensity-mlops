"""Unit tests for CarbonMLOpsConfig and the dual Delta/Parquet storage adapter (US-1.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import read_table, write_table


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
