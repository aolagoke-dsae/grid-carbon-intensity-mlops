"""Smoke test verifying diagnostics.py renders probabilistic fan charts (Sprint 1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from carbon_mlops.viz.diagnostics import plot_probabilistic_forecast


def test_plot_probabilistic_forecast_renders_png(tmp_path: Path) -> None:
    horizons = list(range(1, 25))
    rows = []
    origin = pd.Timestamp("2025-09-01 00:00:00")
    for zone, base in [("DE-LU", 300.0), ("FR", 28.0)]:
        for h in horizons:
            rows.append(
                {
                    "zone": zone,
                    "origin_timestamp": origin,
                    "horizon_h": h,
                    "p05": base - 20.0,
                    "p50": base,
                    "p95": base + 25.0,
                    "target_carbon_intensity": base + 3.0,
                }
            )
    out_file = tmp_path / "fan_chart.png"
    rendered = plot_probabilistic_forecast(pd.DataFrame(rows), output_path=out_file)
    assert rendered.exists()
    assert rendered.stat().st_size > 1000
