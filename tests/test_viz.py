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


def test_plot_coverage_by_horizon_renders_png(tmp_path: Path) -> None:
    from carbon_mlops.viz.diagnostics import plot_coverage_by_horizon

    rows = [
        {
            "zone": zone,
            "breakdown": "horizon_h",
            "level": str(h),
            "coverage": 0.90 - 0.001 * h,
            "coverage_ci_low": 0.86,
            "coverage_ci_high": 0.93,
        }
        for zone in ("DE-LU", "FR")
        for h in range(1, 25)
    ]
    rows.append({"zone": "FR", "breakdown": "all", "level": "all", "coverage": 0.9})
    out = plot_coverage_by_horizon(pd.DataFrame(rows), tmp_path / "coverage.png")
    assert out.exists()
    assert out.stat().st_size > 5000
