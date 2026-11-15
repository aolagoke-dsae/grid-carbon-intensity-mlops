"""Probabilistic fan chart and horizon calibration diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def plot_probabilistic_forecast(
    forecast_df: pd.DataFrame,
    output_path: Path,
    title: str = "24-Hour Probabilistic Grid Carbon Intensity Forecast (90% Split-CQR Interval)",
) -> Path:
    """Render side-by-side DE-LU and FR 24-hour fan charts [p05, p50, p95] vs realized intensity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zones = [z for z in ("DE-LU", "FR") if z in forecast_df["zone"].unique()]
    if not zones:
        zones = sorted(forecast_df["zone"].unique().tolist())

    fig, axes = plt.subplots(1, len(zones), figsize=(6 * len(zones), 4.4), dpi=150)
    if len(zones) == 1:
        axes = [axes]

    palette = {"DE-LU": "#d95f02", "FR": "#1b9e77"}

    for ax, zone in zip(axes, zones, strict=True):
        sub = forecast_df[forecast_df["zone"] == zone].sort_values("horizon_h")
        color = palette.get(zone, "#2b8cbe")
        h = sub["horizon_h"]

        ax.fill_between(
            h,
            sub["p05"],
            sub["p95"],
            color=color,
            alpha=0.22,
            label="90% CQR Interval [p05, p95]",
        )
        ax.plot(h, sub["p50"], color=color, linewidth=2.0, label="Median Forecast (p50)")

        if (
            "target_carbon_intensity" in sub.columns
            and sub["target_carbon_intensity"].notna().any()
        ):
            ax.plot(
                h,
                sub["target_carbon_intensity"],
                color="#222222",
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=1.2,
                label="Realized Intensity",
            )

        origin_lbl = str(
            sub["origin_timestamp"].iloc.item() if False else sub["origin_timestamp"].to_numpy()[0]
        )
        ax.set_title(f"{zone} — Origin: {origin_lbl}")
        ax.set_xlabel("Forecast Lead Time (Hours Ahead)")
        ax.set_ylabel("Carbon Intensity (gCO2eq/kWh)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, frameon=True)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_coverage_by_horizon(
    calibration_df: pd.DataFrame,
    output_path: Path,
    nominal: float = 0.90,
    reference_band: tuple[float, float] = (0.87, 0.96),
) -> Path:
    """Plot empirical coverage per lead time and zone with its block-bootstrap interval."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_lead = calibration_df[calibration_df["breakdown"] == "horizon_h"].copy()
    by_lead["horizon_h"] = by_lead["level"].astype(int)
    palette = {"DE-LU": "#d95f02", "FR": "#1b9e77"}

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    ax.axhspan(*reference_band, color="#888888", alpha=0.12, label="Reference band")
    ax.axhline(nominal, color="#222222", linestyle="--", linewidth=1.2, label="Nominal 90%")
    for zone, grp in by_lead.groupby("zone", sort=True):
        grp = grp.sort_values("horizon_h")
        color = palette.get(zone, "#2b8cbe")
        ax.fill_between(
            grp["horizon_h"],
            grp["coverage_ci_low"],
            grp["coverage_ci_high"],
            color=color,
            alpha=0.2,
        )
        ax.plot(
            grp["horizon_h"], grp["coverage"], color=color, marker="o", markersize=3, label=zone
        )
    ax.set_title("Empirical Coverage of the 90% Interval by Lead Time (95% block-bootstrap band)")
    ax.set_xlabel("Forecast Lead Time (Hours Ahead)")
    ax.set_ylabel("Empirical Coverage")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
