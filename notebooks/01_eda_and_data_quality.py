# Databricks notebook source
"""Sprint 1 (US-1.2): Exploratory Data Analysis & Grid Regime Diagnostics (DE-LU vs. FR)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot

matplotlib.use("Agg")


def generate_eda_diagnostics(cfg: CarbonMLOpsConfig | None = None) -> list[Path]:
    """Generate zone distribution, diurnal heatmap, and ACF lag plots into docs/images/."""
    cfg = cfg or CarbonMLOpsConfig()
    df = ingest_bronze_snapshot(cfg=cfg)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["hour_utc"] = df["timestamp_utc"].dt.hour
    df["month"] = df["timestamp_utc"].dt.month

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # 1. Zone distributions (DE-LU vs FR)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for zone, color in [("DE-LU", "#d95f02"), ("FR", "#1b9e77")]:
        vals = df.loc[df["zone"] == zone, "carbon_intensity_g_kwh"].dropna()
        ax.hist(vals, bins=50, alpha=0.55, density=True, label=f"{zone} (median={vals.median():.1f})", color=color)
    ax.set_title("Hourly Grid Carbon Intensity Distribution: DE-LU vs. FR (2024–2025)")
    ax.set_xlabel("Carbon Intensity (gCO2eq/kWh)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)
    p1 = cfg.images_dir / "eda_zone_distributions.png"
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    outputs.append(p1)

    # 2. Diurnal x Monthly heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
    for ax, zone in zip(axes, cfg.zones, strict=True):
        sub = df[df["zone"] == zone]
        piv = sub.pivot_table(index="hour_utc", columns="month", values="carbon_intensity_g_kwh", aggfunc="median")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_title(f"{zone} Median Carbon Intensity (Hour x Month)")
        ax.set_xlabel("Month")
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(m) for m in range(1, 13)])
        fig.colorbar(im, ax=ax, label="gCO2eq/kWh")
    axes[0].set_ylabel("Hour (UTC)")
    p2 = cfg.images_dir / "eda_diurnal_heatmap.png"
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    outputs.append(p2)

    # 3. Autocorrelation up to 72h lags
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    lags = np.arange(1, 73)
    for zone, color in [("DE-LU", "#d95f02"), ("FR", "#1b9e77")]:
        series = df.loc[df["zone"] == zone].sort_values("timestamp_utc")["carbon_intensity_g_kwh"].dropna()
        acfs = [series.autocorr(lag=int(k)) for k in lags]
        ax.plot(lags, acfs, label=zone, color=color, linewidth=1.8)
    for mark in (24, 48, 72):
        ax.axvline(mark, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Autocorrelation Function (ACF) of Hourly Carbon Intensity (1–72h Lags)")
    ax.set_xlabel("Lag (Hours)")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.grid(alpha=0.25)
    p3 = cfg.images_dir / "eda_acf_lags.png"
    fig.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    outputs.append(p3)

    return outputs


if __name__ == "__main__":
    paths = generate_eda_diagnostics()
    for p in paths:
        print(f"Exported {p}")
