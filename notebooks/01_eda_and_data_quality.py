# Databricks notebook source
"""Sprint 1 (US-1.2): Exploratory Data Analysis & Grid Regime Diagnostics (DE-LU vs. FR).

The Locked Holdout Test period is excluded: every plot uses only hours up to the end of the
Gate-Val split, so exploration cannot inform the hypotheses tested on the locked data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.models.train import compute_split_boundaries

matplotlib.use("Agg")


def generate_eda_diagnostics(cfg: CarbonMLOpsConfig | None = None) -> list[Path]:
    """Generate zone distribution, diurnal heatmap, and ACF lag plots into docs/images/."""
    cfg = cfg or CarbonMLOpsConfig()
    df = ingest_bronze_snapshot(cfg=cfg)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

    features = build_horizon_features(df, cfg=cfg, write_output=False)
    last_pre_test_hour = compute_split_boundaries(features, cfg=cfg)["gate_val"]
    df = df[df["timestamp_utc"] <= last_pre_test_hour].copy()
    period = f"{df['timestamp_utc'].min():%b %Y} – {df['timestamp_utc'].max():%b %Y}"
    df["hour_utc"] = df["timestamp_utc"].dt.hour
    df["month"] = df["timestamp_utc"].dt.month

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # 1. Two-panel Zone Distributions (DE-LU vs FR with independent scales)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
    zone_styles = [("DE-LU", "#d95f02"), ("FR", "#1b9e77")]
    for ax, (zone, color) in zip(axes, zone_styles, strict=True):
        vals = df.loc[df["zone"] == zone, "carbon_intensity_g_kwh"].dropna()
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.hist(
            vals, bins=45, alpha=0.65, density=True, color=color, edgecolor="white", linewidth=0.4
        )
        ax.axvline(med, color="#222222", linestyle="--", linewidth=1.5, label=f"Median: {med:.1f}")
        ax.axvspan(q25, q75, color=color, alpha=0.15, label=f"IQR: {q25:.1f}–{q75:.1f}")
        ax.set_title(f"{zone} Hourly Carbon Intensity ({period})")
        ax.set_xlabel("Carbon Intensity (gCO2eq/kWh)")
        ax.set_ylabel("Probability Density")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Regime Comparison: Variable Renewable/Fossil (DE-LU) vs. Nuclear Baseload (FR)", y=1.02
    )
    p1 = cfg.images_dir / "eda_zone_distributions.png"
    fig.tight_layout()
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    outputs.append(p1)

    # 2. Diurnal x Monthly heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
    for ax, zone in zip(axes, cfg.zones, strict=True):
        sub = df[df["zone"] == zone]
        piv = sub.pivot_table(
            index="hour_utc", columns="month", values="carbon_intensity_g_kwh", aggfunc="median"
        )
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
    for zone, color in zone_styles:
        series = (
            df.loc[df["zone"] == zone]
            .sort_values("timestamp_utc")["carbon_intensity_g_kwh"]
            .dropna()
        )
        acfs = [series.autocorr(lag=int(k)) for k in lags]
        ax.plot(lags, acfs, label=zone, color=color, linewidth=1.8)
    for mark in (24, 48, 72):
        ax.axvline(mark, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"ACF of Hourly Carbon Intensity, 1–72h Lags ({period}, pre-test data)")
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
