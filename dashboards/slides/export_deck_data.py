"""Collect every number the slides show into one JSON file, and check that it is still current.

Run from the repository root with the project's environment:

    python dashboards/slides/export_deck_data.py

Every figure is read from a tracked file or computed by the package. Two workspace facts are the
exception; they are recorded in docs/DATABRICKS_VERIFICATION.md. The file also records a SHA-256
of each input it was built from, so that `stale_reasons` (used by tests/test_slides_data.py) can
tell when a result has changed since the last export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot
from carbon_mlops.models.train import DEFAULT_LGBM_PARAMS, chronological_four_way_split

DATA = Path("dashboards/data")
OUT = Path("dashboards/slides/deck_data.json")
REGISTER = Path("docs/REVIEW_ISSUES.md")
AUDIT = Path("scripts/mutation_audit.py")
ZONES = ("DE-LU", "FR")
TABLES = (
    "forecast_history.csv",
    "gold_monitoring.csv",
    "hypothesis_results.csv",
    "locked_test_calibration.csv",
    "window_strategy_summary.csv",
    "window_value_weekly.csv",
)


def r(x: float, d: int = 3) -> float:
    return round(float(x), d)


def source_fingerprint(cfg: CarbonMLOpsConfig | None = None) -> dict[str, str]:
    """A SHA-256 of every data file the export reads.

    Text tables are hashed with line endings normalised, so a Windows checkout and a Linux one
    agree. The snapshot is binary and hashed as it is.
    """
    cfg = cfg or CarbonMLOpsConfig()
    prints = {
        f"{DATA.as_posix()}/{name}": hashlib.sha256(
            (DATA / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in TABLES
    }
    snapshot = Path(cfg.bronze_snapshot_path)
    prints[snapshot.as_posix()] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return prints


def register_figures() -> dict[str, Any]:
    """The figures the slides quote from the findings register, read out of its text."""
    register = REGISTER.read_text(encoding="utf-8")
    items = re.findall(r"^\| ([A-Z]\d+[a-z]?) \|(.*)\|\s*$", register, flags=re.M)
    open_ids = [
        i for i, rest in items
        if rest.split("|")[-1].strip().lower().startswith(("open", "finding, open"))
    ]
    static = re.search(r"Locked-test coverage after purging: DE-LU ([\d.]+)%, FR ([\d.]+)%", register)
    size_rows = re.findall(
        r"^\| \**(\d+) trees(?: \(chosen\))?\** \| \**([\d.]+)%\** \| \**([\d.]+)%\** \| \**([\d.]+)\** \| \**([\d.]+)\** \| \**([\d.]+) s\** \|",
        register, flags=re.M)
    g17 = re.search(r"^\| G17 \|.*?in DE-LU \(about (\d+)%\)", register, flags=re.M)
    early = re.search(r"^\| FR coverage \| ([\d.]+)% \| ([\d.]+)% \|", register, flags=re.M)
    configs = re.search(r"= (\d+) configurations", register)
    assert static and g17 and early and configs, "a figure quoted on the slides is no longer in the register"
    assert len(size_rows) == 5 and "Change-based beats level-based for all 5 model sizes" in register
    assert "30 days wins for all 5 model sizes" in register
    return {
        "review_items": len(items),
        "review_open": open_ids,
        "static": {"DE-LU": float(static.group(1)), "FR": float(static.group(2))},
        "sizes": [[f"{t} trees", f"{a}%", f"{b}%", c, d, f"{e} s"] for t, a, b, c, d, e in size_rows],
        "quoted": {
            "platform_diff_pct": int(g17.group(1)),
            "fr_first_reported": r(float(early.group(1)), 0),
            "fr_reproduced": r(float(early.group(2)), 0),
            "configurations": int(configs.group(1)),
        },
    }


def breakage_count() -> int:
    return AUDIT.read_text(encoding="utf-8").count("    Mutation(")


def collected_tests() -> int:
    """The number pytest reports: one per test case, so a parametrised test counts once per case."""
    listing = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True, check=True)
    return int(re.search(r"(\d+) tests? collected", listing.stdout).group(1))


def stale_reasons(data: dict[str, Any], tests_collected: int | None = None) -> list[str]:
    """Why `data` (a loaded deck_data.json) no longer matches the project; empty if it is current.

    This is cheap: it hashes the input files and re-reads the register and the audit script. It
    does not recompute the figures, so a change to the package's code with unchanged inputs is
    not seen here; the package's own tests cover that.
    """
    reasons = []
    now, then = source_fingerprint(), data.get("sources", {})
    reasons += [f"{path} has changed" for path in sorted(set(now) | set(then)) if now.get(path) != then.get(path)]
    reg = register_figures()
    checks = [
        ("the number of register items", data["backup"]["review_items"], reg["review_items"]),
        ("the open findings", data["backup"]["review_open"], reg["review_open"]),
        ("the fixed-calibration coverage", data["backup"]["static"], reg["static"]),
        ("the model-size table", data["template"]["sizes"], reg["sizes"]),
        ("the breakage count", data["mlops"]["breakages"], breakage_count()),
    ]
    checks += [(f"the quoted figure {key}", data["facts"][key], value) for key, value in reg["quoted"].items()]
    if tests_collected is not None:
        checks.append(("the test count", data["mlops"]["tests"], tests_collected))
    reasons += [f"{what}: slides have {old!r}, project has {new!r}" for what, old, new in checks if old != new]
    return reasons


def build_deck_data(cfg: CarbonMLOpsConfig | None = None) -> dict[str, Any]:
    """Compute everything the slides show."""
    cfg = cfg or CarbonMLOpsConfig()
    snap = ingest_bronze_snapshot(cfg=cfg)
    feats = build_horizon_features(snap, cfg=cfg, write_output=False)
    splits = chronological_four_way_split(feats, cfg=cfg)
    hist = pd.read_csv(DATA / "forecast_history.csv")
    cal = pd.read_csv(DATA / "locked_test_calibration.csv")
    mon = pd.read_csv(DATA / "gold_monitoring.csv")
    mon["date"] = pd.to_datetime(mon.monitor_date).dt.strftime("%d %b")
    summ = pd.read_csv(DATA / "window_strategy_summary.csv")
    held = summ[(summ.period == "held_out") & (summ.window_hours == 3)].set_index(["strategy", "zone"])
    hyp = pd.read_csv(DATA / "hypothesis_results.csv")
    weekly = pd.read_csv(DATA / "window_value_weekly.csv")
    overall = cal[cal.breakdown == "all"].set_index("zone")

    out: dict = {}

    # slide 1: one real forecast
    day = hist[(hist.zone == "DE-LU") & (hist.forecast_date == "2025-09-16")].sort_values("horizon_h")
    assert len(day) == 24
    out["fan"] = {
        "hours": [f"{h:02d}:00" for h in day.horizon_h],
        **{c: [r(v, 1) for v in day[c]] for c in ("p05", "p50", "p95", "realized")},
    }

    # slide 2: the daily swing
    s = snap.copy()
    s["hour"] = s.timestamp_utc.dt.hour
    s["date"] = s.timestamp_utc.dt.date
    month = s.timestamp_utc.dt.month
    s["season"] = month.map(lambda m: "summer" if m in (6, 7, 8) else "winter" if m in (12, 1, 2) else None)
    daily = s.groupby(["zone", "date"]).carbon_intensity_g_kwh.agg(["min", "max", "mean", "count"])
    daily = daily[daily["count"] == 24]
    out["problem"] = {
        "hours": [f"{h:02d}:00" for h in range(24)],
        "profiles": {
            z: {
                season: [r(v, 1) for v in s[(s.zone == z) & (s.season == season)].groupby("hour").carbon_intensity_g_kwh.mean()]
                for season in ("summer", "winter")
            }
            for z in ZONES
        },
        "ratio": {z: r((daily["max"] / daily["min"]).groupby("zone").median()[z], 2) for z in ZONES},
        "below": {z: r(100 * ((daily["mean"] - daily["min"]) / daily["mean"]).groupby("zone").median()[z], 1) for z in ZONES},
        "mean": {z: r(s.groupby("zone").carbon_intensity_g_kwh.mean()[z], 1) for z in ZONES},
    }

    # slide 3: this project's side of the lineage
    out["lineage"] = {"snapshot_rows": len(snap), "partial_hours": int(snap["is_partial"].sum())}

    # slide 4: method
    names = {"train": "Train", "calib": "Calibration", "gate_val": "Validation", "locked_test": "Held-out test"}
    out["method"] = {
        "splits": [
            {
                "name": names[k],
                "rows": len(v),
                "start": f"{v.origin_timestamp.min():%d %b %Y}",
                "end": f"{v.target_timestamp.max():%d %b %Y}",
            }
            for k, v in splits.items()
        ],
        "feature_rows": len(feats),
        "trees": int(DEFAULT_LGBM_PARAMS["n_estimators"]),
        "window_days": cfg.recalibration_window_days,
        "step_days": cfg.recalibration_step_days,
        "band": [round(100 * b) for b in cfg.coverage_bounds],
    }

    # slide 5: calibration
    lead = cal[cal.breakdown == "horizon_h"].assign(lead=lambda d: d.level.astype(int)).sort_values("lead")
    cov = mon[mon.metric == "coverage_7d"]
    fr_da = cal[(cal.zone == "FR") & (cal.breakdown == "horizon_bucket") & (cal.level == "day_ahead_13_24h")].iloc[0]
    de_q4 = cal[(cal.zone == "DE-LU") & (cal.breakdown == "renewable_quartile") & (cal.level == "Q4")].iloc[0]
    out["trust"] = {
        "coverage": {z: r(100 * overall.loc[z, "coverage"], 1) for z in ZONES},
        "ci": {z: [r(100 * overall.loc[z, "coverage_ci_low"], 1), r(100 * overall.loc[z, "coverage_ci_high"], 1)] for z in ZONES},
        "lead_hours": [int(h) for h in lead[lead.zone == "FR"].lead],
        "lead": {z: [r(v, 4) for v in lead[lead.zone == z].coverage] for z in ZONES},
        "weeks": list(cov[cov.zone == "FR"].sort_values("monitor_date").date),
        "weekly": {z: [r(v, 4) for v in cov[cov.zone == z].sort_values("monitor_date").value] for z in ZONES},
        "weeks_outside": {z: int((cov[cov.zone == z].status != "ok").sum()) for z in ZONES},
        "n_weeks": int(cov[cov.zone == "FR"].shape[0]),
        "fr_day_ahead": r(100 * fr_da.coverage, 1),
        "de_top_renewables": r(100 * de_q4.coverage, 1),
    }


    # slide 6: hypotheses
    def row(h, part, analysis, zone=None, contains=None):
        q = hyp[(hyp.hypothesis == h) & (hyp.analysis == analysis)]
        if part is not None:
            q = q[q.part == part]
        if zone:
            q = q[q.zone == zone]
        if contains:
            q = q[q.quantity.str.contains(contains, regex=False)]
        assert len(q) == 1, (h, part, analysis, zone, contains, len(q))
        return q.iloc[0]


    assert all(v.startswith("falsified") for v in hyp[hyp.analysis == "verdict"].quantity)
    h2 = row("H2", None, "registered")
    h2c = row("H2", None, "corrected")
    de_mid, fr_mid = (float(t.split("=")[1]) for t in h2.note.split("; ")[1].split(", "))
    out["hypotheses"] = {
        "h1_width_bottom": r(row("H1", "context", "context", contains="mean width, bottom quartile").estimate, 0),
        "h1_width_top": r(row("H1", "context", "context", contains="mean width, top quartile").estimate, 0),
        "h1_cleaner_by": r(-row("H1", "a", "corrected").estimate, 0),
        "h1_narrower_by": r(-row("H1", "b", "corrected").estimate, 0),
        "h2_de": r(de_mid, 0),
        "h2_fr": r(fr_mid, 0),
        "h2_days_cleaner": int(h2c.estimate),
        "h2_days": int(h2.n.split(" ")[0]),
        "h2_closest": float(h2c.note.split("= ")[1].split(" ")[0]),
        "h3": {
            z: [r(row("H3", "a", "corrected", zone=z, contains=f"calendar, {b}").estimate, 1) for b in ("short_1_6h", "day_ahead_13_24h")]
            for z in ZONES
        },
    }

    # slide 7: value
    order = ["yesterday", "cautious", "median", "hindsight"]
    out["value"] = {
        "n_days": int(held.n_days.iloc[0]),
        "order": order,
        "tonnes": {z: [r(held.loc[(o, z), "total_avoided_kg_per_daily_mwh"] / 1000, 1) for o in order] for z in ZONES},
        "bad": {z: [r(100 * held.loc[(o, z), "share_of_days_dirtier_than_day_mean"], 0) for o in order[:3]] for z in ZONES},
        "worst": {z: [r(-100 * held.loc[(o, z), "worst_day_saving_vs_day_mean"], 0) for o in order[:3]] for z in ZONES},
        "captured": {z: r(100 * held.loc[("median", z), "share_of_hindsight_value_captured"], 0) for z in ZONES},
    }

    # slide 8: engineering
    out["mlops"] = {
        "tests": collected_tests(),
        "breakages": breakage_count(),
        "champion_version": 7,  # docs/DATABRICKS_VERIFICATION.md, run 466275646757399
        "recalibration_runs": 5,
    }

    # slide 9: monitoring
    skl = mon[mon.metric == "median_skill_vs_persistence"]
    vw = weekly[(weekly.zone == "DE-LU") & (weekly.window_hours == 3)]
    vw = vw.assign(label=pd.to_datetime(vw.week_start).dt.strftime("%d %b"))
    out["limits"] = {
        "weeks": list(skl[skl.zone == "FR"].sort_values("monitor_date").date),
        "skill": {z: [r(v, 4) for v in skl[skl.zone == z].sort_values("monitor_date").value] for z in ZONES},
        "lowest_skill": {z: r(100 * skl[skl.zone == z].value.min(), 0) for z in ZONES},
        "value_weeks": list(vw[vw.strategy == "median"].sort_values("week_start").label),
        "value": {k: [r(v, 1) for v in vw[vw.strategy == k].sort_values("week_start").mean_avoided_kg_per_mwh] for k in ("median", "yesterday")},
        "cautious_bad": r(100 * held.loc[("cautious", "DE-LU"), "share_of_days_dirtier_than_day_mean"], 0),
        "median_bad": r(100 * held.loc[("median", "DE-LU"), "share_of_days_dirtier_than_day_mean"], 0),
    }

    # slide 10: summary
    out["summary"] = {
        "skill": {z: r(100 * (1 - overall.loc[z, "pinball_p50"] / overall.loc[z, "persistence_pinball_p50"]), 0) for z in ZONES}
    }

    # backup slides
    reg = register_figures()
    lead_all = cal[cal.breakdown == "horizon_h"].assign(lead=lambda d: d.level.astype(int)).sort_values("lead")
    quart = cal[cal.breakdown == "renewable_quartile"].sort_values("level")
    months = cal[cal.breakdown == "month"].sort_values("level")
    held_all = summ[summ.period == "held_out"]
    lengths = sorted(int(v) for v in held_all.window_hours.unique())


    def by_window(zone: str, strategy: str, column: str, scale: float = 1.0) -> list[float]:
        part = held_all[(held_all.zone == zone) & (held_all.strategy == strategy)].set_index("window_hours")
        return [r(scale * part.loc[n, column], 1) for n in lengths]


    out["backup"] = {
        "skill_by_lead": {
            z: [r(1 - a / b, 4) for a, b in zip(lead_all[lead_all.zone == z].pinball_p50, lead_all[lead_all.zone == z].persistence_pinball_p50)]
            for z in ZONES
        },
        "quartile": {z: [r(v, 4) for v in quart[quart.zone == z].coverage] for z in ZONES},
        "month_labels": [pd.Timestamp(m + "-01").strftime("%b %Y") for m in months[months.zone == "FR"].level],
        "month": {z: [r(v, 4) for v in months[months.zone == z].coverage] for z in ZONES},
        "static": reg["static"],  # earlier model with static calibration
        "window_lengths": lengths,
        "avoided": {z: {k: by_window(z, k, "mean_avoided_kg_per_mwh") for k in ("yesterday", "median", "hindsight")} for z in ZONES},
        "bad_by_window": {z: {k: by_window(z, k, "share_of_days_dirtier_than_day_mean", 100) for k in ("yesterday", "median")} for z in ZONES},
        "review_items": reg["review_items"],
        "review_open": reg["review_open"],
    }

    # template slides: data introduction, EDA, baseline, model comparison, recommendations
    test_start = splits["locked_test"].origin_timestamp.min()
    eda = snap[snap.timestamp_utc < test_start].copy()  # exploration never used the held-out period
    eda["hour"] = eda.timestamp_utc.dt.hour
    mix_names = ["Wind", "Solar", "Other renewables", "Nuclear", "Fossil"]


    def mix(zone: str) -> list[float]:
        z = eda[eda.zone == zone]
        wind, solar, ren = z.wind_share.mean(), z.solar_share.mean(), z.renewable_share.mean()
        return [r(100 * v, 1) for v in (wind, solar, ren - wind - solar, z.nuclear_share.mean(), z.fossil_share.mean())]


    def acf(zone: str, lags: range) -> list[float]:
        series = eda[eda.zone == zone].set_index("timestamp_utc").carbon_intensity_g_kwh.asfreq("h")
        return [r(series.autocorr(lag), 3) for lag in lags]


    de = eda[eda.zone == "DE-LU"].copy()
    de["decile"] = pd.qcut(de.renewable_share, 10, labels=False)
    summer = eda[eda.timestamp_utc.dt.month.isin([6, 7, 8])]
    buckets = cal[cal.breakdown.isin(["all", "horizon_bucket"])]
    bucket_names = {"all": "All lead times", "short_1_6h": "1\u20136 hours", "mid_7_12h": "7\u201312 hours", "day_ahead_13_24h": "13\u201324 hours"}
    baseline_rows = []
    for z in ZONES:
        for level in bucket_names:
            row_ = buckets[(buckets.zone == z) & (buckets.level == level)].iloc[0]
            baseline_rows.append([z, bucket_names[level], r(row_.persistence_pinball_p50, 2), r(row_.pinball_p50, 2),
                                  r(100 * (1 - row_.pinball_p50 / row_.persistence_pinball_p50), 0)])
    out["template"] = {
        "eda_until": f"{test_start:%d %b %Y}",
        "facts": {
            "rows": len(snap),
            "start": f"{snap.timestamp_utc.min():%d %b %Y}",
            "end": f"{snap.timestamp_utc.max():%d %b %Y}",
            "columns": snap.shape[1],
        },
        "mix_names": mix_names,
        "mix": {z: mix(z) for z in ZONES},
        "acf_lags": list(range(1, 49)),
        "acf": {z: acf(z, range(1, 49)) for z in ZONES},
        "decile_intensity": [r(v, 0) for v in de.groupby("decile").carbon_intensity_g_kwh.mean()],
        "decile_share": [r(100 * v, 0) for v in de.groupby("decile").renewable_share.mean()],
        "summer_day": {z: [r(v, 1) for v in summer[summer.zone == z].assign(hour=lambda d: d.timestamp_utc.dt.hour).groupby("hour").carbon_intensity_g_kwh.mean()] for z in ZONES},
        "baseline": baseline_rows,
        "sizes": reg["sizes"],
        "de_day_ahead": r(100 * buckets[(buckets.zone == "DE-LU") & (buckets.level == "day_ahead_13_24h")].coverage.iloc[0], 1),
    }

    # figures quoted in slide text and speaker notes, so that none is typed into the deck script
    errors = {int(row[0].split()[0]): float(row[3]) for row in reg["sizes"]}
    chosen, largest = int(DEFAULT_LGBM_PARAMS["n_estimators"]), max(errors)
    out["facts"] = {
        **reg["quoted"],
        "largest_multiple": r(largest / chosen, 0),
        "largest_error_cut_pct": r(100 * (1 - errors[largest] / errors[chosen]), 0),
    }

    out["sources"] = source_fingerprint(cfg)
    return out


def main() -> None:
    logging.disable(logging.CRITICAL)  # the pipeline's own progress messages are not wanted here
    data = build_deck_data()
    logging.disable(logging.NOTSET)
    OUT.write_text(json.dumps(data, indent=1) + chr(10), encoding="utf-8", newline=chr(10))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(__name__).info("written %s (%.0f kB)", OUT, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
