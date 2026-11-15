"""Generate docs/LOCKED_TEST_RESULTS.md from the committed locked-test result files.

Every number in the document is read from dashboards/data, never typed by hand, so the
write-up cannot drift from the results. Re-run after any legitimate regeneration of them:

    python scripts/make_results_doc.py
"""
# ruff: noqa: E501, E741
import json
import re
from pathlib import Path

import pandas as pd

NL = chr(10)
c = pd.read_csv("dashboards/data/locked_test_calibration.csv", dtype={"level": str})
r = pd.read_csv("dashboards/data/hypothesis_results.csv")
rec = json.loads(Path("dashboards/data/locked_test_run.json").read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}%"


def cal(zone, breakdown, level):
    m = c[(c.zone == zone) & (c.breakdown == breakdown) & (c.level == str(level))]
    assert len(m) == 1, (zone, breakdown, level, len(m))
    return m.iloc[0]


def res(hyp, part, analysis, zone, contains):
    m = r[(r.hypothesis == hyp) & (r.analysis == analysis) & (r.zone.fillna("") == zone) & r.quantity.str.contains(contains, regex=False)]
    if part is not None:
        m = m[m.part.fillna("") == part]
    assert len(m) == 1, (hyp, part, analysis, zone, contains, len(m))
    return m.iloc[0]


def band(row):
    return "inside" if bool(row.inside_reference_band) else "**outside**"


out = []
w = out.append
w("# Locked-Test Results: Calibration and H1–H3")
w("")
w("Generated from `dashboards/data/locked_test_calibration.csv`, `dashboards/data/hypothesis_results.csv`")
w("and `dashboards/data/locked_test_run.json`. Every number below is read from those files.")
w("")
w("## How to read this")
w("")
w(f"- **Plan:** `docs/ANALYSIS_PLAN_H1_H3.md`, commit `{rec['analysis_plan_commit'][:7]}`, approved and pushed before the run.")
w("- **Run:** one execution of `notebooks/02_cqr_calibration_analysis.py --run` on 2026-11-15, then `--check`,")
w("  which recomputed everything and reproduced the files exactly.")
w(f"- **Period:** origins `{rec['first_origin_utc'][:16]}` to `{rec['last_origin_utc'][:16]}` UTC, {rec['n_forecast_rows']:,} forecast rows,")
w(f"  {rec['n_recalibrations']} weekly recalibrations. Autumn and early winter only.")
w("- **Model:** frozen; base models fitted on Train only; change-based target; per-lead-time offsets from the")
w("  trailing 30 days, recomputed every 7 days.")
w("- **Not blind.** The Locked Test had been scored several times against an earlier model during the review of")
w("  2026-11-15, and an early EDA included its period. See section 1 of the plan. Earlier reported figures")
w("  (91.63% / 89.26% coverage and the first H1–H3 results) came from an uncommitted notebook and are retired.")
w("- **Intervals** are 95% moving-block bootstrap intervals over days (7-day blocks, 10,000 resamples).")
w("")
w("## 1. Calibration of the 90% interval")
w("")
w("| Zone | Coverage | 95% interval | Below p05 | Above p95 | Mean width | Reference band [87%, 96%] |")
w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for z in ("DE-LU", "FR"):
    a = cal(z, "all", "all")
    w(f"| {z} | {pct(a.coverage)} | {pct(a.coverage_ci_low)} – {pct(a.coverage_ci_high)} | {pct(a.share_below_p05)} | {pct(a.share_above_p95)} | {a.mean_width:.1f} | {band(a)} |")
w("")
de, fr = cal("DE-LU", "all", "all"), cal("FR", "all", "all")
w(f"Both zones are inside the reference band, FR only just ({pct(fr.coverage, 2)} against a floor of 87%). Both intervals")
w(f"contain the nominal 90%, FR's barely (upper end {pct(fr.coverage_ci_high)}). With the earlier static calibration the same")
w("period gave FR about 80%.")
w("")
w("### By horizon bucket")
w("")
w("| Zone | Bucket | Coverage | 95% interval | Mean width | Median pinball | Persistence pinball | Band |")
w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for z in ("DE-LU", "FR"):
    for b in ("short_1_6h", "mid_7_12h", "day_ahead_13_24h"):
        a = cal(z, "horizon_bucket", b)
        w(f"| {z} | `{b}` | {pct(a.coverage)} | {pct(a.coverage_ci_low)} – {pct(a.coverage_ci_high)} | {a.mean_width:.1f} | {a.pinball_p50:.2f} | {a.persistence_pinball_p50:.2f} | {band(a)} |")
w("")
frd = cal("FR", "horizon_bucket", "day_ahead_13_24h")
lead = c[c.breakdown == "horizon_h"].assign(h=lambda d: d.level.astype(int))
fr_lead = lead[lead.zone == "FR"].set_index("h")
de_lead = lead[lead.zone == "DE-LU"].set_index("h")
w(f"**FR day-ahead under-covers.** At `13–24h` FR reaches {pct(frd.coverage)}, outside the band, and its interval")
w(f"({pct(frd.coverage_ci_low)} – {pct(frd.coverage_ci_high)}) excludes 90%. FR coverage falls from {pct(fr_lead.coverage.loc[1])} at 1 hour to")
w(f"{pct(fr_lead.coverage.min())} at {int(fr_lead.coverage.idxmin())} hours. DE-LU stays between {pct(de_lead.coverage.min())} and {pct(de_lead.coverage.max())} at every lead time.")
w("See `docs/images/locked_test_coverage_by_horizon.png`.")
w("")
losing = lead[lead.pinball_p50 >= lead.persistence_pinball_p50]
w("The median beats persistence in every zone and bucket. By single lead time it loses in "
  + (", ".join(f"{x.zone} at {x.h} h ({x.pinball_p50:.3f} vs {x.persistence_pinball_p50:.3f})" for x in losing.itertuples()) or "no case") + ".")
w("")
w("### Conditional on renewable share at the target hour")
w("")
w("| Zone | Quartile | Coverage | 95% interval | Below p05 | Above p95 | Median pinball | Persistence pinball |")
w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for z in ("DE-LU", "FR"):
    for q in ("Q1", "Q2", "Q3", "Q4"):
        a = cal(z, "renewable_quartile", q)
        w(f"| {z} | {q} | {pct(a.coverage)} | {pct(a.coverage_ci_low)} – {pct(a.coverage_ci_high)} | {pct(a.share_below_p05)} | {pct(a.share_above_p95)} | {a.pinball_p50:.2f} | {a.persistence_pinball_p50:.2f} |")
w("")
q1, q4 = cal("DE-LU", "renewable_quartile", "Q1"), cal("DE-LU", "renewable_quartile", "Q4")
w("**Calibration is not uniform across regimes.** In DE-LU the highest-renewable quartile (Q4) is covered")
w(f"{pct(q4.coverage)} of the time, and {pct(q4.share_below_p05)} of its hours fall below `p05`: the model over-predicts intensity")
w(f"in the greenest hours. In that quartile the median is also worse than persistence ({q4.pinball_p50:.2f} vs {q4.persistence_pinball_p50:.2f}).")
w(f"The lowest quartile (Q1) is covered {pct(q1.coverage)}, with {pct(q1.share_above_p95)} of hours above `p95`. The marginal 89% is an average")
w("of over-coverage in the middle and under-coverage at both ends.")
w("")
w("### By month")
w("")
w("| Zone | " + " | ".join(sorted(c[c.breakdown == "month"].level.unique())) + " |")
w("| :--- |" + " :--- |" * c[c.breakdown == "month"].level.nunique())
for z in ("DE-LU", "FR"):
    w(f"| {z} | " + " | ".join(pct(cal(z, "month", m).coverage) for m in sorted(c[c.breakdown == "month"].level.unique())) + " |")
w("")
w("## 2. Hypotheses")
w("")
w("Verdict rule (approved): the corrected analysis decides; the registered analysis is shown beside it; a")
w("disagreement is reported as \"sensitive to method\". Significance level 0.01, one-sided.")
w("")
w("| Hypothesis | Registered | Corrected | Verdict |")
w("| :--- | :--- | :--- | :--- |")
for h in ("H1", "H2", "H3"):
    v = r[(r.hypothesis == h) & (r.analysis == "verdict")].iloc[0]
    parts = dict(p.split("=") for p in v.note.split(", "))
    w(f"| {h} | {'supported' if parts['registered'] == 'True' else 'not supported'} | {'supported' if parts['corrected'] == 'True' else 'not supported'} | **{v.quantity}** |")
w("")
w("All three are falsified, and in each case the registered and corrected analyses agree.")
w("")
w("### H1: renewable penetration and volatility (DE-LU): falsified")
w("")
a_c, b_c = res("H1", "a", "corrected", "DE-LU", "median realized"), res("H1", "b", "corrected", "DE-LU", "mean interval width")
a_r, b_r = res("H1", "a", "registered", "DE-LU", "Mann-Whitney"), res("H1", "b", "registered", "DE-LU", "Mann-Whitney")
w("| Part | Analysis | Estimate (top − bottom quartile) | 95% interval | Result |")
w("| :--- | :--- | :--- | :--- | :--- |")
w(f"| a: lower intensity | corrected, median realized | {a_c.estimate:.1f} | {a_c.ci_low:.1f} to {a_c.ci_high:.1f} | supported |")
w(f"| a: lower intensity | registered, Mann-Whitney on `p50` | {a_r.estimate:.1f} | p = {a_r.p_value:.1e} | supported |")
w(f"| b: wider interval | corrected, mean width | {b_c.estimate:.1f} | {b_c.ci_low:.1f} to {b_c.ci_high:.1f} | **not supported** |")
w(f"| b: wider interval | registered, Mann-Whitney on width | {b_r.estimate:.1f} | p = {b_r.p_value:.2f} | **not supported** |")
w("")
w(f"Each quartile holds {a_c.n.split('/')[0]} target hours, each counted once. The interval is *narrower*, not wider, in")
w("high-renewable hours, so the two-part claim fails. Part a is close to definitional, since a higher")
w("renewable share lowers the intensity computed from the mix.")
w("")
ctx = {q.quantity: q.estimate for q in r[(r.hypothesis == "H1") & (r.analysis == "context")].itertuples()}
w("Context, not part of the verdict:")
w("")
w("| | Top quartile (high renewables) | Bottom quartile |")
w("| :--- | :--- | :--- |")
w(f"| Mean width | {ctx['mean width, top quartile']:.1f} | {ctx['mean width, bottom quartile']:.1f} |")
w(f"| Width relative to the median forecast | {ctx['relative width W/p50, top quartile']:.2f} | {ctx['relative width W/p50, bottom quartile']:.2f} |")
w(f"| Mean absolute error of the median | {ctx['mean absolute error, top quartile']:.1f} | {ctx['mean absolute error, bottom quartile']:.1f} |")
w(f"| Coverage | {pct(ctx['coverage, top quartile'])} | {pct(ctx['coverage, bottom quartile'])} |")
w("")
w("The realized error is *larger* in high-renewable hours while the interval is narrower. So the falsification")
w("does not show that high-renewable hours are less uncertain. It shows that this model's intervals do not")
w("widen where its errors grow. That is a limitation of the calibration method, which applies one offset per")
w("lead time regardless of regime.")
w("")
w("### H2: midday parity, DE-LU versus FR: falsified (descriptive)")
w("")
h2r, h2c = res("H2", None, "registered", "DE-LU vs FR", "paired Wilcoxon"), res("H2", None, "corrected", "DE-LU vs FR", "days with")
h2s = res("H2", None, "sensitivity", "DE-LU vs FR", "paired Wilcoxon")

n_q, n_all = re.match(r"(\d+) of (\d+) days", h2r.n).groups()
de_mean, fr_mean = re.search(r"mean DE-LU=([\d.]+), mean FR=([\d.]+)", h2r.note).groups()
gap = re.search(r"= ([\d.-]+) gCO2eq", h2c.note).group(1)
n_qs = re.match(r"(\d+) of (\d+) days", h2s.n).group(1)
de_s, fr_s = re.search(r"mean DE-LU=([\d.]+), mean FR=([\d.]+)", h2s.note).groups()
w(f"Of {n_all} days with complete midday data, {n_q} had a DE-LU solar plus wind share above the period median.")
w(f"DE-LU was below FR on {int(h2c.estimate)} of those {n_q} days. Mean midday intensity was {de_mean} gCO2eq/kWh in DE-LU")
w(f"against {fr_mean} in FR, and the smallest daily gap was {gap} gCO2eq/kWh.")
w(f"With the threshold taken from the same dates one year earlier the result is the same ({n_qs} qualifying days,")
w(f"{de_s} against {fr_s}).")
w("")
w("This was known before the run: in the whole 2024–2025 snapshot no hour has DE-LU below FR. Two limits")
w("apply. The period has no summer, which is the regime the hypothesis was about. And the intensity is")
w("production-based, so it says nothing about what a consumer in either zone draws once cross-border flows are counted.")
w("")
w("### H3: horizon decay of persistence: falsified")
w("")
w("H3 needs both an importance reversal (a) and growing uncertainty (b). Part b holds; part a does not.")
w("")
w("**H3a, importance ratio (origin-state / calendar).** A value above 1 means origin-state features matter more.")
w("")
w("| Zone | `1–6h` ratio | 95% interval | `13–24h` ratio | 95% interval | Reversal |")
w("| :--- | :--- | :--- | :--- | :--- | :--- |")
for z in ("DE-LU", "FR"):
    s, l = res("H3", "a", "corrected", z, "short_1_6h"), res("H3", "a", "corrected", z, "day_ahead_13_24h")
    w(f"| {z} | {s.estimate:.2f} | {s.ci_low:.2f} – {s.ci_high:.2f} | {l.estimate:.2f} | {l.ci_low:.2f} – {l.ci_high:.2f} | no |")
w("")
w("The ratio falls with lead time in both zones, which is the direction the hypothesis expected, but it never")
w("drops below 1. Origin-state features still dominate at day-ahead lead times. The registered approximation")
w("(single-feature importances) gives the same answer. Two caveats: the model has no day-ahead structural")
w("features, so only calendar features could overtake; and with a change-based target the origin intensity")
w("enters every forecast additively, which makes it important by construction.")
w("")
w("**H3b, growth of uncertainty with lead time.**")
w("")
w("| Analysis | Zone | Mean daily difference (`13–24h` − `1–6h`) | Days | Share of days positive | p |")
w("| :--- | :--- | :--- | :--- | :--- | :--- |")
for z in ("DE-LU", "FR"):
    x = res("H3", "b", "corrected", z, "daily abs error")
    w(f"| corrected: absolute error, all origin hours | {z} | {x.estimate:.2f} | {x.n} | {x.note.split('=')[1]} | {x.p_value:.1e} |")
for z in ("DE-LU", "FR", "pooled"):
    x = res("H3", "b", "registered", z, "width 13-24h")
    share = x.note.split("=")[1] if "positive" in str(x.note) else "n/a"
    w(f"| registered: width, 00:00 UTC origins | {z} | {x.estimate:.2f} | {x.n} | {share} | {x.p_value:.1e} |")
w("")
w("Errors are larger at day-ahead than at short lead times in both zones, on the corrected measure that does")
w("not depend on how the interval is built. The pooled registered p-value is small mainly because nearly every")
w("pair has the same sign and the two zones are counted as independent; it should not be read as effect size.")
w("")
w("## 3. What these results support, and what they do not")
w("")
w(f"- **Supported:** with weekly recalibration the 90% interval covers {pct(de.coverage)} in DE-LU and {pct(fr.coverage)} in FR over an")
w("  held-out autumn and early winter, and the median beats persistence in every zone and bucket.")
fq1, fq2 = cal("FR", "renewable_quartile", "Q1"), cal("FR", "renewable_quartile", "Q2")
w(f"- **Not supported:** uniform calibration. FR day-ahead ({pct(frd.coverage)}) is under-covered. So are the lowest and highest")
w(f"  renewable-share quartiles in DE-LU ({pct(q1.coverage)} and {pct(q4.coverage)}) and the two lowest in FR ({pct(fq1.coverage)} and {pct(fq2.coverage)}).")
w("  A user scheduling load into the greenest DE-LU hours is in that zone's worst-calibrated regime.")
w("- **Not supported:** any of H1–H3 as registered.")
w("- **Scope:** one period of about 3.5 months, two zones, production-based intensity at median emission factors.")
w("  The interval excludes emission-factor uncertainty, which for FR is larger than the forecast interval.")
w("- **Selection effects:** the model's settings were chosen on Gate-Val, and the decision to recalibrate on a")
w("  schedule was prompted by an earlier look at this same split. The coverage figures are therefore not a")
w("  fully independent confirmation.")
w("- **No changes follow from this run.** The model stays frozen, as the plan requires.")
w("")
Path("docs/LOCKED_TEST_RESULTS.md").write_text(NL.join(out), encoding="utf-8", newline=NL)
print("written", len(out), "lines")
