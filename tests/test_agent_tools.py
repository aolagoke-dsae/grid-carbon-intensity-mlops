"""Gate G4: every number the QA agent states comes from a deterministic tool."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest

from carbon_mlops.agent import qa_agent
from carbon_mlops.agent.qa_agent import (
    BAD_REQUEST,
    NEEDS_ZONE,
    REFUSAL,
    Route,
    answer,
    check_grounding,
    keyword_route,
    parse_route,
    render,
    validate_route,
)
from carbon_mlops.agent.tools import (
    TOOLS,
    ToolInputError,
    check_model_health,
    compare_zones_24h,
    find_greenest_window,
    load_forecasts,
    load_monitoring,
)
from carbon_mlops.config import CarbonMLOpsConfig

CFG = CarbonMLOpsConfig()
ORIGIN = pd.Timestamp("2025-10-01 00:00", tz="UTC")
QUESTIONS = json.loads(
    (Path(qa_agent.__file__).parent / "eval_questions.json").read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def _no_tracing():
    """Tracing writes to a local MLflow store; the tests only need the tools' return values."""
    mlflow.tracing.disable()
    yield


def _day(
    zone: str, p50: list[float], p95: list[float] | None = None, origin=ORIGIN
) -> pd.DataFrame:
    p50_a = np.asarray(p50, dtype=float)
    p95_a = np.asarray(p95, dtype=float) if p95 is not None else p50_a + 20.0
    return pd.DataFrame(
        {
            "zone": zone,
            "origin_timestamp": origin,
            "horizon_h": range(1, 25),
            "target_timestamp": [origin + pd.Timedelta(hours=h) for h in range(1, 25)],
            "p05": p50_a - 10.0,
            "p50": p50_a,
            "p95": p95_a,
        }
    )


def _dip(at: int, base: float = 100.0, low: float = 40.0) -> list[float]:
    """A flat profile with a 3-hour dip starting at position `at` (0-based)."""
    profile = [base] * 24
    for i in range(at, at + 3):
        profile[i] = low
    return profile


# ----------------------------------------------------------------------------- tools
def test_greenest_window_matches_hand_computed_values() -> None:
    result = find_greenest_window("FR", 3, forecasts=_day("FR", _dip(10)), cfg=CFG)
    facts, times = result["facts"], result["times"]

    # A dip at position 10 starts at lead time 11, i.e. 11:00, and ends three hours later.
    assert facts["start_lead_time_h"] == 11
    assert times["window_start"] == "2025-10-01T11:00:00+00:00"
    assert times["window_end"] == "2025-10-01T14:00:00+00:00"
    assert times["forecast_issued"] == "2025-10-01T00:00:00+00:00"
    # Day mean of a flat 100 with three hours at 40 is (21 * 100 + 3 * 40) / 24 = 92.5.
    assert facts["expected_intensity"] == pytest.approx(40.0)
    assert facts["day_mean_expected_intensity"] == pytest.approx(92.5)
    assert facts["expected_saving_pct"] == pytest.approx(100 * (1 - 40.0 / 92.5))
    assert facts["expected_avoided_kg_per_mwh"] == pytest.approx(52.5)
    assert facts["expected_lower_bound"] == pytest.approx(30.0)
    assert facts["expected_upper_bound"] == pytest.approx(60.0)
    assert facts["forecast_horizon_h"] == 24
    assert facts["interval_level_pct"] == pytest.approx(90.0)
    assert result["args"] == {"zone": "FR", "window_hours": 3, "choose_by": "median"}


def test_cautious_choice_ranks_by_the_upper_bound() -> None:
    # The median dips at 4, the upper bound at 15: the two choices must differ.
    forecasts = _day("FR", _dip(4), p95=_dip(15, base=130.0, low=60.0))
    assert find_greenest_window("FR", forecasts=forecasts)["facts"]["start_lead_time_h"] == 5
    cautious = find_greenest_window("FR", choose_by="cautious", forecasts=forecasts)["facts"]
    assert cautious["start_lead_time_h"] == 16
    # Chosen by the upper bound, but the expected intensity reported is still the median: 100.
    assert cautious["expected_intensity"] == pytest.approx(100.0)
    assert cautious["expected_saving_pct"] < 0


def test_tools_use_only_the_latest_complete_forecast_and_reject_bad_requests() -> None:
    older = _day("FR", _dip(2), origin=ORIGIN - pd.Timedelta(hours=6))
    both = pd.concat([older, _day("FR", _dip(10))], ignore_index=True)
    assert find_greenest_window("FR", forecasts=both)["facts"]["start_lead_time_h"] == 11

    incomplete = pd.concat([older, _day("FR", _dip(10)).iloc[:20]], ignore_index=True)
    with pytest.raises(ToolInputError, match="every lead time"):
        find_greenest_window("FR", forecasts=incomplete)
    with pytest.raises(ToolInputError, match="Unknown zone"):
        find_greenest_window("ES", forecasts=both)
    with pytest.raises(ToolInputError, match="window_hours"):
        find_greenest_window("FR", 25, forecasts=both)
    with pytest.raises(ToolInputError, match="choose_by"):
        find_greenest_window("FR", choose_by="reckless", forecasts=both)


def test_compare_zones_reports_each_grid_and_their_ratio() -> None:
    forecasts = pd.concat([_day("DE-LU", _dip(10, base=300.0, low=120.0)), _day("FR", [30.0] * 24)])
    result = compare_zones_24h(forecasts=forecasts, cfg=CFG)
    facts = result["facts"]

    # DE-LU mean: (21 * 300 + 3 * 120) / 24 = 277.5; FR is flat at 30.
    assert facts["de_lu_mean_expected_intensity"] == pytest.approx(277.5)
    assert facts["fr_mean_expected_intensity"] == pytest.approx(30.0)
    assert facts["ratio_dirtier_to_cleaner"] == pytest.approx(277.5 / 30.0)
    assert facts["de_lu_cleanest_hour_intensity"] == pytest.approx(120.0)
    assert facts["de_lu_dirtiest_hour_intensity"] == pytest.approx(300.0)
    assert facts["de_lu_mean_interval_width"] == pytest.approx(30.0)
    assert result["labels"] == {"cleaner_zone": "FR", "dirtier_zone": "DE-LU"}
    assert result["times"]["de_lu_cleanest_hour"] == "2025-10-01T11:00:00+00:00"


def _monitoring(coverage: float, status: str, skill: float, alerts: int = 2) -> pd.DataFrame:
    rows = []
    for week, (cov, stat) in enumerate([(0.90, "ok"), (0.84, "below_band"), (coverage, status)]):
        date = pd.Timestamp("2025-10-02", tz="UTC") + pd.Timedelta(weeks=week)
        base = {
            "monitor_date": date,
            "zone": "FR",
            "n": 4000,
            "lower_bound": 0.87,
            "upper_bound": 0.96,
        }
        rows.append(
            base
            | {"metric_group": "coverage", "metric": "coverage_7d", "value": cov, "status": stat}
        )
        rows.append(
            base
            | {
                "metric_group": "forecast_quality",
                "metric": "median_skill_vs_persistence",
                "value": skill,
                "status": "info",
            }
        )
        for i in range(5):
            rows.append(
                base
                | {
                    "metric_group": "feature_drift_same_season",
                    "metric": f"psi_same_season_f{i}",
                    "value": 0.3,
                    "status": "alert" if i < alerts else "ok",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("coverage", "status", "skill", "verdict"),
    [
        (0.91, "ok", 0.20, "healthy"),
        (0.85, "below_band", 0.20, "needs attention"),
        (0.97, "above_band", 0.20, "needs attention"),
        (0.91, "ok", -0.05, "needs attention"),
    ],
)
def test_model_health_reads_the_latest_week_and_applies_the_rule(
    coverage: float, status: str, skill: float, verdict: str
) -> None:
    result = check_model_health("FR", monitoring=_monitoring(coverage, status, skill), cfg=CFG)
    facts = result["facts"]

    assert facts["weekly_coverage_pct"] == pytest.approx(100 * coverage)
    assert facts["skill_vs_no_change_pct"] == pytest.approx(100 * skill)
    assert (facts["band_lower_pct"], facts["band_upper_pct"]) == pytest.approx((87.0, 96.0))
    assert (facts["drift_features_on_alert"], facts["drift_features_checked"]) == (2, 5)
    assert facts["weeks_monitored"] == 3
    # One earlier week was below the band; the latest counts too when it is outside.
    assert facts["weeks_outside_band"] == 1 + (status != "ok")
    assert facts["forecasts_in_week"] == 4000
    assert result["times"]["week_ending"] == "2025-10-16T00:00:00+00:00"
    assert result["labels"] == {"coverage_status": status, "verdict": verdict}


def test_tools_on_the_committed_snapshots_match_an_independent_computation() -> None:
    forecasts = load_forecasts(CFG)
    for zone in CFG.zones:
        day = forecasts[forecasts["zone"] == zone].sort_values("horizon_h")
        rolling = day["p50"].rolling(3).mean().dropna()
        facts = find_greenest_window(zone, 3, cfg=CFG)["facts"]
        assert facts["expected_intensity"] == pytest.approx(rolling.min())
        assert facts["start_lead_time_h"] == int(day["horizon_h"].iloc[rolling.argmin()])

    monitoring = load_monitoring(CFG)
    latest = monitoring[monitoring["monitor_date"] == monitoring["monitor_date"].max()]
    for zone in CFG.zones:
        row = latest[(latest["zone"] == zone) & (latest["metric"] == "coverage_7d")].iloc[0]
        facts = check_model_health(zone, cfg=CFG)["facts"]
        assert facts["weekly_coverage_pct"] == pytest.approx(100 * row["value"])


def test_every_tool_is_traced_and_registered() -> None:
    assert set(TOOLS) == {"find_greenest_window", "compare_zones_24h", "check_model_health"}
    for tool in TOOLS.values():
        assert hasattr(tool, "__wrapped__"), f"{tool.__name__} is not decorated with mlflow.trace"


# ----------------------------------------------------------------------------- routing
@pytest.mark.parametrize("case", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_keyword_router_handles_the_evaluation_questions(case: dict) -> None:
    route = keyword_route(case["question"], CFG)
    expect = case["expect"]
    if "tool" in expect:
        assert route == Route(expect["tool"], expect["args"])
    elif expect.get("clarify"):
        assert route == NEEDS_ZONE
    elif expect.get("bad_request"):
        assert route == BAD_REQUEST
    else:
        assert route is None


OFF_TOPIC_QUESTIONS = (
    "When will electricity prices be lowest in France?",
    "What is the cheapest window to run a load in Germany?",
    "When does the sun rise in Germany?",
    "What is the weather forecast for France this week?",
    "How does France differ from Spain?",
    "Germany vs Brazil at football?",
    "When should I start charging my car in Poland?",
    "Can I trust the price forecast for France?",
    "When does the conference start in Germany?",
)


@pytest.mark.parametrize("question", OFF_TOPIC_QUESTIONS)
def test_questions_the_tables_cannot_answer_are_refused_without_figures(question: str) -> None:
    # Each of these holds a zone name or a routing keyword, and must still be refused.
    assert keyword_route(question, CFG) is None
    reply = answer(question, CFG)
    assert reply.text == REFUSAL and reply.result is None
    assert re.search(r"\d", reply.text) is None


def test_keyword_router_reads_vs_and_ignores_the_forecast_length() -> None:
    assert keyword_route("Germany vs the other grid?", CFG) == Route("compare_zones_24h", {})
    assert keyword_route("Do TVs use more power at night?", CFG) is None  # "vs" inside a word
    for question in (
        "What is the cleanest window in the 24h forecast for Germany?",
        "Cleanest window in Germany, 24 hours ahead?",
    ):
        assert keyword_route(question, CFG) == Route(
            "find_greenest_window", {"zone": "DE-LU", "window_hours": 3, "choose_by": "median"}
        ), question
    assert keyword_route("Cleanest 24-hour window in Germany?", CFG) == Route(
        "find_greenest_window", {"zone": "DE-LU", "window_hours": 24, "choose_by": "median"}
    )


def test_route_validation_rejects_anything_outside_the_schema() -> None:
    ok = Route("find_greenest_window", {"zone": "FR", "window_hours": 4, "choose_by": "median"})
    assert validate_route(ok, CFG) == ok
    # Defaults are filled in; unknown arguments are dropped.
    assert validate_route(Route("find_greenest_window", {"zone": "FR", "note": "x"}), CFG) == Route(
        "find_greenest_window", {"zone": "FR", "window_hours": 3, "choose_by": "median"}
    )
    for bad in (
        Route("delete_tables", {}),
        Route("find_greenest_window", {"zone": "ES"}),
        Route("find_greenest_window", {"zone": "FR", "window_hours": 30}),
        Route("find_greenest_window", {"zone": "FR", "window_hours": "4"}),
        Route("find_greenest_window", {"zone": "FR", "window_hours": True}),
        Route("find_greenest_window", {"zone": "FR", "choose_by": "reckless"}),
        Route("check_model_health", {}),
    ):
        assert validate_route(bad, CFG) is None, bad


def test_router_reply_parsing_accepts_one_json_object_only() -> None:
    assert parse_route('{"tool": "compare_zones_24h", "args": {}}') == Route(
        "compare_zones_24h", {}
    )
    fenced = (
        '<think>hmm, 42</think>```json\n{"tool": "check_model_health", "args": {"zone": "FR"}}\n```'
    )
    assert parse_route(fenced) == Route("check_model_health", {"zone": "FR"})
    for bad in ("The answer is 42.", "{not json}", '["compare_zones_24h"]', '{"args": {}}'):
        assert parse_route(bad) is None, bad


# ----------------------------------------------------------------------------- grounding
def test_every_answer_to_the_evaluation_questions_is_grounded() -> None:
    for case in QUESTIONS:
        reply = answer(case["question"], CFG)
        assert reply.grounded and reply.violations == (), case["id"]
        expect = case["expect"]
        if "tool" in expect:
            assert reply.result is not None and reply.result["tool"] == expect["tool"]
            assert check_grounding(reply.text, reply.result) == []
        else:
            # Refusals, clarifications and bad requests must not contain a single figure.
            assert reply.result is None
            assert re.search(r"\d", reply.text) is None, case["id"]
            assert reply.text in (REFUSAL, NEEDS_ZONE, BAD_REQUEST)


def test_grounding_check_catches_numbers_that_no_tool_produced() -> None:
    result = find_greenest_window("FR", 3, forecasts=_day("FR", _dip(10)), cfg=CFG)
    text = render(result)
    assert check_grounding(text, result) == []
    assert "CO2" in text, "the unit must not be mistaken for a number"

    # 92.5 is written as "92": changing it, or adding a figure, must be caught.
    assert "against 92 on average" in text
    assert check_grounding(text.replace("against 92 on", "against 93 on"), result) == ["93"]
    assert check_grounding(text + " Roughly 12,345 tonnes a year.", result) == ["12,345"]
    assert check_grounding(text + " It is 57% lower.", result) == []  # 56.76 at no decimals
    assert check_grounding(text + " It is 56.7% lower.", result) == ["56.7"]  # but not this
    # A timestamp is allowed only if it is one of the tool's own.
    assert check_grounding(text + " Or try 02 Oct 2025 05:00.", result) != []
    # With no tool result, any figure is a violation.
    assert check_grounding("About 5 percent.", None) == ["5"]


def test_a_template_that_drifts_from_its_facts_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_agent, "render", lambda result: "The saving is 12.3 percent.")
    reply = answer("What is the cleanest 4-hour window in France?", CFG)
    assert not reply.grounded and reply.violations == ("12.3",)
    assert reply.text == qa_agent.UNGROUNDED


# ----------------------------------------------------------------------------- model router
def _client(content: str | Exception) -> SimpleNamespace:
    def create(**kwargs: object) -> SimpleNamespace:
        if isinstance(content, Exception):
            raise content
        assert kwargs["temperature"] == 0
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_a_model_can_only_choose_the_tool_never_the_numbers() -> None:
    question = "What is the cleanest 4-hour window in France?"
    by_keyword = answer(question, CFG)

    chatty = (
        "The saving will be 99.9% and intensity 1 g. "
        '{"tool": "find_greenest_window", "args": {"zone": "FR", "window_hours": 4}}'
    )
    by_model = answer(question, CFG, router="llm", client=_client(chatty), model="fake")
    assert by_model.router_used == "llm"
    assert by_model.text == by_keyword.text, "the model's own prose and figures must be discarded"
    assert "99.9" not in by_model.text

    # A model that picks a different valid tool changes the tool, and nothing else.
    other = '{"tool": "check_model_health", "args": {"zone": "FR"}}'
    rerouted = answer(question, CFG, router="llm", client=_client(other), model="fake")
    assert rerouted.result["tool"] == "check_model_health" and rerouted.grounded


@pytest.mark.parametrize(
    "reply",
    [
        "I think the answer is 42.",
        '{"tool": "drop_everything", "args": {}}',
        '{"tool": "find_greenest_window", "args": {"zone": "FR", "window_hours": 99}}',
        RuntimeError("endpoint unavailable"),
    ],
)
def test_an_unusable_model_reply_falls_back_to_the_keyword_router(reply: str | Exception) -> None:
    question = "What is the cleanest 4-hour window in France?"
    result = answer(question, CFG, router="llm", client=_client(reply), model="fake")
    assert result.router_used == "keyword (model reply rejected)"
    assert result.text == answer(question, CFG).text and result.grounded


@pytest.mark.parametrize(
    ("question", "guessed"),
    [
        (
            "When is the cleanest time to run?",
            '{"tool": "find_greenest_window", "args": {"zone": "DE-LU"}}',
        ),
        ("Is the model drifting?", '{"tool": "check_model_health", "args": {"zone": "DE-LU"}}'),
    ],
)
def test_a_model_may_not_guess_a_zone_the_question_does_not_name(
    question: str, guessed: str
) -> None:
    # Seen on four hosted models on 2026-11-18: with no zone in the question, each chose DE-LU.
    reply = answer(question, CFG, router="llm", client=_client(guessed), model="fake")
    assert reply.text == qa_agent.NEEDS_ZONE and reply.result is None
    assert reply.router_used == "keyword (model reply rejected)"


def test_a_model_may_not_pick_a_tool_for_a_subject_the_tables_do_not_hold() -> None:
    question = "When will electricity prices be lowest in France?"
    picked = '{"tool": "find_greenest_window", "args": {"zone": "FR", "window_hours": 3}}'
    reply = answer(question, CFG, router="llm", client=_client(picked), model="fake")
    assert reply.text == REFUSAL and reply.result is None
    assert reply.router_used == "keyword (model reply rejected)"


def test_a_model_may_not_swap_the_zone_the_question_names() -> None:
    question = "What is the cleanest 4-hour window in France?"
    wrong = '{"tool": "find_greenest_window", "args": {"zone": "DE-LU", "window_hours": 4}}'
    reply = answer(question, CFG, router="llm", client=_client(wrong), model="fake")
    assert reply.route is not None and reply.route.args["zone"] == "FR"
    assert reply.text == answer(question, CFG).text


def test_a_model_may_not_change_the_window_length_the_question_states() -> None:
    # Seen on a hosted model on 2026-11-18: asked for 30 hours, it answered for 24.
    question = "Find the best 30 hour window in Germany."
    clamped = '{"tool": "find_greenest_window", "args": {"zone": "DE-LU", "window_hours": 24}}'
    reply = answer(question, CFG, router="llm", client=_client(clamped), model="fake")
    assert reply.text == qa_agent.BAD_REQUEST and reply.result is None

    # A length written in words is the model's to read; digits in the question are not.
    worded = "When should I charge for two hours in France?"
    two = '{"tool": "find_greenest_window", "args": {"zone": "FR", "window_hours": 2}}'
    by_model = answer(worded, CFG, router="llm", client=_client(two), model="fake")
    assert by_model.router_used == "llm" and by_model.route.args["window_hours"] == 2
