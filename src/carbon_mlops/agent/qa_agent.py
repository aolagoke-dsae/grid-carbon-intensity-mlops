"""A question-answering agent whose numbers all come from tested tools (gate G4).

The design keeps a language model away from the numbers:

1. **Route.** A question is mapped to one tool and its arguments. A language model may do this
   (`llm_route`), or a keyword router does (`keyword_route`), which is also the fallback when the
   model is unavailable or returns something invalid. The model's only output is a tool name and
   arguments, validated against a fixed schema; any prose it writes is discarded.
2. **Compute.** The tool in `agent/tools.py` computes the answer from a Gold table.
3. **Render.** A fixed template turns the tool's facts into a sentence. No model is involved.
4. **Check.** `check_grounding` re-reads the finished answer and fails it if it contains any
   number that is not one of the tool's facts.

A question no tool can answer gets a refusal that contains no numbers at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from carbon_mlops.agent.tools import (
    CHOICES,
    TOOLS,
    ToolInputError,
    ToolResult,
    fact_values,
)
from carbon_mlops.config import CarbonMLOpsConfig

logger = logging.getLogger(__name__)

ZONE_NAMES = {"DE-LU": "Germany-Luxembourg (DE-LU)", "FR": "France (FR)"}
ZONE_PATTERNS = {
    "DE-LU": r"\bde[-_ ]?lu\b|german|deutschland|luxemb",
    "FR": r"\bfr\b|france|french",
}
HEALTH_WORDS = r"health|trust|reliab|coverage|calibrat|drift|monitor|skill|still (good|ok|working)"
COMPARE_WORDS = (
    r"compar|versus|vs|both (grids|zones|countries)|two (grids|zones)"
    r"|which (grid|zone|country)|differ"
)
WINDOW_WORDS = r"when|window|greenest|cleanest|best (time|hours?)|schedul|shift|charg|\brun\b|start"
CAUTIOUS_WORDS = r"cautious|upper bound|worst[- ]case|conservativ|\bsafe"
DEFAULT_WINDOW_HOURS = 3
TIME_FORMAT = "%d %b %Y %H:%M"
REFUSAL = (
    "I can only answer from the project's forecast and monitoring tables, and that question is "
    "outside them. I can tell you when the cleanest hours are in a zone's latest forecast, how the "
    "two grids compare over the forecast day, or whether the model is still healthy in a zone."
)
EVAL_QUESTIONS = Path(__file__).with_name("eval_questions.json")
NEEDS_ZONE = "Which zone do you mean: Germany-Luxembourg (DE-LU) or France (FR)?"
BAD_REQUEST = (
    "I cannot answer that as asked: the request is outside what the tools accept, for example an "
    "unknown zone or a window longer than the forecast."
)
UNGROUNDED = (
    "I could not produce an answer in which every number is traceable, so I am not answering."
)
DEFAULT_DATABRICKS_MODEL = "databricks-meta-llama-3-3-70b-instruct"
ROUTER_PROMPT = """You route questions about electricity carbon-intensity forecasts to one tool.
Reply with JSON only, no prose, in exactly this shape: {"tool": "<name>", "args": {...}}

Tools:
- find_greenest_window: when to run a flexible load. args: zone ("DE-LU" or "FR"),
  window_hours (integer 1-24, default 3), choose_by ("median", or "cautious" for the upper bound).
- compare_zones_24h: compare the two grids over the forecast day. args: {}
- check_model_health: is the model still trustworthy: its coverage, its error against the
  baseline, drift in its inputs. args: zone ("DE-LU" or "FR").
- none: the question is about anything else (prices, weather, other countries, opinions).

Germany or Luxembourg means "DE-LU". France means "FR". Never answer the question yourself and
never write numbers other than window_hours."""


@dataclass(frozen=True)
class Route:
    """One tool call: the only thing a router, human-written or learned, is allowed to produce."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    """An answer with everything needed to audit it."""

    text: str
    route: Route | None
    router_used: str
    grounded: bool
    violations: tuple[str, ...] = ()
    result: ToolResult | None = None


# ----------------------------------------------------------------------------- routing
def _zone_in(text: str) -> str | None:
    found = [zone for zone, pattern in ZONE_PATTERNS.items() if re.search(pattern, text)]
    return found[0] if len(found) == 1 else None


def _hours_in(text: str) -> int | None:
    """The window length a question states in digits, such as "4-hour" or "6h"."""
    hours = re.search(r"(\d+)\s*-?\s*(?:hours?|hrs?|h)\b", text)
    return int(hours.group(1)) if hours else None


def validate_route(route: Route, cfg: CarbonMLOpsConfig | None = None) -> Route | None:
    """Return a cleaned route if it names a real tool with acceptable arguments, else None."""
    cfg = cfg or CarbonMLOpsConfig()
    args = dict(route.args) if isinstance(route.args, dict) else {}
    if route.tool == "compare_zones_24h":
        return Route("compare_zones_24h", {})
    if route.tool == "check_model_health":
        return Route(route.tool, {"zone": args["zone"]}) if args.get("zone") in cfg.zones else None
    if route.tool == "find_greenest_window":
        if args.get("zone") not in cfg.zones:
            return None
        hours = args.get("window_hours", DEFAULT_WINDOW_HOURS)
        choose_by = args.get("choose_by", "median")
        if isinstance(hours, bool) or not isinstance(hours, int):
            return None
        if not 1 <= hours <= len(cfg.horizons) or choose_by not in CHOICES:
            return None
        return Route(
            route.tool, {"zone": args["zone"], "window_hours": hours, "choose_by": choose_by}
        )
    return None


def keyword_route(question: str, cfg: CarbonMLOpsConfig | None = None) -> Route | str | None:
    """Route by keywords: a Route, a clarifying question (str), or None when out of scope."""
    text = question.lower()
    zone = _zone_in(text)
    both = len([z for z, pattern in ZONE_PATTERNS.items() if re.search(pattern, text)]) == 2
    if re.search(HEALTH_WORDS, text):
        return validate_route(Route("check_model_health", {"zone": zone}), cfg) or NEEDS_ZONE
    if both or re.search(COMPARE_WORDS, text):
        return Route("compare_zones_24h", {})
    if re.search(WINDOW_WORDS, text):
        if zone is None:
            return NEEDS_ZONE
        hours = _hours_in(text)
        args = {
            "zone": zone,
            "window_hours": hours if hours is not None else DEFAULT_WINDOW_HOURS,
            "choose_by": "cautious" if re.search(CAUTIOUS_WORDS, text) else "median",
        }
        return validate_route(Route("find_greenest_window", args), cfg) or BAD_REQUEST
    return None


def parse_route(reply: str) -> Route | None:
    """Read a router model's reply. Anything but one JSON object naming a tool is rejected."""
    cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("tool"), str):
        return None
    args = payload.get("args", {})
    return Route(payload["tool"], args if isinstance(args, dict) else {})


def llm_route(
    question: str,
    client: Any,
    model: str,
    cfg: CarbonMLOpsConfig | None = None,
) -> Route | None:
    """Ask a chat model for a tool call. Returns None unless the reply is a valid route.

    A route is also rejected if it names a zone the question does not name, or a window length
    other than the one the question states.
    """
    try:
        reply = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        content = reply.choices[0].message.content or ""
    except Exception as exc:  # any backend failure falls back to the keyword router
        logger.warning("Router model failed (%s); using the keyword router", type(exc).__name__)
        return None
    route = parse_route(content)
    route = validate_route(route, cfg) if route is not None else None
    if route is None:
        return None
    # Seen on hosted models: they pick a zone when the question names none, and shorten a
    # window that is too long instead of refusing it. Neither is the model's call, so such a
    # route is rejected and the keyword router asks, or refuses, in fixed words.
    text = question.lower()
    zone = route.args.get("zone")
    if zone is not None and _zone_in(text) != zone:
        logger.warning("Router model chose a zone the question does not name; rejected")
        return None
    stated = _hours_in(text)
    if stated is not None and route.args.get("window_hours", stated) != stated:
        logger.warning("Router model changed the window length the question states; rejected")
        return None
    return route


def make_client(cfg: CarbonMLOpsConfig | None = None) -> tuple[Any, str]:
    """An OpenAI-compatible client and model name for the configured backend.

    No credential is stored. For Databricks model serving they come from `DATABRICKS_HOST` and
    `DATABRICKS_TOKEN` when both are set, and otherwise from the Databricks SDK's own
    authentication: the job's identity inside Databricks, a CLI profile on a laptop. None are
    needed for a local Ollama server.
    """
    from openai import OpenAI  # optional dependency: pip install .[agent]

    cfg = cfg or CarbonMLOpsConfig()
    if cfg.llm_backend == "databricks":
        model = os.getenv("CARBON_LLM_MODEL", DEFAULT_DATABRICKS_MODEL)
        if "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ:
            host = os.environ["DATABRICKS_HOST"].rstrip("/")
            return OpenAI(
                base_url=f"{host}/serving-endpoints", api_key=os.environ["DATABRICKS_TOKEN"]
            ), model
        from databricks.sdk import WorkspaceClient

        # A short-lived token for this process, taken from the SDK's request headers. It is
        # never written anywhere, and it outlasts the minute or two a check takes.
        workspace = WorkspaceClient().config
        token = workspace.authenticate()["Authorization"].removeprefix("Bearer ")
        host = workspace.host.rstrip("/")
        return OpenAI(base_url=f"{host}/serving-endpoints", api_key=token), model
    return OpenAI(base_url=cfg.ollama_base_url, api_key="ollama"), cfg.ollama_model


# ----------------------------------------------------------------------------- rendering
def _time(iso: str) -> str:
    return pd.Timestamp(iso).strftime(TIME_FORMAT)


def _n(value: float, decimals: int = 0) -> str:
    return f"{abs(value):,.{decimals}f}"


def render(result: ToolResult) -> str:
    """Turn a tool result into a sentence. Every number written here is one of `result['facts']`."""
    f, t = result["facts"], result["times"]
    if result["tool"] == "find_greenest_window":
        zone = ZONE_NAMES[result["args"]["zone"]]
        basis = "upper bound" if result["args"]["choose_by"] == "cautious" else "median forecast"
        direction = "lower" if f["expected_saving_pct"] >= 0 else "higher"
        moved = "avoided" if f["expected_avoided_kg_per_mwh"] >= 0 else "added"
        return (
            f"In {zone}, the cleanest {_n(f['window_hours'])}-hour window by the {basis} runs from "
            f"{_time(t['window_start'])} to {_time(t['window_end'])} UTC, in the forecast issued "
            f"{_time(t['forecast_issued'])} UTC. Expected carbon intensity in that window is "
            f"{_n(f['expected_intensity'])} g CO2 per kWh "
            f"({_n(f['interval_level_pct'])}% interval: {_n(f['expected_lower_bound'])} to "
            f"{_n(f['expected_upper_bound'])}), against {_n(f['day_mean_expected_intensity'])} on "
            f"average over the {_n(f['forecast_horizon_h'])} forecast hours. That is "
            f"{_n(f['expected_saving_pct'])}% {direction}, or "
            f"{_n(f['expected_avoided_kg_per_mwh'])} kg CO2 {moved} per MWh moved into the window. "
            "This is what the forecast expects, not a measured saving."
        )
    if result["tool"] == "compare_zones_24h":
        cleaner, dirtier = result["labels"]["cleaner_zone"], result["labels"]["dirtier_zone"]
        ck, dk = (z.lower().replace("-", "_") for z in (cleaner, dirtier))
        parts = [
            f"In the forecast issued {_time(t['forecast_issued'])} UTC, "
            f"{ZONE_NAMES[cleaner]} is the "
            f"cleaner grid: {_n(f[f'{ck}_mean_expected_intensity'])} g CO2 per kWh on average over "
            f"the {_n(f['forecast_horizon_h'])} forecast hours, against "
            f"{_n(f[f'{dk}_mean_expected_intensity'])} in {ZONE_NAMES[dirtier]}, a factor of "
            f"{_n(f['ratio_dirtier_to_cleaner'], 1)}."
        ]
        for zone, key in ((dirtier, dk), (cleaner, ck)):
            parts.append(
                f"{ZONE_NAMES[zone]}: cleanest hour {_time(t[f'{key}_cleanest_hour'])} at "
                f"{_n(f[f'{key}_cleanest_hour_intensity'])}, dirtiest "
                f"{_time(t[f'{key}_dirtiest_hour'])} "
                f"at {_n(f[f'{key}_dirtiest_hour_intensity'])}; the interval is "
                f"{_n(f[f'{key}_mean_interval_width'])} wide on average."
            )
        return " ".join(parts)
    if result["tool"] == "check_model_health":
        zone = ZONE_NAMES[result["args"]["zone"]]
        status = {"ok": "inside", "below_band": "below", "above_band": "above"}[
            result["labels"]["coverage_status"]
        ]
        direction = "lower" if f["skill_vs_no_change_pct"] >= 0 else "higher"
        return (
            f"{zone}, week ending {_time(t['week_ending'])} UTC: the "
            f"{_n(f['target_coverage_pct'])}% interval covered "
            f"{_n(f['weekly_coverage_pct'], 1)}% of "
            f"{_n(f['forecasts_in_week'])} forecasts, {status} the accepted band of "
            f"{_n(f['band_lower_pct'])} to {_n(f['band_upper_pct'])}%. The median forecast's error "
            f"was {_n(f['skill_vs_no_change_pct'])}% {direction} than assuming no change. "
            f"{_n(f['drift_features_on_alert'])} of {_n(f['drift_features_checked'])} input "
            "features "
            f"are on drift alert. Over {_n(f['weeks_monitored'])} weeks monitored, "
            f"{_n(f['weeks_outside_band'])} were outside the band. Verdict by rule: "
            f"{result['labels']['verdict']}."
        )
    raise ValueError(f"No template for tool {result['tool']!r}")


# ----------------------------------------------------------------------------- grounding check
NUMBER = re.compile(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?")


def check_grounding(text: str, result: ToolResult | None) -> list[str]:
    """Every number in `text` that is not one of the tool's facts, at the precision it is written.

    Timestamps are removed first, since they are formatted from `result['times']`. A digit that
    is part of a word, as in "CO2", is not a number. With no tool result, any number is a
    violation: a refusal must not contain figures.
    """
    remaining = text
    allowed: list[float] = []
    if result is not None:
        for iso in result["times"].values():
            remaining = remaining.replace(_time(iso), " ")
        allowed = [abs(v) for v in fact_values(result)]
    violations = []
    for token in NUMBER.findall(remaining):
        plain = token.replace(",", "").rstrip(".")
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        value = float(plain)
        if not any(round(fact, decimals) == value for fact in allowed):
            violations.append(token)
    return violations


# ----------------------------------------------------------------------------- the agent
@mlflow.trace(name="qa_agent.answer", span_type="AGENT")
def answer(
    question: str,
    cfg: CarbonMLOpsConfig | None = None,
    router: str = "keyword",
    client: Any = None,
    model: str | None = None,
    tool_inputs: dict[str, pd.DataFrame] | None = None,
) -> Answer:
    """Answer one question. `router` is "keyword" or "llm"; "llm" falls back to keywords."""
    cfg = cfg or CarbonMLOpsConfig()
    route: Route | str | None = None
    used = "keyword"
    if router == "llm" and client is not None and model:
        route = llm_route(question, client, model, cfg)
        used = "llm" if route is not None else "keyword (model reply rejected)"
    if route is None:
        route = keyword_route(question, cfg)
    if route is None:
        return Answer(REFUSAL, None, used, grounded=True)
    if isinstance(route, str):
        return Answer(route, None, used, grounded=True)

    try:
        result = TOOLS[route.tool](**route.args, cfg=cfg, **(tool_inputs or {}))
    except ToolInputError as exc:
        logger.info("Tool refused the request: %s", exc)
        return Answer(BAD_REQUEST, route, used, grounded=True)
    text = render(result)
    violations = check_grounding(text, result)
    if violations:
        logger.error("Ungrounded numbers in a rendered answer: %s", violations)
        return Answer(UNGROUNDED, route, used, False, tuple(violations), result)
    return Answer(text, route, used, True, (), result)


def evaluate(
    cfg: CarbonMLOpsConfig | None = None,
    router: str = "keyword",
    client: Any = None,
    model: str | None = None,
    pause_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Ask every evaluation question and record what came back.

    Each record says whether the answer is what `eval_questions.json` expects (`ok`), which
    router produced the route (`router_used`), and keeps the tool's facts and source so the
    answer can be checked against the table it was read from. `pause_s` spaces out the calls to
    a hosted model, which is rate-limited.
    """
    cfg = cfg or CarbonMLOpsConfig()
    cases = json.loads(EVAL_QUESTIONS.read_text(encoding="utf-8"))
    fixed = {"clarify": NEEDS_ZONE, "bad_request": BAD_REQUEST, "refuse": REFUSAL}
    records = []
    for case in cases:
        if router == "llm" and pause_s > 0:
            time.sleep(pause_s)
        reply = answer(case["question"], cfg, router=router, client=client, model=model)
        expect = case["expect"]
        if "tool" in expect:
            ok = reply.grounded and reply.route == Route(expect["tool"], expect["args"])
        else:
            ok = reply.result is None and reply.text == fixed[next(iter(expect))]
        result = reply.result or {}
        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "ok": bool(ok),
                "grounded": reply.grounded,
                "router_used": reply.router_used,
                "tool": reply.route.tool if reply.route else None,
                "args": reply.route.args if reply.route else None,
                "text": reply.text,
                "facts": result.get("facts"),
                "times": result.get("times"),
                "source": result.get("source"),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the grounded QA agent one question.")
    parser.add_argument("question")
    parser.add_argument("--router", choices=("keyword", "llm"), default="keyword")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client, model = make_client() if args.router == "llm" else (None, None)
    reply = answer(args.question, router=args.router, client=client, model=model)
    logger.info("%s", reply.text)
    logger.info(
        "[router: %s; tool: %s]", reply.router_used, reply.route.tool if reply.route else "none"
    )


if __name__ == "__main__":
    main()
