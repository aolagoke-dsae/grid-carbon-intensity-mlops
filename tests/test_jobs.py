"""Checks for the Databricks deployment that can be made without a workspace (Phase G).

These tests cannot prove the jobs run on Databricks. They prove that the bundle, the wheel
metadata and the entry points agree with each other, and that the job logic works against a
local registry.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.inference.batch_score import select_latest_complete_origins
from carbon_mlops.ingestion import ingest_entsoe
from carbon_mlops.ingestion.ingest_entsoe import ingest_bronze_snapshot, packaged_snapshot_path
from carbon_mlops.io.store import qualified_table_name

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TASK_KEYS = {"task_key", "depends_on", "environment_key", "python_wheel_task"}
ALLOWED_WHEEL_TASK_KEYS = {"package_name", "entry_point", "parameters", "named_parameters"}


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _bundle() -> dict:
    return yaml.safe_load((REPO_ROOT / "databricks.yml").read_text(encoding="utf-8"))


def test_qualified_table_name_adds_the_catalog_once() -> None:
    assert qualified_table_name("carbon_mlops.gold", "workspace") == "workspace.carbon_mlops.gold"
    assert qualified_table_name("workspace.carbon_mlops.gold", "workspace") == (
        "workspace.carbon_mlops.gold"
    )
    assert qualified_table_name("carbon_mlops.gold", None) == "carbon_mlops.gold"


def test_every_wheel_entry_point_resolves_to_a_no_argument_function() -> None:
    scripts = _pyproject()["project"]["scripts"]
    assert set(scripts) == {
        "carbon-ingest",
        "carbon-features",
        "carbon-train",
        "carbon-gate",
        "carbon-recalibrate",
        "carbon-score",
        "carbon-monitor",
        "carbon-extracts",
        "carbon-agent-check",
        "carbon-agent-check-llm",
    }
    for name, target in scripts.items():
        module_name, function_name = target.split(":")
        function = getattr(importlib.import_module(module_name), function_name)
        assert callable(function), name
        # A console entry point's return value becomes the exit status, so it must be None.
        assert function.__annotations__.get("return") in (None, "None"), name
        assert function.__code__.co_argcount == 0, name


def test_bundle_tasks_match_the_wheel_and_are_valid_for_serverless() -> None:
    bundle = _bundle()
    project = _pyproject()["project"]
    scripts = project["scripts"]
    wheel_package = project["name"].replace("-", "_")

    assert bundle["artifacts"]["default"]["type"] == "whl"
    assert "wheel" in bundle["artifacts"]["default"]["build"]

    jobs = bundle["resources"]["jobs"]
    assert set(jobs) == {
        "carbon_training_pipeline",
        "carbon_batch_inference",
        "carbon_qa_agent_check",
    }
    seen_entry_points: list[str] = []
    for job_name, job in jobs.items():
        environments = {env["environment_key"]: env["spec"] for env in job["environments"]}
        for spec in environments.values():
            assert "environment_version" in spec, job_name
            assert any(dep.endswith(".whl") for dep in spec["dependencies"]), job_name
            assert "databricks-feature-engineering" in spec["dependencies"], job_name

        task_keys = [task["task_key"] for task in job["tasks"]]
        assert len(task_keys) == len(set(task_keys)), job_name
        for task in job["tasks"]:
            where = f"{job_name}.{task['task_key']}"
            # A key at the wrong indentation level (e.g. `parameters` beside
            # `python_wheel_task`) is silently dropped by the bundle.
            assert set(task) <= ALLOWED_TASK_KEYS, where
            assert task["environment_key"] in environments, where
            wheel_task = task["python_wheel_task"]
            assert set(wheel_task) <= ALLOWED_WHEEL_TASK_KEYS, where
            assert wheel_task["package_name"] == wheel_package, where
            assert wheel_task["entry_point"] in scripts, where
            seen_entry_points.append(wheel_task["entry_point"])
            for dependency in task.get("depends_on", []):
                assert dependency["task_key"] in task_keys, where

    assert sorted(seen_entry_points) == sorted(scripts), (
        "every entry point runs in exactly one task"
    )

    training = [t["task_key"] for t in jobs["carbon_training_pipeline"]["tasks"]]
    assert training == [
        "ingest_bronze",
        "materialize_uc_features",
        "train_challenger",
        "promotion_gate",
    ]
    scoring = {t["task_key"]: t for t in jobs["carbon_batch_inference"]["tasks"]}
    assert scoring["spark_udf_batch_score"]["python_wheel_task"]["parameters"] == [
        "--mode",
        "databricks",
    ]
    assert scoring["spark_udf_batch_score"]["depends_on"] == [{"task_key": "recalibrate_champion"}]
    assert jobs["carbon_batch_inference"]["schedule"]["pause_status"] == "PAUSED"
    agent = {t["task_key"]: t for t in jobs["carbon_qa_agent_check"]["tasks"]}
    assert agent["qa_agent_check_llm"]["depends_on"] == [{"task_key": "qa_agent_check"}]
    # The model-router task imports the OpenAI client, which serverless does not guarantee.
    agent_env = jobs["carbon_qa_agent_check"]["environments"][0]["spec"]["dependencies"]
    assert any(dep.startswith("openai") for dep in agent_env)


def test_wheel_ships_the_pinned_snapshot_beside_the_package() -> None:
    cfg = CarbonMLOpsConfig()
    force_include = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    source, destination = next(iter(force_include.items()))
    assert Path(source) == cfg.bronze_snapshot_path
    assert (REPO_ROOT / source).exists()
    # The destination must be where packaged_snapshot_path() looks after installation.
    packaged = packaged_snapshot_path(cfg)
    assert Path(destination) == Path(packaged.parent.name) / packaged.name
    assert packaged.parent.parent == Path(ingest_entsoe.__file__).resolve().parents[2]


def test_ingestion_falls_back_to_the_packaged_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = CarbonMLOpsConfig()
    # Simulate a Databricks job: the repository-relative default path does not exist.
    cfg = CarbonMLOpsConfig(
        bronze_snapshot_path=tmp_path / "absent" / real.bronze_snapshot_path.name
    )
    monkeypatch.setattr(
        ingest_entsoe, "packaged_snapshot_path", lambda _cfg: real.bronze_snapshot_path.resolve()
    )
    df = ingest_bronze_snapshot(cfg=cfg, base_dir=tmp_path)
    assert len(df) == 35082
    assert (df["source_file_sha256"] == cfg.bronze_snapshot_sha256).all()

    # The packaged copy is verified against the pin as well.
    tampered = tmp_path / "tampered.parquet"
    pd.read_parquet(real.bronze_snapshot_path).head(50).to_parquet(tampered, index=False)
    monkeypatch.setattr(ingest_entsoe, "packaged_snapshot_path", lambda _cfg: tampered)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ingest_bronze_snapshot(cfg=cfg, base_dir=tmp_path)


def test_latest_complete_origin_is_chosen_per_zone() -> None:
    def rows(zone: str, origin: str, horizons: range) -> list[dict]:
        return [
            {"zone": zone, "origin_timestamp": pd.Timestamp(origin, tz="UTC"), "horizon_h": h}
            for h in horizons
        ]

    frame = pd.DataFrame(
        rows("DE-LU", "2025-12-31 20:00", range(1, 25))
        + rows("DE-LU", "2025-12-31 21:00", range(1, 25))
        + rows("FR", "2025-12-31 19:00", range(1, 25))
        + rows("FR", "2025-12-31 20:00", range(1, 20))  # incomplete: must be skipped
    )
    latest = select_latest_complete_origins(frame, n_horizons=24)

    assert len(latest) == 48
    chosen = latest.groupby("zone")["origin_timestamp"].first()
    assert chosen["DE-LU"] == pd.Timestamp("2025-12-31 21:00", tz="UTC")
    assert chosen["FR"] == pd.Timestamp("2025-12-31 19:00", tz="UTC")
    assert (
        latest.groupby("zone")["horizon_h"].apply(list).map(lambda h: h == list(range(1, 25))).all()
    )


def test_databricks_jobs_select_an_mlflow_experiment_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from carbon_mlops.models import train

    chosen: list[str] = []
    monkeypatch.setattr(train.mlflow, "set_experiment", chosen.append)

    # Locally MLflow's default experiment is left alone.
    monkeypatch.setattr(train, "is_databricks_runtime", lambda: False)
    assert train.ensure_mlflow_experiment() is None
    assert chosen == []

    # As a Databricks wheel task there is no default experiment, so one must be set.
    monkeypatch.setattr(train, "is_databricks_runtime", lambda: True)
    monkeypatch.setenv(train.EXPERIMENT_ENV_VAR, "/Users/someone/custom-experiment")
    assert train.ensure_mlflow_experiment() == "/Users/someone/custom-experiment"
    assert chosen == ["/Users/someone/custom-experiment"]


def test_agent_check_job_saves_every_answer_and_fails_on_a_wrong_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import mlflow

    from carbon_mlops import jobs
    from carbon_mlops.agent import qa_agent

    mlflow.tracing.disable()
    flushed: list[bool] = []
    monkeypatch.setattr(jobs.mlflow, "flush_trace_async_logging", lambda: flushed.append(True))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    path = Path(jobs.run_agent_check_job(out_dir=tmp_path))
    assert flushed == [True], "traces must be sent before the task ends"
    records = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == jobs.AGENT_ANSWERS_FILE and len(records) == 18
    assert all(record["ok"] for record in records)
    answered = [record for record in records if record["tool"]]
    assert len(answered) == 11 and all(r["facts"] and r["source"] for r in answered)
    # A refusal or a clarifying question carries no tool result and no figures.
    assert all(r["facts"] is None for r in records if not r["tool"])

    # An answer that is not the expected one stops the task, after the answers are saved.
    monkeypatch.setattr(qa_agent, "render", lambda result: "Probably about 42.")
    with pytest.raises(RuntimeError, match="window-default"):
        jobs.run_agent_check_job(out_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [r["grounded"] for r in saved if r["tool"]] == [False] * 11
    assert all(r["ok"] for r in saved if not r["tool"])


def test_agent_check_with_a_model_router_fails_when_the_model_routes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from types import SimpleNamespace

    import mlflow

    from carbon_mlops import jobs
    from carbon_mlops.agent import qa_agent

    def client(fail: bool) -> SimpleNamespace:
        def create(**kwargs: object) -> SimpleNamespace:
            if fail:
                raise ConnectionError("no model")
            route = qa_agent.keyword_route(kwargs["messages"][-1]["content"])
            content = (
                json.dumps({"tool": route.tool, "args": route.args})
                if isinstance(route, qa_agent.Route)
                else '{"tool": "none"}'
            )
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    mlflow.tracing.disable()
    monkeypatch.setattr(jobs.mlflow, "flush_trace_async_logging", lambda: None)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    default_model = qa_agent.DEFAULT_DATABRICKS_MODEL
    run = {"out_dir": tmp_path, "router": "llm", "model": default_model, "pause_s": 0}
    # Each model other than the default keeps its own answers file.
    assert jobs.agent_answers_file("llm", "other/model") == "qa_agent_answers_llm_other_model.json"
    assert jobs.agent_answers_file("keyword", "other/model") == jobs.AGENT_ANSWERS_FILE

    path = Path(jobs.run_agent_check_job(client=client(fail=False), **run))
    records = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == jobs.AGENT_LLM_ANSWERS_FILE
    assert all(record["ok"] for record in records)
    assert [r["router_used"] for r in records].count("llm") == 11

    # A model that never answers leaves every route to the fallback. The answers are still
    # right, so only this check stops the run from passing without having tested the model.
    with pytest.raises(RuntimeError, match="routed no question"):
        jobs.run_agent_check_job(client=client(fail=True), **run)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert all(r["ok"] and r["router_used"] != "llm" for r in saved)


def test_the_model_task_tries_every_model_and_fails_if_one_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from carbon_mlops import jobs

    tried: list[str | None] = []

    def fake_run(router: str, model: str | None) -> str:
        assert router == "llm"
        tried.append(model)
        if model == "b":
            raise RuntimeError("answers not as expected")
        return "saved"

    monkeypatch.setattr(jobs, "run_agent_check_job", fake_run)
    monkeypatch.setattr(sys, "argv", ["carbon-agent-check-llm", "--models", "a", "b", "c"])
    with pytest.raises(RuntimeError, match="^b: answers not as expected$"):
        jobs.agent_check_llm_main()
    assert tried == ["a", "b", "c"], "a failing model must not stop the ones after it"

    # With no list, the configured model is used.
    tried.clear()
    monkeypatch.setattr(sys, "argv", ["carbon-agent-check-llm"])
    jobs.agent_check_llm_main()
    assert tried == [None]
