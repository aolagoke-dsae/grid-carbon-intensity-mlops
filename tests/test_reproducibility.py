"""The Dockerfile, the CI workflow and the DVC record must agree with each other and the data."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from carbon_mlops.config import CarbonMLOpsConfig

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "data/bronze_snapshot/entsoe_generation_sample.parquet"


def _workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/ci_cd.yml").read_text(encoding="utf-8"))


def test_dvc_lock_records_the_snapshot_that_is_in_the_repository() -> None:
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text(encoding="utf-8"))
    (out,) = lock["stages"]["bronze_snapshot"]["outs"]
    path = ROOT / out["path"]

    assert out["path"] == SNAPSHOT
    assert path == CarbonMLOpsConfig().bronze_snapshot_path.resolve()
    assert out["size"] == path.stat().st_size
    assert out["md5"] == hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def test_dvc_leaves_the_snapshot_in_git_and_never_deletes_it() -> None:
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stage = stages["bronze_snapshot"]
    (out,) = stage["outs"]

    # cache: false keeps the file in Git; persist: true stops `dvc repro` removing it first.
    assert out == {SNAPSHOT: {"cache": False, "persist": True}}
    assert "scripts/export_bronze_snapshot.py" in stage["cmd"]
    assert (ROOT / "scripts/export_bronze_snapshot.py").exists()


def test_dockerfile_uses_the_python_version_and_install_command_of_ci() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    steps = _workflow()["jobs"]["validate-and-test"]["steps"]
    setup = next(s for s in steps if "setup-python" in s.get("uses", ""))
    ci_python = setup["with"]["python-version"]
    ci_install = next(s["run"] for s in steps if s.get("name", "").startswith("Install package"))

    assert re.search(r"^FROM python:(\S+?)-slim$", dockerfile, re.MULTILINE).group(1) == ci_python
    assert f"RUN {ci_install}" in dockerfile
    assert 'CMD ["make", "test"]' in dockerfile
    assert re.search(r"^USER (?!root)\w+$", dockerfile, re.MULTILINE), "must not run as root"


def test_ci_builds_the_image_and_runs_the_tests_inside_it() -> None:
    runs = [s.get("run", "") for s in _workflow()["jobs"]["docker-image"]["steps"]]
    built = [re.search(r"docker build -t (\S+) \.$", r) for r in runs]
    tag = next(m.group(1) for m in built if m)

    # No command after the tag: the container runs the image's own CMD, which is `make test`.
    assert f"docker run --rm {tag}" in runs


def test_the_image_leaves_out_credentials_and_unpublished_folders() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").split())

    assert {".git/", ".env", ".venv/", ".databricks/", "backups/", "dashboards/design/"} <= ignored
    # The tests read the snapshot and the workflow file, so neither may be excluded.
    needed = ("data/bronze_snapshot", ".github", "tests")
    assert not any(line.startswith(needed) for line in ignored)
