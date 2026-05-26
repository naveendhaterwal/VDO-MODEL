import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import preload_models


def test_dockerfile_is_pinned_and_has_healthcheck():
    dockerfile = (ROOT / "docker" / "Dockerfile.worker").read_text()
    assert "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04" in dockerfile
    assert "torch==2.6.0" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert ":latest" not in dockerfile


def test_nosana_job_uses_fixed_image_tag_and_health_check():
    config = json.loads((ROOT / "nosana-job.json").read_text())
    args = config["ops"][0]["args"]
    assert args["image"].endswith(":v1.0.0")
    expose = args["expose"][0]
    assert expose["port"] == 8000
    assert expose["health_checks"][0]["path"] == "/health"


def test_start_worker_preserves_child_exit_codes():
    script = (ROOT / "scripts" / "start_worker.sh").read_text()
    assert "wait -n -p EXITED_PID" in script
    assert 'cleanup "${CHILD_EXIT_CODE}"' in script


def test_requirements_are_exact_pins():
    requirements = (ROOT / "requirements.txt").read_text().strip().splitlines()
    pinned = [line for line in requirements if line and not line.startswith("#")]
    assert pinned
    assert all("==" in line for line in pinned)


def test_transformers_validator_rejects_incomplete_model(tmp_path):
    model_dir = tmp_path / "Qwen2.5-7B-Instruct"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    with pytest.raises(Exception):
        preload_models.validate_transformers_model(model_dir)
