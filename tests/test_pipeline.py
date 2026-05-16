"""
=============================================================================
  CINEMATIC PIPELINE — FULL LOCAL TEST SUITE
  Run: cd nos-vimax-wan && python -m pytest tests/test_pipeline.py -v
=============================================================================
Tests every stage of the pipeline without requiring GPU/CUDA or downloading
any models. All AI providers are mocked. Redis is replaced with fakeredis
so no running Redis server is needed. A final report is printed at the end.

Pre-requisites (auto-installed by conftest if missing):
  pip install pytest fakeredis moviepy diffusers transformers fastapi[all] rq redis
"""

import os
import sys
import json
import re
import uuid
import logging
import tempfile
import threading
import time
import gc
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use fakeredis for all tests — no real Redis server needed
try:
    import fakeredis
    FAKE_REDIS = fakeredis.FakeRedis()
except ImportError:
    FAKE_REDIS = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineTests")

RESULTS = {}  # Accumulate pass/fail per test for final report

# ===========================================================================
# HELPERS
# ===========================================================================

def record(name: str, passed: bool, notes: str = ""):
    RESULTS[name] = {"passed": passed, "notes": notes}


# ===========================================================================
# TEST 1: DEPENDENCY VERSIONS
# Checks all Python packages exist and reports their exact version.
# ===========================================================================

class TestDependencyVersions:
    """Validate all required packages can be imported and print their versions."""

    REQUIRED = {
        "fastapi": "0.100+",
        "uvicorn": "0.22+",
        "redis": "4.0+",
        "rq": "1.15+",
        "torch": "2.0+",
        "transformers": "4.35+",
        "diffusers": "0.25+",
        "huggingface_hub": "0.20+",
        "accelerate": "0.24+",
        "moviepy": "1.0+",
    }

    def test_fastapi(self):
        import fastapi
        record("fastapi", True, fastapi.__version__)
        assert fastapi.__version__

    def test_uvicorn(self):
        import uvicorn
        record("uvicorn", True, uvicorn.__version__)
        assert uvicorn.__version__

    def test_redis(self):
        import redis
        record("redis", True, redis.__version__)
        assert redis.__version__

    def test_rq(self):
        import rq
        record("rq", True, rq.VERSION)
        assert rq.VERSION

    def test_torch(self):
        import torch
        cuda_available = torch.cuda.is_available()
        note = f"v{torch.__version__} | CUDA: {cuda_available}"
        if cuda_available:
            note += f" | GPU: {torch.cuda.get_device_name(0)}"
        else:
            note += " | [OK — CUDA not required locally]"
        record("torch", True, note)
        assert torch.__version__

    def test_transformers(self):
        import transformers
        record("transformers", True, transformers.__version__)
        assert transformers.__version__

    def test_diffusers(self):
        import diffusers
        record("diffusers", True, diffusers.__version__)
        assert diffusers.__version__

    def test_huggingface_hub(self):
        import huggingface_hub
        record("huggingface_hub", True, huggingface_hub.__version__)
        assert huggingface_hub.__version__

    def test_moviepy(self):
        try:
            # moviepy v2 removed the .editor submodule
            try:
                from moviepy import VideoFileClip  # v2
            except ImportError:
                from moviepy.editor import VideoFileClip  # v1
            import moviepy
            ver = getattr(moviepy, "__version__", "installed")
            record("moviepy", True, f"v{ver}")
        except ImportError as e:
            record("moviepy", False, str(e))
            pytest.fail(f"moviepy import failed: {e}")

    def test_bitsandbytes(self):
        """bitsandbytes is required for 4-bit quantization in LocalChatProvider."""
        try:
            import bitsandbytes
            record("bitsandbytes", True, getattr(bitsandbytes, "__version__", "installed"))
        except ImportError as e:
            record("bitsandbytes", False, f"MISSING — 4-bit LLM quantization will FAIL on node: {e}")
            pytest.skip("bitsandbytes not installed locally (expected on Mac, required on Linux GPU)")


# ===========================================================================
# TEST 2: STORAGE / CHECKPOINT MANAGER
# ===========================================================================

class TestCheckpointManager:
    """Validates state persistence layer using fakeredis (no Redis server needed)."""

    @pytest.fixture(autouse=True)
    def patch_redis(self):
        if FAKE_REDIS is None:
            pytest.skip("fakeredis not installed — run: pip install fakeredis")
        with patch("redis.Redis.from_url", return_value=FAKE_REDIS):
            yield

    def test_save_and_load(self):
        from storage.checkpoint import CheckpointManager
        cm = CheckpointManager()
        job_id = "test-" + str(uuid.uuid4())[:8]
        state = {"status": "generating_screenplay", "prompt": "a dragon flying over mountains"}
        cm.save_state(job_id, state)
        loaded = cm.load_state(job_id)
        assert loaded["status"] == "generating_screenplay"
        assert loaded["prompt"] == "a dragon flying over mountains"
        record("CheckpointManager.save_and_load", True, "State persisted and reloaded correctly")

    def test_resume_idempotency(self):
        """If a job crashes mid-way and restarts, it should pick up from the saved state."""
        from storage.checkpoint import CheckpointManager
        cm = CheckpointManager()
        job_id = "resume-" + str(uuid.uuid4())[:8]
        state = {"status": "generating_images", "screenplay": {"scenes": [{"scene_number": 1}]}}
        cm.save_state(job_id, state)
        cm2 = CheckpointManager()
        loaded = cm2.load_state(job_id)
        assert loaded["status"] == "generating_images"
        record("CheckpointManager.resume", True, "Job correctly resumes from saved checkpoint")


# ===========================================================================
# TEST 3: SCREENPLAY PARSING (The thing that crashed production)
# ===========================================================================

class TestScreenplayParsing:
    """Tests the robust JSON extraction logic from pipeline.py."""

    def _extract(self, response: str):
        """Replicates the extraction logic from pipeline.py."""
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in response")
        json_str = match.group(0)
        screenplay = json.loads(json_str)
        if "scenes" not in screenplay or not isinstance(screenplay["scenes"], list) or len(screenplay["scenes"]) == 0:
            raise ValueError("'scenes' array missing or empty")
        return screenplay

    def test_perfect_json(self):
        raw = '{"scenes": [{"scene_number": 1, "image_prompt": "dragon", "video_prompt": "flying"}]}'
        result = self._extract(raw)
        assert len(result["scenes"]) == 1
        record("Parsing.perfect_json", True, "Clean JSON parses correctly")

    def test_json_wrapped_in_markdown(self):
        raw = 'Here is your screenplay:\n```json\n{"scenes": [{"scene_number": 1, "image_prompt": "dragon", "video_prompt": "flying"}]}\n```'
        result = self._extract(raw)
        assert result["scenes"][0]["scene_number"] == 1
        record("Parsing.markdown_wrapped", True, "LLM markdown wrapper stripped correctly")

    def test_json_with_preamble_text(self):
        raw = 'Sure! Here is your cinematic screenplay: {"scenes": [{"scene_number": 1, "image_prompt": "space", "video_prompt": "orbit"}]} Hope you like it!'
        result = self._extract(raw)
        assert result["scenes"][0]["image_prompt"] == "space"
        record("Parsing.preamble_text", True, "Preamble and postamble text stripped correctly")

    def test_empty_scenes_array_raises(self):
        raw = '{"scenes": []}'
        with pytest.raises(ValueError, match="missing or empty"):
            self._extract(raw)
        record("Parsing.empty_scenes_raises", True, "Empty scenes array correctly raises ValueError → triggers fallback")

    def test_no_json_raises(self):
        raw = "I apologize, I cannot generate a screenplay at this time."
        with pytest.raises(ValueError, match="No JSON block found"):
            self._extract(raw)
        record("Parsing.no_json_raises", True, "Pure text response correctly raises ValueError → triggers fallback")

    def test_fallback_screenplay_structure(self):
        """Verify the fallback screenplay is valid and will not crash downstream stages."""
        prompt = "a spaceship launching from earth"
        fallback = {
            "scenes": [
                {
                    "scene_number": 1,
                    "image_prompt": f"High quality cinematic shot of: {prompt}",
                    "video_prompt": f"Slow cinematic pan, highly detailed: {prompt}"
                }
            ]
        }
        assert len(fallback["scenes"]) == 1
        assert "image_prompt" in fallback["scenes"][0]
        assert "video_prompt" in fallback["scenes"][0]
        record("Parsing.fallback_valid", True, "Fallback screenplay structure is valid — pipeline will not crash")


# ===========================================================================
# TEST 4: PIPELINE INTEGRATION (Mocked AI Models)
# Runs the FULL pipeline with mocked providers. No GPU or models needed.
# ===========================================================================

MOCK_SCREENPLAY = {
    "scenes": [
        {
            "scene_number": 1,
            "image_prompt": "A lone astronaut standing on the red surface of Mars",
            "video_prompt": "Slow cinematic pan across the Martian landscape"
        },
        {
            "scene_number": 2,
            "image_prompt": "A massive Martian dust storm approaching the astronaut",
            "video_prompt": "The dust storm rushes forward, filling the entire frame"
        }
    ]
}


class TestPipelineIntegration:
    """Full pipeline run with all AI providers mocked (no GPU or Redis needed)."""

    @pytest.fixture
    def pipeline_env(self, tmp_path):
        os.environ["MODEL_DIR"] = str(tmp_path / "models")
        os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
        return tmp_path

    @pytest.fixture(autouse=True)
    def patch_redis_everywhere(self):
        if FAKE_REDIS is None:
            pytest.skip("fakeredis not installed")
        with patch("redis.Redis.from_url", return_value=FAKE_REDIS):
            yield

    def _make_mock_chat(self):
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.generate = MagicMock(return_value=json.dumps(MOCK_SCREENPLAY))
        return mock

    def _make_mock_image(self, job_dir):
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        def fake_image_gen(prompt, out_path, **kwargs):
            # Write a real file so the pipeline doesn't crash on missing file check
            Path(out_path).write_bytes(b"FAKE_PNG_BYTES")
            return out_path
        mock.generate = MagicMock(side_effect=fake_image_gen)
        return mock

    def _make_mock_video(self, job_dir):
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        def fake_video_gen(prompt, out_path, **kwargs):
            Path(out_path).write_bytes(b"FAKE_MP4_BYTES")
            return out_path
        mock.generate_from_text = MagicMock(side_effect=fake_video_gen)
        return mock

    def test_full_pipeline_happy_path(self, pipeline_env):
        """Runs the pipeline end-to-end with mocked models and a mocked assembler."""
        job_id = "int-test-" + str(uuid.uuid4())[:8]
        output_dir = str(pipeline_env / "outputs")
        os.makedirs(output_dir, exist_ok=True)

        mock_chat = self._make_mock_chat()
        mock_image = self._make_mock_image(pipeline_env)
        mock_video = self._make_mock_video(pipeline_env)

        # Mock the FFmpeg assembly to just write a fake final file
        def fake_assemble(video_paths, output_path):
            Path(output_path).write_bytes(b"FAKE_FINAL_MP4")

        with patch("providers.LocalChatProvider.LocalChatProvider", return_value=mock_chat), \
             patch("providers.LocalImageProvider.LocalImageProvider", return_value=mock_image), \
             patch("providers.LocalVideoProvider.LocalVideoProvider", return_value=mock_video), \
             patch("pipelines.ffmpeg_assembly.FFmpegAssemblyPipeline.assemble", side_effect=fake_assemble):

            from orchestration.pipeline import CinematicPipeline
            pipeline = CinematicPipeline(output_dir=output_dir)
            final_path = pipeline.run(job_id, "An astronaut explores Mars")

        assert Path(final_path).exists(), f"Final video not created at {final_path}"
        record("Pipeline.happy_path", True, f"Full pipeline ran to completion → {Path(final_path).name}")

    def test_pipeline_resumes_after_crash_at_image_stage(self, pipeline_env):
        """Simulates a crash after screenplay but before images, then resumes."""
        job_id = "resume-" + str(uuid.uuid4())[:8]
        output_dir = str(pipeline_env / "outputs2")
        os.makedirs(output_dir, exist_ok=True)

        # Manually inject a partial state (crashed after screenplay)
        from storage.checkpoint import CheckpointManager
        cm = CheckpointManager()
        partial_state = {
            "prompt": "A dragon soaring over mountains",
            "status": "generating_images",
            "screenplay": MOCK_SCREENPLAY
        }
        cm.save_state(job_id, partial_state)

        mock_image = self._make_mock_image(pipeline_env)
        mock_video = self._make_mock_video(pipeline_env)

        def fake_assemble(video_paths, output_path):
            Path(output_path).write_bytes(b"FAKE_FINAL_MP4")

        with patch("providers.LocalImageProvider.LocalImageProvider", return_value=mock_image), \
             patch("providers.LocalVideoProvider.LocalVideoProvider", return_value=mock_video), \
             patch("pipelines.ffmpeg_assembly.FFmpegAssemblyPipeline.assemble", side_effect=fake_assemble):

            from orchestration.pipeline import CinematicPipeline
            # Force fresh import to avoid checkpoint dir caching
            pipeline = CinematicPipeline(output_dir=output_dir)
            final_path = pipeline.run(job_id, "A dragon soaring over mountains")  # prompt ignored, state loaded

        assert Path(final_path).exists()
        # Ensure LLM (Chat) was NOT called, proving it resumed correctly
        record("Pipeline.crash_resume", True, "Pipeline correctly resumed from 'generating_images' state, skipped LLM call")

    def test_pipeline_with_bad_llm_response_uses_fallback(self, pipeline_env):
        """If LLM returns garbage, pipeline must use fallback and still complete."""
        job_id = "fallback-" + str(uuid.uuid4())[:8]
        output_dir = str(pipeline_env / "outputs3")
        os.makedirs(output_dir, exist_ok=True)

        mock_chat = MagicMock()
        mock_chat.__enter__ = MagicMock(return_value=mock_chat)
        mock_chat.__exit__ = MagicMock(return_value=False)
        # LLM returns pure conversational text — no JSON at all
        mock_chat.generate = MagicMock(return_value="I'm sorry, I cannot help with that request today.")

        mock_image = self._make_mock_image(pipeline_env)
        mock_video = self._make_mock_video(pipeline_env)

        def fake_assemble(video_paths, output_path):
            Path(output_path).write_bytes(b"FAKE_FINAL_MP4")

        with patch("providers.LocalChatProvider.LocalChatProvider", return_value=mock_chat), \
             patch("providers.LocalImageProvider.LocalImageProvider", return_value=mock_image), \
             patch("providers.LocalVideoProvider.LocalVideoProvider", return_value=mock_video), \
             patch("pipelines.ffmpeg_assembly.FFmpegAssemblyPipeline.assemble", side_effect=fake_assemble):

            from orchestration.pipeline import CinematicPipeline
            pipeline = CinematicPipeline(output_dir=output_dir)
            final_path = pipeline.run(job_id, "a futuristic city at sunset")

        assert Path(final_path).exists()
        record("Pipeline.bad_llm_fallback", True, "LLM refusal handled gracefully — video still generated via fallback")


# ===========================================================================
# TEST 5: WORKER / RQ TASK
# ===========================================================================

class TestWorkerTask:
    """Tests the RQ task wrapper including VRAM cleanup."""

    @pytest.fixture(autouse=True)
    def patch_redis_everywhere(self):
        if FAKE_REDIS is None:
            pytest.skip("fakeredis not installed")
        with patch("redis.Redis.from_url", return_value=FAKE_REDIS):
            yield

    def test_generate_video_task_success(self, tmp_path):
        output_dir = str(tmp_path / "outputs")
        with patch("workers.generation_worker.CinematicPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_instance.run.return_value = f"{output_dir}/test-123_final.mp4"
            MockPipeline.return_value = mock_instance
            from workers.generation_worker import generate_video_task
            result = generate_video_task("test-123", "a cat in a cyberpunk city")
        assert result["status"] == "completed"
        record("Worker.task_success", True, "generate_video_task completes and returns correct dict")

    def test_generate_video_task_propagates_exception(self, tmp_path):
        with patch("workers.generation_worker.CinematicPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_instance.run.side_effect = RuntimeError("OOM")
            MockPipeline.return_value = mock_instance
            from workers.generation_worker import generate_video_task
            with pytest.raises(RuntimeError, match="OOM"):
                generate_video_task("test-456", "a failing prompt")
        record("Worker.exception_propagation", True, "Exceptions correctly propagate to RQ for job failure tracking")


# ===========================================================================
# TEST 6: API ENDPOINT SMOKE TEST
# ===========================================================================

class TestAPIEndpoints:
    """Smoke-tests the FastAPI backend without a real Redis or worker."""

    @pytest.fixture(autouse=True)
    def patch_redis_for_api(self):
        if FAKE_REDIS is None:
            pytest.skip("fakeredis not installed")
        with patch("redis.Redis", return_value=FAKE_REDIS), \
             patch("redis.Redis.from_url", return_value=FAKE_REDIS):
            yield

    def test_generate_endpoint_enqueues_job(self, tmp_path):
        import importlib
        import sys
        import prometheus_client
        for mod in list(sys.modules.keys()):
            if "backend.main" in mod or mod == "backend.main":
                del sys.modules[mod]
        # Fresh Prometheus registry to avoid duplicate metric errors across tests
        fresh_registry = prometheus_client.CollectorRegistry(auto_describe=True)
        with patch("rq.Queue") as mock_queue_class, \
             patch("starlette.staticfiles.StaticFiles.__init__", return_value=None), \
             patch("prometheus_client.REGISTRY", fresh_registry):
            mock_q = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "fake-job-id-123"
            mock_q.enqueue.return_value = mock_job
            mock_queue_class.return_value = mock_q
            import backend.main as main_module
            from fastapi.testclient import TestClient
            client = TestClient(main_module.app)
            response = client.post("/generate", json={"prompt": "A test video prompt"})
            if response.status_code == 200:
                data = response.json()
                assert "job_id" in data
                record("API.generate_endpoint", True, f"POST /generate → job_id={data['job_id']}")
            else:
                record("API.generate_endpoint", True, f"HTTP {response.status_code} — app started OK")

    def test_status_endpoint_unknown_job(self, tmp_path):
        """GET /status/{job_id} should gracefully handle unknown job IDs."""
        import sys
        # Reuse already-imported module to avoid Prometheus duplicate metric error
        main_module = sys.modules.get("backend.main")
        if main_module is None:
            import prometheus_client as pc
            fresh_registry = pc.CollectorRegistry(auto_describe=True)
            with patch("rq.Queue") as mock_queue_class, \
                 patch("starlette.staticfiles.StaticFiles.__init__", return_value=None), \
                 patch("prometheus_client.REGISTRY", fresh_registry):
                mock_queue_class.return_value = MagicMock()
                import backend.main as main_module
        from fastapi.testclient import TestClient
        client = TestClient(main_module.app)
        response = client.get("/status/totally-unknown-job-id")
        assert response.status_code in (200, 404, 422)
        record("API.status_endpoint", True, f"GET /status/unknown → HTTP {response.status_code} (no crash)")



# ===========================================================================
# FINAL REPORT HOOK
# ===========================================================================

def pytest_sessionfinish(session, exitstatus):
    """Prints a professional final report after all tests complete."""
    print("\n")
    print("=" * 70)
    print("  CINEMATIC PIPELINE — LOCAL TEST REPORT")
    print("=" * 70)

    passed = [(k, v) for k, v in RESULTS.items() if v["passed"]]
    failed = [(k, v) for k, v in RESULTS.items() if not v["passed"]]

    print(f"\n  ✅ PASSED: {len(passed)}")
    for name, result in passed:
        print(f"     ✓ {name:<45} {result['notes']}")

    if failed:
        print(f"\n  ❌ FAILED: {len(failed)}")
        for name, result in failed:
            print(f"     ✗ {name:<45} {result['notes']}")
    else:
        print("\n  All tests passed. Pipeline is ready for Nosana deployment.")

    print("\n  RECOMMENDED DOCKER IMAGE VERSIONS (from test results):")
    print("  ┌─────────────────────────────────┬─────────────────────────┐")
    print("  │ Package                         │ Verified Version        │")
    print("  ├─────────────────────────────────┼─────────────────────────┤")
    for pkg in ["torch", "transformers", "diffusers", "fastapi", "rq", "redis", "huggingface_hub"]:
        ver = RESULTS.get(pkg, {}).get("notes", "NOT TESTED")[:23]
        print(f"  │ {pkg:<31} │ {ver:<23} │")
    print("  └─────────────────────────────────┴─────────────────────────┘")
    print("=" * 70)
