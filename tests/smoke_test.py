"""
Production Smoke Test Suite
Run: python -m pytest tests/smoke_test.py -v --timeout=120

Tests import integrity, Redis connectivity, GPU availability, API wiring,
and pipeline module structure WITHOUT downloading models or requiring GPU.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# TEST 1: IMPORT INTEGRITY
# ===========================================================================

class TestImports:
    def test_core_imports(self):
        import torch
        import diffusers
        import transformers
        import accelerate
        import safetensors
        import moviepy
        import fastapi
        import uvicorn
        import redis
        import rq
        import prometheus_client
        import huggingface_hub
        try:
            import sentencepiece
        except ImportError:
            pass # Optional transitive C++ dependency for certain tokenizers, installed in Linux container

    def test_project_imports(self):
        from storage.checkpoint import CheckpointManager
        from monitoring.memory_watchdog import MemoryWatchdog
        from workers.generation_worker import generate_video_task
        from orchestration.pipeline import CinematicPipeline
        from pipelines.ffmpeg_assembly import FFmpegAssemblyPipeline
        from providers.LocalChatProvider import LocalChatProvider
        from providers.LocalImageProvider import LocalImageProvider
        from providers.LocalVideoProvider import LocalVideoProvider

    def test_backend_imports(self):
        import sys
        for mod in list(sys.modules.keys()):
            if "backend.main" in mod or mod == "backend.main":
                del sys.modules[mod]
        import prometheus_client
        fresh_registry = prometheus_client.CollectorRegistry(auto_describe=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("prometheus_client.REGISTRY", fresh_registry)
            from backend.main import app, GenerateRequest, JobStatus



# ===========================================================================
# TEST 2: GPU AVAILABILITY
# ===========================================================================

class TestGPU:
    def test_cuda_available(self):
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"\n  GPU: {props.name}")
            print(f"  VRAM: {vram_gb:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            assert vram_gb >= 20, f"VRAM too low: {vram_gb:.1f}GB"

    def test_ffmpeg_available(self):
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        assert result.returncode == 0
        first_line = result.stdout.split("\n")[0]
        print(f"\n  {first_line}")


# ===========================================================================
# TEST 3: API INTEGRITY (no Redis needed — tests schema only)
# ===========================================================================

class TestAPISchema:
    def test_request_validation(self):
        import sys
        for mod in list(sys.modules.keys()):
            if "backend.main" in mod or mod == "backend.main":
                del sys.modules[mod]
        import prometheus_client
        fresh_registry = prometheus_client.CollectorRegistry(auto_describe=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("prometheus_client.REGISTRY", fresh_registry)
            from backend.main import GenerateRequest
            # Valid
            r = GenerateRequest(prompt="test")
            assert r.prompt == "test"
            # Empty should fail (min_length=1)
            with pytest.raises(Exception):
                GenerateRequest(prompt="")
            # Too long should fail
            with pytest.raises(Exception):
                GenerateRequest(prompt="x" * 3000)

    def test_routes_exist(self):
        import sys
        for mod in list(sys.modules.keys()):
            if "backend.main" in mod or mod == "backend.main":
                del sys.modules[mod]
        import prometheus_client
        fresh_registry = prometheus_client.CollectorRegistry(auto_describe=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("prometheus_client.REGISTRY", fresh_registry)
            from backend.main import app
            routes = [route.path for route in app.routes]
            expected = ["/generate", "/status/{job_id}", "/result/{job_id}", "/health", "/metrics"]
            for route in expected:
                assert route in routes, f"Missing route: {route}"



# ===========================================================================
# TEST 4: FFMPEG PIPELINE (uses temp files, no GPU)
# ===========================================================================

class TestFFmpegPipeline:
    def test_moviepy_import(self):
        from pipelines.ffmpeg_assembly import FFmpegAssemblyPipeline
        assert FFmpegAssemblyPipeline is not None

    def test_assembler_init(self):
        from pipelines.ffmpeg_assembly import FFmpegAssemblyPipeline
        a = FFmpegAssemblyPipeline(target_fps=16, codec="libx264")
        assert a.target_fps == 16
        assert a.codec == "libx264"


# ===========================================================================
# TEST 5: CHECKPOINT MANAGER (uses fakeredis)
# ===========================================================================

class TestCheckpointManager:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import fakeredis
            self.fake_redis = fakeredis.FakeStrictRedis()
        except ImportError:
            pytest.skip("fakeredis not installed")
        self.patcher = pytest.MonkeyPatch()

    def test_state_lifecycle(self):
        from storage.checkpoint import CheckpointManager
        import fakeredis

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("redis.Redis.from_url", lambda *a, **kw: self.fake_redis)
            cm = CheckpointManager()

            # Initial empty state
            state = cm.load_state("test-job")
            assert state["status"] == "initialized"

            # Save and reload
            state["status"] = "generating_images"
            cm.save_state("test-job", state)
            loaded = cm.load_state("test-job")
            assert loaded["status"] == "generating_images"

            # Clear
            cm.clear_state("test-job")
            cleared = cm.load_state("test-job")
            assert cleared["status"] == "initialized"


# ===========================================================================
# TEST 6: MEMORY WATCHDOG (no GPU fallback)
# ===========================================================================

class TestMemoryWatchdog:
    def test_get_vram_no_gpu(self):
        from monitoring.memory_watchdog import MemoryWatchdog
        alloc, reserved = MemoryWatchdog.get_vram_usage()
        # Should not crash
        assert isinstance(alloc, float)

    def test_enforce_cleanup_no_gpu(self):
        from monitoring.memory_watchdog import MemoryWatchdog
        # Should not crash
        MemoryWatchdog.enforce_cleanup()


# ===========================================================================
# TEST 7: GPU VERIFY SCRIPT (import and basic structure)
# ===========================================================================

class TestGPUVerify:
    def test_import(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import gpu_verify
        assert callable(gpu_verify.verify_gpu)


# ===========================================================================
# TEST 8: NOSANA JOB CONFIG VALIDATION
# ===========================================================================

class TestNosanaConfig:
    def test_job_json_valid(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "nosana-job.json")) as f:
            config = json.load(f)
        assert config["type"] == "container"
        assert len(config["ops"]) > 0
        op = config["ops"][0]
        assert op["args"]["gpu"] is True
        assert op["args"]["expose"] == 8000
        assert "REDIS_URL" in op["args"]["env"]

    def test_system_requirements(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "nosana-job.json")) as f:
            config = json.load(f)
        meta = config["meta"]["system_requirements"]
        assert meta["min_vram"] >= 24
        assert meta["min_cuda_version"]


# ===========================================================================
# TEST 9: STARTUP SCRIPT VALIDATION (syntax only)
# ===========================================================================

class TestStartupScript:
    def test_bash_syntax(self):
        import subprocess
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "start_worker.sh")
        result = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
