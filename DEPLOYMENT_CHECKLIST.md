# Production Deployment Checklist

## Pre-Deployment

- [ ] `.dockerignore` exists and excludes: .git/, models_bundle/, node_modules/, .pytest_cache/
- [ ] `Dockerfile.worker` uses `cudnn8-runtime` (not `devel`) base image
- [ ] No `pydantic<2.0.0` pin (let fastapi pull pydantic v2)
- [ ] `numpy<2.0.0` pinned
- [ ] `xformers==0.0.27` installed
- [ ] `hf_transfer` removed (not imported, env var unset)
- [ ] `REDIS_URL` env var respected by `start_worker.sh`
- [ ] No `exec uvicorn` — process manager handles background workers
- [ ] Trap handlers installed for SIGTERM/SIGINT cleanup
- [ ] `gc.collect()` called BEFORE `torch.cuda.empty_cache()` everywhere
- [ ] `heartbeat_thread.join(timeout=10)` has timeout parameter
- [ ] `concatenate_videoclips` uses `method="chain"` (not `"compose"`)
- [ ] `max_workers=8` in `preload_models.py`
- [ ] `local_dir_use_symlinks=True` in `preload_models.py`
- [ ] `REDIS_URL` in `nosana-job.json` matches `start_worker.sh` default
- [ ] `GPU_verify.py` threshold matches actual VRAM requirements (22GB min)
- [ ] GPU capability detection for bfloat16 fallback added

## Docker Build

- [ ] Build context is clean: `docker build -t test .` succeeds
- [ ] Image size < 15GB (check with `docker images`)
- [ ] Frontend builds without errors
- [ ] All pip packages resolve without conflicts
- [ ] `docker run --gpus all test` passes GPU verification
- [ ] Model download completes without segfault

## CI/CD

- [ ] GitHub Actions workflow runs tests before building
- [ ] Docker cache uses buildx with `mode=max`
- [ ] Tags include SHA-based unique tags + `latest`
- [ ] Docker secrets stored as GitHub Actions secrets
- [ ] Smoke tests run in CI (not requiring GPU)

## Nosana Deployment

- [ ] Image pushed to public Docker Hub repository
- [ ] `nosana-job.json` has correct `expose` port (8000)
- [ ] `required_vram: 24` set in system_requirements
- [ ] `min_cuda_version: "12.1"` set
- [ ] Wallet has enough SOL for gas + NOS for compute
- [ ] Cold-start timeout expectation documented (20-30 min model download)
