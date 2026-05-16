# Nosana-Specific Deployment Recommendations

## Current Architecture Analysis

The current architecture runs Redis, RQ worker, FastAPI, and model inference in a **single container**. While this works for simple deployments, it has significant risks on Nosana.

## Single-Container Limitations

1. **No persistent storage** — models (27GB) re-downloaded every cold start
2. **No job state persistence** — Redis in-memory, container restart loses everything
3. **No GPU time-sharing** — A100 80GB VRAM is wasted on single-job sequential execution
4. **20-30 min cold start** — model download blocks API availability
5. **Not scalable** — can't separate API (CPU) from inference (GPU)

## Nosana Deployment-Manager (Recommended)

For production, use Nosana's deployment-manager for persistent services instead of single-shot jobs.

### Multi-Job Topology

```
Job 1: API + Redis (CPU-only, no GPU)
  - Runs FastAPI + Redis
  - Exposes port 8000
  - Cheaper (no GPU cost)

Job 2: Worker (GPU, connects to Job 1's Redis)
  - Runs RQ worker only
  - GPU reserved for inference
  - Can scale horizontally: multiple workers per API

Persistent Volume: /app/models (cache models across restarts)
```

### Nosana Job Config for Persistent Service

```json
{
  "ops": [
    {
      "id": "cinematic-api",
      "args": {
        "cmd": ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"],
        "gpu": false,
        "image": "docker.io/naveendhaterwal/vimax:latest",
        "expose": 8000,
        "env": {
          "REDIS_URL": "redis://<redis-service-url>:6379",
          "MODEL_DIR": "/app/models",
          "OUTPUT_DIR": "/app/outputs"
        }
      },
      "type": "container/run"
    },
    {
      "id": "cinematic-worker-1",
      "args": {
        "cmd": ["rq", "worker", "default", "--url", "redis://<redis-service-url>:6379"],
        "gpu": true,
        "image": "docker.io/naveendhaterwal/vimax:latest",
        "env": {
          "REDIS_URL": "redis://<redis-service-url>:6379",
          "MODEL_DIR": "/app/models",
          "OUTPUT_DIR": "/app/outputs"
        }
      },
      "type": "container/run"
    }
  ],
  "type": "container",
  "version": "0.3"
}
```

## Model Caching Strategy

1. **Pre-bake base models** into a derived Docker image:
   ```
   FROM naveendhaterwal/vimax:latest AS model-baked
   RUN python3 /app/scripts/preload_models.py
   ```
   This creates a ~35GB image but eliminates cold-start download time.

2. **Use `start_worker.sh` checking logic**: If models exist in `/app/models`, skip download entirely.

3. **Use Nosana deployment-manager persistent volumes**: Mount a volume at `/app/models` that persists across container restarts.

## GPU Selection

| GPU | VRAM | bf16 Support | Verdict |
|-----|------|-------------|---------|
| RTX 4090 | 24GB | Yes | Marginal — OOM risk at default settings |
| RTX 3090 | 24GB | Yes | Marginal — same VRAM, slower compute |
| A100 | 40/80GB | Yes | Safe — plenty of headroom |
| H100 | 80GB | Yes | Safe — ideal |
| L40S | 48GB | Yes | Safe |
| A6000 | 48GB | Yes | Safe |

**Recommendation:** Target A100 40GB minimum for production. RTX 4090 is acceptable for dev/light usage with quality tuning.

## Cost Optimization

- Use `enable_model_cpu_offload()` only if needed (costs ~30% inference time)
- Reduce `num_inference_steps` from 50 to 25 for 2x speed at slight quality drop
- Reduce `num_frames` from 81 to 49 for ~40% less VRAM
- Set `job_timeout` to match expected wall time (don't over-allocate)
