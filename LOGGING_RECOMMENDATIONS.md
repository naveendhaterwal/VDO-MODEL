# Production Logging Recommendations

## Structured JSON Logging

Replace plain print/logging with structured JSON for log aggregation tools.

**Add to `backend/main.py`:**
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
```

## Log Levels per Component

| Component | Level | Notes |
|-----------|-------|-------|
| `backend.main` | INFO | Request logging via middleware |
| `workers.generation_worker` | INFO | Job lifecycle |
| `orchestration.pipeline` | INFO | Pipeline stage transitions |
| `providers.Local*` | INFO | Model load/unload |
| `monitoring.memory_watchdog` | WARNING | Only VRAM warnings |
| `pipelines.ffmpeg_assembly` | INFO | Assembly progress |
| `storage.checkpoint` | WARNING | Only errors |
| `rq.worker` | WARNING | Worker lifecycle |
| `diffusers` | WARNING | Model loading info |
| `transformers` | WARNING | Tokenizer/model load info |
| `torch` | ERROR | Only errors |

## Request Logging

The Prometheus middleware already tracks request count and latency. Add request ID:
```python
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

## Critical Events to Always Log

- Model load/unload with VRAM stats
- Pipeline stage transitions with timing
- OOM warnings with VRAM snapshot
- Download failures with retry count
- Worker heartbeats (every 30s)
- Container shutdown (SIGTERM received)
