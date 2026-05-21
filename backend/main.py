import os
import uuid
import gc
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.job import Job
import time
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")





from workers.generation_worker import generate_video_task

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
RQ_QUEUE = os.environ.get("RQ_QUEUE", "cinematic_jobs")

redis_conn = Redis.from_url(REDIS_URL)
q = Queue(RQ_QUEUE, connection=redis_conn)


from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

REQUEST_COUNT = Counter("api_requests_total", "Total requests received", ["endpoint", "method"], registry=REGISTRY)
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"], registry=REGISTRY)
JOB_CREATED = Counter("jobs_created_total", "Total generation jobs enqueued", registry=REGISTRY)



@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Shutting down...")
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")


app = FastAPI(title="Nosana Cinematic Video API", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
    except Exception:
        body = b""
    logger.error(f"Validation error: {exc.errors()} | Body: {body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(body)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    path = request.url.path
    if path not in ["/metrics"]:
        REQUEST_COUNT.labels(endpoint=path, method=request.method).inc()
        REQUEST_LATENCY.labels(endpoint=path).observe(duration)
    return response


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: str | None = None


@app.post("/generate", response_model=JobStatus)
async def generate_video(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job = q.enqueue(
        generate_video_task,
        job_id,
        req.prompt,
        job_id=job_id,
        job_timeout="2h",
    )
    JOB_CREATED.inc()
    return JobStatus(job_id=job.id, status="queued")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()
    result = None
    if status == "finished":
        result_dict = job.result
        if result_dict and "video_path" in result_dict:
            result = f"/result/{job_id}"
    elif status == "failed":
        result = str(job.exc_info)

    return JobStatus(job_id=job_id, status=status, result=result)


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    safe_job_id = job_id.replace("/", "_").replace("..", "_")
    video_path = os.path.join(OUTPUT_DIR, f"{safe_job_id}_final.mp4")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")
    raise HTTPException(status_code=404, detail="Video not ready or not found")


@app.get("/health")
async def health_check():
    try:
        redis_conn.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Mount frontend static files if present (e.g. in Docker container), otherwise skip to allow local testing
static_dir = os.environ.get("STATIC_DIR", "/app/frontend/out")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
else:
    logger.warning(f"Static directory {static_dir} not found. Skipping frontend mount.")
