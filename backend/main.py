import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

from workers.generation_worker import generate_video_task

app = FastAPI(title="Nosana Cinematic Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total requests received", ["endpoint", "method"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
JOB_CREATED = Counter("jobs_created_total", "Total generation jobs enqueued")

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    path = request.url.path
    if path not in ["/metrics"]: # Exclude metrics from cluttering logs
        REQUEST_COUNT.labels(endpoint=path, method=request.method).inc()
        REQUEST_LATENCY.labels(endpoint=path).observe(duration)
        
    return response

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(redis_url)
q = Queue('default', connection=redis_conn)

class GenerateRequest(BaseModel):
    prompt: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: str = None

@app.post("/generate", response_model=JobStatus)
async def generate_video(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job = q.enqueue(
        generate_video_task,
        job_id,
        req.prompt,
        job_id=job_id,
        job_timeout='2h' # Increased to 2h for robust retries
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
        # The result of our worker function is a dict with the video_path
        result_dict = job.result
        if result_dict and "video_path" in result_dict:
            result = f"/result/{job_id}"
            
    elif status == "failed":
        result = str(job.exc_info)

    return JobStatus(job_id=job_id, status=status, result=result)

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    video_path = f"/app/outputs/{job_id}_final.mp4"
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=f"{job_id}.mp4")
    raise HTTPException(status_code=404, detail="Video not ready or not found")

@app.get("/health")
async def health_check():
    """Liveness probe for Nosana/Kubernetes"""
    try:
        redis_conn.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Mount the static frontend build
app.mount("/", StaticFiles(directory="/app/frontend/out", html=True), name="frontend")
