#!/bin/bash
set -e

echo "========================================"
echo " Starting Nosana Cinematic Worker Boot  "
echo "========================================"

# 1. Validate GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. This worker requires a GPU."
    exit 1
fi

echo "[INFO] GPU detected:"
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader

# 2. Check Disk Space (Require at least 50GB free for models + outputs)
FREE_SPACE=$(df -BG /app | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$FREE_SPACE" -lt 50 ]; then
    echo "[WARNING] Low disk space: Only ${FREE_SPACE}GB free. Models and video generation require significant space."
fi

# 3. Model Preloading and Cache Verification
echo "[INFO] Starting model preload and verification pipeline..."
export MODEL_DIR=/app/models
python3 /app/scripts/preload_models.py

echo "[INFO] Models verified successfully."

# 4. Start Background Worker & API
echo "[INFO] Launching Redis..."
# In a pure single-container deployment on Nosana, we need Redis running locally.
redis-server --daemonize yes
sleep 2

echo "[INFO] Starting RQ Worker in background..."
rq worker default --url redis://localhost:6379 --with-scheduler &

echo "[INFO] Starting FastAPI Backend on port 8000..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
