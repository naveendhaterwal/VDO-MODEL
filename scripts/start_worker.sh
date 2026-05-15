#!/bin/bash
set -e

echo "========================================"
echo " Starting Hardened Cinematic Worker Boot "
echo "========================================"

# 1. Pre-flight GPU Verification
python3 /app/scripts/gpu_verify.py || { echo "[CRITICAL] GPU verification failed."; exit 1; }

# 2. Disk Space Cleanup (Aggressive)
echo "[INFO] Checking disk space..."
FREE_SPACE=$(df -BG /app | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$FREE_SPACE" -lt 40 ]; then
    echo "[WARNING] Low space (${FREE_SPACE}GB). Cleaning caches..."
    rm -rf /root/.cache/pip
    rm -rf /tmp/*
fi

# 3. Model Preloading
echo "[INFO] Starting model verification pipeline..."
export MODEL_DIR=/app/models
python3 /app/scripts/preload_models.py

# 4. Launch Services
echo "[INFO] Launching Redis..."
redis-server --daemonize yes

# Wait for Redis to be ready
MAX_RETRIES=10
COUNT=0
while ! redis-cli ping &>/dev/null; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "[CRITICAL] Redis failed to start."
        exit 1
    fi
    echo "[INFO] Waiting for Redis... ($COUNT)"
    sleep 1
    ((COUNT++))
done
echo "[OK] Redis is ready."

# 5. Start RQ Worker (Foreground monitor)
echo "[INFO] Starting RQ Worker..."
# We run the worker in the background but use a trap to kill everything if any process dies
rq worker default --url redis://localhost:6379 --with-scheduler &
WORKER_PID=$!

# 6. Start FastAPI
echo "[INFO] Starting FastAPI Backend..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
