#!/bin/bash
set -eo pipefail

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
    rm -rf /root/.cache/pip 2>/dev/null || true
    rm -rf /tmp/* 2>/dev/null || true
fi

# 3. Model Preloading
echo "[INFO] Starting model verification pipeline..."
export MODEL_DIR="${MODEL_DIR:-/app/models}"
python3 /app/scripts/preload_models.py

# 4. Determine Redis URL
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
echo "[INFO] Redis URL: $REDIS_URL"

# 5. Start Redis only if connecting to localhost (embedded mode)
# In docker-compose or multi-container setups, Redis runs externally
if [[ "$REDIS_URL" == redis://localhost:* ]] || [[ "$REDIS_URL" == redis://127.0.0.1:* ]]; then
    echo "[INFO] Starting embedded Redis..."
    redis-server --daemonize yes

    # Wait for Redis to be ready
    MAX_RETRIES=15
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
else
    echo "[INFO] Using external Redis at $REDIS_URL"
    # Verify connection
    MAX_RETRIES=10
    COUNT=0
    while ! redis-cli -u "$REDIS_URL" ping &>/dev/null; do
        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo "[CRITICAL] Cannot connect to Redis at $REDIS_URL"
            exit 1
        fi
        echo "[INFO] Waiting for Redis... ($COUNT)"
        sleep 2
        ((COUNT++))
    done
    echo "[OK] Redis connection verified."
fi

# 6. Start RQ Worker (Background)
echo "[INFO] Starting RQ Worker..."
RQ_QUEUE="${RQ_QUEUE:-cinematic_jobs}"
rq worker "$RQ_QUEUE" --url "$REDIS_URL" --with-scheduler &
WORKER_PID=$!


# 7. Trap signals for graceful shutdown
cleanup() {
    echo "[INFO] Shutting down gracefully..."
    kill $WORKER_PID 2>/dev/null
    wait $WORKER_PID 2>/dev/null
    if [[ "$REDIS_URL" == redis://localhost:* ]] || [[ "$REDIS_URL" == redis://127.0.0.1:* ]]; then
        redis-cli shutdown 2>/dev/null
    fi
    echo "[INFO] Shutdown complete."
    exit 0
}
trap cleanup SIGTERM SIGINT

# 8. Start FastAPI (Foreground)
echo "[INFO] Starting FastAPI Backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# 9. Wait for either process to exit
wait -n $UVICORN_PID $WORKER_PID
CLEANUP_EXIT_CODE=$?
cleanup
exit $CLEANUP_EXIT_CODE
