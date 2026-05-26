#!/usr/bin/env bash
set -Eeuo pipefail

REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
RQ_QUEUE="${RQ_QUEUE:-cinematic_jobs}"
MODEL_DIR="${MODEL_DIR:-/app/models}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/outputs}"
SHUTTING_DOWN=0
WORKER_PID=""
UVICORN_PID=""

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1"
}

cleanup() {
  local exit_code="${1:-0}"
  SHUTTING_DOWN=1

  log "Shutting down services..."

  if [[ -n "${UVICORN_PID}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi

  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill "${WORKER_PID}" 2>/dev/null || true
    wait "${WORKER_PID}" 2>/dev/null || true
  fi

  if [[ "${REDIS_URL}" == redis://localhost:* ]] || [[ "${REDIS_URL}" == redis://127.0.0.1:* ]]; then
    redis-cli shutdown 2>/dev/null || true
  fi

  log "Shutdown complete with exit code ${exit_code}."
  exit "${exit_code}"
}

trap 'cleanup 143' SIGTERM SIGINT

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"

log "========================================"
log " Starting Production Cinematic Worker Boot "
log "========================================"
log "MODEL_DIR=${MODEL_DIR}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "REDIS_URL=${REDIS_URL}"
log "RQ_QUEUE=${RQ_QUEUE}"

python /app/scripts/gpu_verify.py
python /app/scripts/preload_models.py

if [[ "${REDIS_URL}" == redis://localhost:* ]] || [[ "${REDIS_URL}" == redis://127.0.0.1:* ]]; then
  log "Starting embedded Redis..."
  redis-server --daemonize yes
  for attempt in $(seq 1 15); do
    if redis-cli ping >/dev/null 2>&1; then
      log "Embedded Redis is ready."
      break
    fi
    if [[ "${attempt}" == "15" ]]; then
      log "Redis failed to start."
      exit 1
    fi
    sleep 1
  done
else
  log "Using external Redis at ${REDIS_URL}"
  for attempt in $(seq 1 10); do
    if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      log "External Redis connection verified."
      break
    fi
    if [[ "${attempt}" == "10" ]]; then
      log "Cannot connect to external Redis."
      exit 1
    fi
    sleep 2
  done
fi

log "Starting RQ worker..."
rq worker "${RQ_QUEUE}" --url "${REDIS_URL}" --with-scheduler &
WORKER_PID=$!

log "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

log "Services started. worker_pid=${WORKER_PID} api_pid=${UVICORN_PID}"

EXITED_PID=""
wait -n -p EXITED_PID "${UVICORN_PID}" "${WORKER_PID}"
CHILD_EXIT_CODE=$?

if [[ "${SHUTTING_DOWN}" == "0" ]]; then
  log "Child process ${EXITED_PID} exited unexpectedly with code ${CHILD_EXIT_CODE}."
fi

cleanup "${CHILD_EXIT_CODE}"
