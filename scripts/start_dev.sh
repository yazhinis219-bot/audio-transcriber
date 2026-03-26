#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# NHS Real-Time Transcription — Local Development Startup Script
# Starts Redis, Celery worker, FastAPI server, and frontend in parallel.
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[nhs]${NC} $1"; }
ok()   { echo -e "${GREEN}[ok]${NC}  $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
fail() { echo -e "${RED}[err]${NC}  $1"; exit 1; }

# PIDs file for cleanup
PID_FILE="/tmp/nhs_transcription.pids"
> "$PID_FILE"

cleanup() {
    echo ""
    log "Shutting down all processes..."
    while read -r pid; do
        kill "$pid" 2>/dev/null && echo "  killed $pid"
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    log "Done."
}
trap cleanup EXIT INT TERM

# ── Check prerequisites ───────────────────────────────────────────────────────
log "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
command -v redis-cli >/dev/null 2>&1 || warn "redis-cli not found — make sure Redis is running"
command -v node >/dev/null 2>&1     || warn "node not found — frontend won't start"
command -v npm >/dev/null 2>&1      || warn "npm not found — frontend won't start"

# ── Load env ──────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    ok "Loaded .env"
else
    warn ".env not found — using defaults (copy .env.example to .env)"
    export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
    export WHISPER_MODEL=${WHISPER_MODEL:-base.en}
    export PORT=${PORT:-8000}
fi

# ── Python virtualenv ─────────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/backend/.venv"
if [ ! -d "$VENV_DIR" ]; then
    log "Creating Python virtualenv..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
log "Virtualenv activated: $VENV_DIR"

# Install/upgrade backend deps
log "Installing backend dependencies..."
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt
ok "Backend dependencies installed"

# ── Start Redis ───────────────────────────────────────────────────────────────
if redis-cli -u "${REDIS_URL}" ping 2>/dev/null | grep -q PONG; then
    ok "Redis already running"
else
    log "Starting Redis..."
    redis-server --daemonize yes --logfile /tmp/nhs_redis.log --port 6379
    sleep 1
    redis-cli ping | grep -q PONG && ok "Redis started" || fail "Redis failed to start"
fi

# ── Start Celery Worker ───────────────────────────────────────────────────────
log "Starting Celery transcription worker..."
(
    cd backend
    export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
    celery -A workers.transcription_worker.celery_app worker \
        --loglevel=info \
        --concurrency="${CELERY_CONCURRENCY:-2}" \
        --queues=transcription \
        --pool=threads \
        --logfile=/tmp/nhs_celery.log \
        &
    echo $! >> "$PID_FILE"
    wait
) &

sleep 3
ok "Celery worker started (logs: /tmp/nhs_celery.log)"

# ── Start FastAPI Server ──────────────────────────────────────────────────────
log "Starting FastAPI server on port ${PORT}..."
(
    cd backend
    export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
    uvicorn api.server:app \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --reload \
        --ws websockets \
        --log-level info \
        2>&1 | tee /tmp/nhs_api.log &
    echo $! >> "$PID_FILE"
    wait
) &

sleep 2
ok "FastAPI server started → http://localhost:${PORT}"

# ── Start Frontend ────────────────────────────────────────────────────────────
if command -v npm >/dev/null 2>&1; then
    log "Installing frontend dependencies..."
    (cd frontend && npm install --silent)
    ok "Frontend dependencies installed"

    log "Starting Vite dev server on port 3000..."
    (
        cd frontend
        VITE_WS_URL="ws://localhost:${PORT}/ws/transcribe" \
        npm run dev -- --host 0.0.0.0 --port 3000 \
        2>&1 | tee /tmp/nhs_frontend.log &
        echo $! >> "$PID_FILE"
        wait
    ) &
    sleep 3
    ok "Frontend started → http://localhost:3000"
else
    warn "npm not found — skipping frontend. Open http://localhost:${PORT} instead."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  NHS Real-Time Transcription — Running${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${CYAN}Frontend${NC}   →  http://localhost:3000"
echo -e "  ${CYAN}API${NC}        →  http://localhost:${PORT}"
echo -e "  ${CYAN}WebSocket${NC}  →  ws://localhost:${PORT}/ws/transcribe"
echo -e "  ${CYAN}Health${NC}     →  http://localhost:${PORT}/health"
echo -e "  ${CYAN}Metrics${NC}    →  http://localhost:${PORT}/metrics"
echo -e "  ${CYAN}Whisper${NC}    →  ${WHISPER_MODEL}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Logs:  /tmp/nhs_api.log  /tmp/nhs_celery.log"
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop all services"
echo ""

# Keep alive
wait
