# NHS Real-Time Transcription System

Production-grade live audio transcription for NHS online consultations.
Built with Whisper, FastAPI WebSockets, Redis queues, Celery workers, and React.

---

## Architecture

```
Browser Mic
    │
    │  PCM Int16 binary frames (WebSocket)
    ▼
FastAPI WebSocket server
    │
    ├─ AudioPreprocessor (VAD, silence detection, chunking)
    │      └── WebRTC VAD (30ms frames)
    │          └── silence_threshold=800ms → flush utterance
    │          └── context_keep=1500ms → overlap for re-transcription
    │
    ├─ Redis Queue (per session FIFO)
    │      └── RPUSH on enqueue, BLPOP on consume
    │
    └─ Celery Task dispatch ──────────────────────────────┐
                                                          │
                                          Celery Worker (×N)
                                              │
                                          Whisper (loaded once)
                                              │  greedy CTC, beam_size=1
                                              │  no_speech_threshold=0.6
                                              │  suppress hallucination tokens
                                              │
                                          Redis Pub/Sub publish result
                                                          │
    FastAPI result relay ◄───────────────────────────────┘
    (subscribed to session channel)
    │
    │  JSON {"type":"transcript","text":"..."}
    ▼
Browser — React UI displays utterance
```

---

## File Structure

```
nhs-transcription/
├── backend/
│   ├── api/
│   │   └── server.py           ← FastAPI + WebSocket endpoint
│   ├── core/
│   │   ├── audio_processor.py  ← VAD, silence detection, chunking
│   │   ├── transcriber.py      ← Whisper wrapper + hallucination filters
│   │   └── queue_manager.py    ← Redis async/sync queue operations
│   ├── workers/
│   │   └── transcription_worker.py  ← Celery task (Whisper inference)
│   ├── config/
│   │   └── settings.py         ← Pydantic settings (reads from .env)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── hooks/
│   │   │   └── useTranscription.js  ← AudioWorklet + WebSocket hook
│   │   ├── App.jsx             ← NHS-styled React UI
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── tests/
│   └── test_pipeline.py        ← Integration tests (no browser needed)
├── scripts/
│   └── start_dev.sh            ← One-command local start
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start (Docker — recommended)

### Prerequisites
- Docker 24+ and Docker Compose v2
- 4 GB RAM minimum (8 GB recommended for medium.en model)
- Microphone access in browser

### Step 1 — Clone and configure

```bash
git clone <your-repo>
cd nhs-transcription
cp .env.example .env
```

Edit `.env` to choose your Whisper model:
```
WHISPER_MODEL=base.en    # fast, good for testing
# WHISPER_MODEL=small.en  # better accuracy
# WHISPER_MODEL=medium.en # recommended for NHS production
```

### Step 2 — Start everything

```bash
docker compose up --build
```

First run downloads the Whisper model (~74 MB for base.en, ~244 MB for small.en).
Subsequent starts use the cached model.

### Step 3 — Open the app

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| Health check | http://localhost:8000/health |
| Prometheus metrics | http://localhost:8000/metrics |
| Celery Flower (worker monitor) | http://localhost:5555 |

### Step 4 — Use

1. Open http://localhost:3000
2. Click **Start Consultation**
3. Allow microphone access
4. Speak — transcription appears after each sentence pause
5. Click **Stop Recording** to end

---

## Local Development (without Docker)

### Prerequisites

```bash
# Python 3.10+
python3 --version

# Redis
# macOS:
brew install redis
# Ubuntu/Debian:
sudo apt-get install redis-server
# Windows: use WSL2 + Ubuntu

# Node.js 18+
node --version
npm --version

# FFmpeg (required by Whisper)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg
```

### Option A — One command

```bash
chmod +x scripts/start_dev.sh
./scripts/start_dev.sh
```

This script:
1. Creates a Python virtualenv in `backend/.venv`
2. Installs all Python dependencies
3. Starts Redis (if not already running)
4. Starts the Celery transcription worker
5. Starts the FastAPI server with hot-reload
6. Installs Node dependencies and starts Vite dev server
7. Prints all URLs and waits — Ctrl+C stops everything

### Option B — Manual (4 terminals)

**Terminal 1 — Redis**
```bash
redis-server
```

**Terminal 2 — Celery worker**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export REDIS_URL=redis://localhost:6379/0
export WHISPER_MODEL=base.en
export PYTHONPATH=$(pwd)

celery -A workers.transcription_worker.celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    --queues=transcription \
    --pool=threads
```

**Terminal 3 — FastAPI server**
```bash
cd backend
source .venv/bin/activate

export REDIS_URL=redis://localhost:6379/0
export WHISPER_MODEL=base.en
export PYTHONPATH=$(pwd)

uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload --ws websockets
```

**Terminal 4 — Frontend**
```bash
cd frontend
npm install
VITE_WS_URL=ws://localhost:8000/ws/transcribe npm run dev
```

Open http://localhost:3000

---

## Running Tests (no browser needed)

```bash
cd backend
source .venv/bin/activate
export PYTHONPATH=$(pwd)

# Test with synthetic audio (no WAV file needed)
python ../tests/test_pipeline.py

# Test with a real WAV file (16kHz mono recommended, any sample rate accepted)
python ../tests/test_pipeline.py /path/to/your/audio.wav
```

This tests:
1. VAD + silence detection (does it emit the right number of chunks?)
2. Whisper transcription (does it transcribe without hallucinating on silence?)
3. Redis queue push/pop (is Redis reachable and working?)

---

## Scaling for Production

### Scale workers (more concurrent sessions)

```bash
# docker-compose — run 4 worker instances
docker compose up --scale worker=4
```

Or set `CELERY_CONCURRENCY=4` in `.env` for 4 threads per worker.

### Use a better model (GPU)

```bash
# Set in .env:
WHISPER_MODEL=medium.en

# In docker-compose.yml, uncomment the GPU deploy block for worker:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

### Nginx reverse proxy (production)

Add nginx in front of the API to terminate TLS and proxy WebSockets:

```nginx
server {
    listen 443 ssl;
    server_name your.nhs.domain;

    location /ws/ {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }

    location / {
        proxy_pass http://api:8000;
    }
}
```

---

## Key Design Decisions

### Why queues AND direct WebSocket?

| Segment | Mechanism | Reason |
|---------|-----------|--------|
| Mic → browser AudioWorklet | Shared memory ring buffer | Hardware/software clock decoupling |
| Browser → FastAPI | Direct WebSocket binary stream | Minimum latency (no broker overhead) |
| FastAPI → Whisper | Redis queue + Celery | Decouple I/O from heavy GPU inference |
| Whisper → browser | Redis Pub/Sub → WebSocket relay | Non-blocking result delivery |

### Why not stream audio directly into Whisper?

Whisper is not a streaming model — it was trained on fixed 30-second windows.
The AudioPreprocessor solves this by:
1. Running WebRTC VAD on every 30ms frame
2. Accumulating speech frames
3. Detecting silence >= 800ms (end of utterance)
4. Emitting just that utterance (typically 1–10 seconds) to Whisper
5. Prepending 1500ms of context for accurate boundary transcription

### Why does it keep context frames?

The last 1500ms of the previous utterance is prepended to the next one.
This prevents Whisper from mis-transcribing words at sentence boundaries
(a known issue with streaming + VAD segmentation).

### Hallucination suppression

When Whisper sees near-silence it hallucinates phrases like "Thank you for watching",
"Subscribe", or just "you". Three defences:
1. `no_speech_threshold=0.6` — rejects chunks Whisper itself rates as non-speech
2. `logprob_threshold=-1.2` — rejects low-confidence output
3. `suppress_tokens` — explicitly suppresses known hallucination token IDs

---

## Troubleshooting

**"Microphone not found"**
- Ensure the browser has microphone permission
- Chrome/Edge: `chrome://settings/content/microphone`
- Firefox: click the lock icon in the address bar

**"WebSocket connection failed"**
- Check the API server is running: `curl http://localhost:8000/health`
- Check `VITE_WS_URL` points to the correct host

**Whisper model download stuck**
- First run downloads the model. Check network connectivity.
- Manual download: `python -c "import whisper; whisper.load_model('base.en')"`

**"No transcripts appearing" despite speaking**
- Check Celery worker is running: `celery -A workers.transcription_worker.celery_app inspect active`
- Check Redis: `redis-cli ping` → should return `PONG`
- Try louder/clearer speech — VAD may be filtering background noise
- Lower `VAD_AGGRESSIVENESS` to `1` in `.env`

**High latency (> 3 seconds)**
- Switch to `WHISPER_MODEL=tiny.en` for faster inference on CPU
- Add GPU support (see Scaling section)
- Reduce `SILENCE_THRESHOLD_MS` to `500` for faster flush

---

## NHS Clinical Considerations

- All audio is processed in-memory; no audio is written to disk by default
- Add HTTPS/WSS for production deployments (patient data in transit)
- Consider deploying within NHS N3/HSCN network for data sovereignty
- Whisper `base.en` / `small.en` are suitable for general English speech
- For accented speech or medical terminology, consider fine-tuning on NHS-specific data
- Implement session logging and audit trails per NHS DSP Toolkit requirements
