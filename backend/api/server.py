"""
server.py — Production Real-Time Transcription Server
======================================================
Architecture:
  - Run with: uvicorn api.server:app --workers 4 --host 0.0.0.0 --port 8000
  - Each OS worker process owns its own WhisperTranscriber (model loaded per process)
  - Per session: full audio accumulation (bytearray, no trimming, no chunk loss)
  - Live partials: rolling read-only window every PARTIAL_INTERVAL_S seconds
  - Final flush: single full-buffer Whisper inference on stop/disconnect
  - Admission control: RAM-based (rejects when free RAM < MIN_FREE_RAM_MB or RAM% > MAX_RAM_PCT)
  - No hard session count limit — server handles as many clients as hardware allows
  - ThreadPoolExecutor(max_workers=2) per worker → 8 threads total, balanced across cores
"""

import os
import uuid
import time
import json
import asyncio
import threading
import traceback
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.transcriber import WhisperTranscriber
from core.punctuator import fix_punctuation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Per-worker config ──────────────────────────────────────────────────────────
# Admission control: RAM-based thresholds (not a hard session count)
# The server accepts connections until real hardware pressure forces a rejection.
MIN_FREE_RAM_MB  = 600   # Reject if system free RAM drops below this
MAX_RAM_PCT      = 94.0  # Reject if system RAM% exceeds this
MAX_CPU_PCT      = 98.0  # Reject if CPU has been pegged at this for >10s (future)

# Hard ceiling: absolute safety net — never allow more than this per worker
# Set high so RAM threshold fires first under normal conditions.
HARD_SESSION_CEIL = 30

# Thread pool for Whisper inference: 2 per worker process.
# 4 workers × 2 threads = 8 active inference threads across all cores.
EXECUTOR_THREADS = 2

# Live partial transcription interval (rolling read-only window)
PARTIAL_INTERVAL_S = 3.0          # How often to run rolling-window inference
PARTIAL_WINDOW_S   = 10.0         # How much audio to feed for partials (seconds)

# Deduplication: ignore words already confirmed within this many seconds
DEDUP_MARGIN_S     = 0.1

SAMPLE_RATE        = 16000        # 16kHz PCM int16
BYTES_PER_SAMPLE   = 2            # int16
SAMPLES_PER_CHUNK  = int(SAMPLE_RATE * 0.1)   # 100ms per chunk
BYTES_PER_CHUNK    = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

OUTPUT_DIR         = r"d:\dummy\transcripts"

# ── Process-level globals (one per uvicorn worker) ─────────────────────────────
_transcriber: WhisperTranscriber  = None   # type: ignore
_executor:    ThreadPoolExecutor  = None   # type: ignore
_active_lock  = threading.Lock()
_active_sessions: int = 0
_total_sessions_served: int = 0    # lifetime counter for monitoring
_total_sessions_rejected: int = 0  # rejected by RAM pressure


def _get_proc_ram_mb() -> float:
    """Real RSS RAM of this worker process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _admission_check() -> tuple[bool, str]:
    """
    Dynamic RAM-based admission gate.
    Returns (allowed: bool, reason: str).
    """
    # Hard ceiling — absolute safety net
    if _active_sessions >= HARD_SESSION_CEIL:
        return False, f"hard ceiling reached ({_active_sessions}/{HARD_SESSION_CEIL})"

    vm = psutil.virtual_memory()
    free_mb = vm.available / (1024 * 1024)
    ram_pct = vm.percent

    if free_mb < MIN_FREE_RAM_MB:
        return False, (
            f"insufficient free RAM: {free_mb:.0f}MB free "
            f"(threshold={MIN_FREE_RAM_MB}MB)  RAM={ram_pct:.1f}%"
        )

    if ram_pct > MAX_RAM_PCT:
        return False, (
            f"RAM pressure too high: {ram_pct:.1f}% "
            f"(threshold={MAX_RAM_PCT}%)  free={free_mb:.0f}MB"
        )

    return True, "ok"


async def _resource_monitor():
    pid = os.getpid()
    while True:
        try:
            vm      = psutil.virtual_memory()
            cpu     = psutil.cpu_percent()
            proc_mb = _get_proc_ram_mb()
            free_mb = vm.available / (1024 * 1024)
            allowed, reason = _admission_check()
            gate = "OPEN" if allowed else f"CLOSED ({reason})"
            logger.info(
                f"active={_active_sessions}  served={_total_sessions_served}  "
                f"rejected={_total_sessions_rejected}  "
                f"proc_ram={proc_mb:.0f}MB  "
                f"sys_ram={vm.percent:.1f}%  free={free_mb:.0f}MB  "
                f"CPU={cpu:.1f}%  gate={gate}"
            )
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber, _executor
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _executor = ThreadPoolExecutor(
        max_workers=EXECUTOR_THREADS,
        thread_name_prefix=f"whisper-w{os.getpid()}"
    )
    _transcriber = WhisperTranscriber(
        model_name="small.en",
        no_speech_threshold=0.55,
        logprob_threshold=-0.7,
        cpu_threads=2,
        num_workers=1,
    )
    logger.info(
        f"Worker ready. ceil={HARD_SESSION_CEIL} "
        f"min_free_ram={MIN_FREE_RAM_MB}MB max_ram={MAX_RAM_PCT}% "
        f"executor_threads={EXECUTOR_THREADS} "
        f"proc_ram={_get_proc_ram_mb():.0f}MB"
    )
    monitor_task = asyncio.create_task(_resource_monitor())
    yield
    monitor_task.cancel()
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan, title="Transcription Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/status")
async def status():
    """Expose per-worker health. Load balancer will round-robin across workers."""
    vm = psutil.virtual_memory()
    free_mb = vm.available / (1024 * 1024)
    allowed, gate_reason = _admission_check()
    return JSONResponse({
        "pid":                    os.getpid(),
        "active_sessions":        _active_sessions,
        "hard_session_ceil":      HARD_SESSION_CEIL,
        "sessions_served":        _total_sessions_served,
        "sessions_rejected":      _total_sessions_rejected,
        "admission_gate":         "open" if allowed else "closed",
        "admission_reason":       gate_reason,
        "min_free_ram_mb":        MIN_FREE_RAM_MB,
        "max_ram_pct":            MAX_RAM_PCT,
        "executor_threads":       EXECUTOR_THREADS,
        "proc_ram_mb":            round(_get_proc_ram_mb(), 1),
        "sys_ram_pct":            vm.percent,
        "sys_ram_free_mb":        round(free_mb, 0),
        "sys_ram_free_gb":        round(vm.available / 1e9, 2),
        "cpu_pct":                psutil.cpu_percent(),
    })


# ── WebSocket Handler ──────────────────────────────────────────────────────────
@app.websocket("/ws/consultation/{session_id}")
async def websocket_transcribe(websocket: WebSocket, session_id: str):
    global _active_sessions

    # ── Admission control (RAM-based) ─────────────────────────────────────────
    with _active_lock:
        allowed, reason = _admission_check()
        if not allowed:
            # Reject before accepting the WebSocket handshake
            await websocket.close(code=4008, reason=f"Server at capacity: {reason}")
            global _total_sessions_rejected
            _total_sessions_rejected += 1
            vm = psutil.virtual_memory()
            logger.warning(
                f"[{session_id}] REJECTED — {reason}  "
                f"(active={_active_sessions}  "
                f"free={vm.available/1e6:.0f}MB  RAM={vm.percent:.1f}%)"
            )
            return
        _active_sessions += 1
        global _total_sessions_served
        _total_sessions_served += 1

    await websocket.accept()

    # ── Session state ─────────────────────────────────────────────────────────
    pid            = os.getpid()
    session_start  = time.time()
    ts_start_ms    = session_start * 1000

    # Full audio accumulation — raw int16 bytes, NEVER trimmed
    full_audio_bytes = bytearray()

    # Track position of last confirmed word (in seconds of audio) to deduplicate partials
    last_confirmed_s = 0.0

    # Ordered list of all confirmed transcript segments for final file
    session_texts: list[str] = []

    # Chunk counter (for telemetry)
    chunks_received = 0

    # Paths
    telemetry_path  = os.path.join(OUTPUT_DIR, f"telemetry_{session_id}.tsv")
    transcript_path = os.path.join(OUTPUT_DIR, f"transcript_{session_id}.txt")

    with open(telemetry_path, "w", encoding="utf-8") as f:
        f.write("UtteranceID\tType\tAudioSentTS\tProcessedTS\tLatencyMS\tInferenceMS\tAudioDurationS\tText\n")

    def fmt_ts(ms: float) -> str:
        return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S.%f")[:-3]

    def append_telemetry(uid: str, kind: str, ts_sent_ms: float, inf_ms: float, dur_s: float, text: str):
        now_ms = time.time() * 1000
        try:
            with open(telemetry_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{uid}\t{kind}\t{fmt_ts(ts_sent_ms)}\t{fmt_ts(now_ms)}\t"
                    f"{int(now_ms - ts_sent_ms)}\t{int(inf_ms)}\t{dur_s:.2f}\t{text}\n"
                )
        except Exception:
            pass

    def audio_bytes_to_float32(raw: bytearray) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # ── Send helpers ──────────────────────────────────────────────────────────
    async def send_partial(text: str, dur_s: float):
        if not text.strip():
            return
        try:
            await websocket.send_json({
                "type":        "partial",
                "text":        text,
                "session_id":  session_id,
                "pid":         pid,
                "audio_s":     round(dur_s, 2),
                "active_sessions": _active_sessions,
            })
        except Exception:
            pass

    async def send_transcript(text: str, uid: str, chunk_idx: int, inf_ms: float, is_final: bool):
        if not text.strip():
            return
        try:
            await websocket.send_json({
                "type":            "transcript",
                "text":            text,
                "utterance_id":    uid,
                "chunk_index":     chunk_idx,
                "is_final":        is_final,
                "inference_ms":    round(inf_ms),
                "session_id":      session_id,
                "pid":             pid,
                "active_sessions": _active_sessions,
                "proc_ram_mb":     round(_get_proc_ram_mb(), 1),
            })
        except Exception:
            pass

    # ── Partial inference (rolling read-only window) ───────────────────────────
    # Reads last PARTIAL_WINDOW_S of accumulated audio — does NOT modify full_audio_bytes
    async def run_partial_inference():
        nonlocal last_confirmed_s
        if len(full_audio_bytes) < BYTES_PER_SAMPLE * SAMPLE_RATE:
            return  # < 1s of audio, skip

        total_dur_s   = len(full_audio_bytes) / (BYTES_PER_SAMPLE * SAMPLE_RATE)
        window_start_s = max(0.0, total_dur_s - PARTIAL_WINDOW_S)
        window_start_b = int(window_start_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
        # align to sample boundary
        window_start_b = (window_start_b // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE

        window_bytes  = bytes(full_audio_bytes[window_start_b:])
        audio_f32     = audio_bytes_to_float32(bytearray(window_bytes))
        window_dur_s  = len(audio_f32) / SAMPLE_RATE

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: _transcriber.transcribe_fast(audio_f32, beam_size=1)
            )
        except Exception as exc:
            logger.debug(f"[{session_id}] Partial inference error: {exc}")
            return

        if not result or not result.words:
            return

        # Only emit words that are NEW (beyond last_confirmed_s in absolute time)
        abs_words = [
            w for w in result.words
            if (window_start_s + w.start) >= last_confirmed_s - DEDUP_MARGIN_S
        ]
        if not abs_words:
            return

        partial_text = fix_punctuation(
            "".join(w.word for w in abs_words).strip(),
            is_final=False
        )
        await send_partial(partial_text, total_dur_s)

    # ── Final full-buffer inference ────────────────────────────────────────────
    async def run_final_inference():
        nonlocal last_confirmed_s, session_texts
        if len(full_audio_bytes) == 0:
            return

        total_dur_s = len(full_audio_bytes) / (BYTES_PER_SAMPLE * SAMPLE_RATE)
        audio_f32   = audio_bytes_to_float32(full_audio_bytes)

        logger.info(
            f"[{session_id}] Final inference: {total_dur_s:.1f}s audio, "
            f"{len(full_audio_bytes)/1024:.0f}KB"
        )
        ts_sent = time.time() * 1000
        loop    = asyncio.get_running_loop()

        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: _transcriber.transcribe_final(audio_f32, beam_size=3)
            )
        except Exception as exc:
            logger.error(f"[{session_id}] Final inference error: {exc}\n{traceback.format_exc()}")
            return

        if not result or not result.text.strip():
            logger.warning(f"[{session_id}] Final inference returned no text (dur={total_dur_s:.1f}s)")
            return

        uid  = str(uuid.uuid4())
        text = fix_punctuation(result.text.strip(), is_final=True)

        session_texts.append(text)
        last_confirmed_s = total_dur_s

        await send_transcript(
            text       = text,
            uid        = uid,
            chunk_idx  = chunks_received,
            inf_ms     = result.inference_ms,
            is_final   = True,
        )
        append_telemetry(uid, "final", ts_sent, result.inference_ms, total_dur_s, text)

        logger.info(
            f"[{session_id}] Final transcript ({len(text)} chars, "
            f"inf={result.inference_ms:.0f}ms): {text[:80]}..."
        )

    # ── Partial scheduler ─────────────────────────────────────────────────────
    partial_stop_event = asyncio.Event()

    async def partial_scheduler():
        """Fires partial inference every PARTIAL_INTERVAL_S while session is live."""
        while not partial_stop_event.is_set():
            await asyncio.sleep(PARTIAL_INTERVAL_S)
            if partial_stop_event.is_set():
                break
            try:
                await run_partial_inference()
            except Exception as exc:
                logger.debug(f"[{session_id}] Partial scheduler error: {exc}")

    partial_task = asyncio.create_task(partial_scheduler())

    # ── Notify client of successful connection ────────────────────────────────
    vm = psutil.virtual_memory()
    try:
        await websocket.send_json({
            "type":             "connected",
            "session_id":       session_id,
            "pid":              pid,
            "active_sessions":  _active_sessions,
            "sys_ram_pct":      vm.percent,
            "sys_free_mb":      round(vm.available / 1e6, 0),
            "proc_ram_mb":      round(_get_proc_ram_mb(), 1),
        })
    except Exception:
        pass

    vm = psutil.virtual_memory()
    logger.info(
        f"[{session_id}] Connected. "
        f"active={_active_sessions}  "
        f"proc_ram={_get_proc_ram_mb():.0f}MB  "
        f"sys_free={vm.available/1e6:.0f}MB  sys_ram={vm.percent:.1f}%"
    )

    # ── Main receive loop ─────────────────────────────────────────────────────
    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg:
                raw = msg["bytes"]
                if not raw:
                    continue

                # Accumulate ALL audio — never trim, never discard
                full_audio_bytes.extend(raw)
                chunks_received += 1

            elif "text" in msg:
                try:
                    data  = json.loads(msg["text"])
                    mtype = data.get("type", "")
                except Exception:
                    continue

                if mtype == "stop":
                    logger.info(f"[{session_id}] STOP received after {chunks_received} chunks.")
                    break

                elif mtype == "ping":
                    try:
                        await websocket.send_json({"type": "pong"})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected after {chunks_received} chunks.")
    except Exception as exc:
        logger.error(f"[{session_id}] Receive loop error: {exc}\n{traceback.format_exc()}")

    finally:
        # ── Shutdown partial scheduler ────────────────────────────────────────
        partial_stop_event.set()
        if not partial_task.done():
            partial_task.cancel()
            try:
                await partial_task
            except asyncio.CancelledError:
                pass

        # ── Run final full-buffer inference ───────────────────────────────────
        try:
            await run_final_inference()
        except Exception as exc:
            logger.error(f"[{session_id}] Final inference failed: {exc}")

        # ── Write full transcript file ────────────────────────────────────────
        if session_texts:
            full_text = " ".join(session_texts)
            try:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                logger.info(f"[{session_id}] Transcript saved: {transcript_path}")
            except Exception:
                pass

        # ── Release session slot ──────────────────────────────────────────────
        with _active_lock:
            if _active_sessions > 0:
                _active_sessions -= 1

        session_dur = time.time() - session_start
        logger.info(
            f"[{session_id}] Session closed. "
            f"dur={session_dur:.1f}s  chunks={chunks_received}  "
            f"audio={len(full_audio_bytes)/1024:.0f}KB  "
            f"active={_active_sessions}/{MAX_SESSIONS_PER_WORKER}  "
            f"proc_ram={_get_proc_ram_mb():.0f}MB"
        )
