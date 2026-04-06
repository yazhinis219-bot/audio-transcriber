import asyncio
import threading
import time
import re
import numpy as np
from collections import deque
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import get_settings
settings = get_settings()

app = FastAPI(title=settings.APP_NAME + " (Gateway - Enhanced No-Overlap Chunking)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict = {}
session_tasks:   dict = {}

from backend.worker.main import model_pool, NUM_WORKERS  # noqa: E402

SR                = settings.AUDIO_SAMPLE_RATE
MIN_START_SAMPLES = int(settings.MIN_CHUNK_TO_START_SEC * SR)

# ── CHANGE 1: REMOVE GLOBAL LOCK ─────────────────────────────────────────────
# Before:
#   _MODEL      = model_pool[0]
#   _MODEL_LOCK = threading.Lock()
# After: no lock needed — each model instance is independent

# ── CHANGE 2: ROUND-ROBIN MODEL SELECTION ────────────────────────────────────
# Distributes requests evenly across all model instances.
# Request 1 → model_pool[0], Request 2 → model_pool[1], ...
_model_index      = 0
_model_index_lock = threading.Lock()   # protects only the counter, NOT the model

def _get_next_model():
    """Return the next model in round-robin order (thread-safe counter)."""
    global _model_index
    with _model_index_lock:
        model = model_pool[_model_index % len(model_pool)]
        _model_index += 1
    return model

# ── Timing constants ─────────────────────────────────────────────────────────
SILENCE_TAIL_SEC     = 0.8   # safe silence gap to commit sentences (lowered for faster UI reflection)
MAX_AUDIO_SEC        = 25.0  # max buffer size before forced commit (keeps chunk boundaries safe from VAD drops)
TRANSCRIBE_EVERY_SEC = 0.5   # frequent transcription for live partials
MIN_NEW_SAMPLES      = int(SR * 0.3)

# ── Hallucination patterns (unchanged) ─────────────────────────────────────────
_HALLUCINATION_RE = re.compile(
    r"^\s*("
    r"subscribe[\s\w]*|"
    r"like and (subscribe|share)[\s\w]*|"
    r"thanks for watching[\s,!.]*|"
    r"\[.*?\]|"
    r"\(.*?\)"
    r")\s*$",
    re.IGNORECASE,
)
_FILLER_RE = re.compile(r"(\b\w+\b)(\s+\1){3,}", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# AudioBuffer for No-Overlap Chunking
# ─────────────────────────────────────────────────────────────────────────────

class AudioBuffer:
    def __init__(self):
        self._buf = np.empty(0, dtype=np.float32)
        self._total_session_samples = 0
        
    def append(self, pcm: np.ndarray):
        self._buf = np.concatenate([self._buf, pcm])
        self._total_session_samples += pcm.size
        
        # Failsafe limit to prevent absolute memory overflow.
        # Increased to 150s (2.5 minutes) to give slow CPUs a generous backlog buffer without dropping words.
        max_samples = int(150 * SR)
        if len(self._buf) > max_samples:
            self._buf = self._buf[-max_samples:]

    def get_audio(self) -> np.ndarray:
        return self._buf

    def get_duration(self) -> float:
        return len(self._buf) / SR

    def consume(self, seconds: float):
        samples = int(seconds * SR)
        self._buf = self._buf[samples:]

    @property
    def total_session_samples(self) -> int:
        return self._total_session_samples


# ─────────────────────────────────────────────────────────────────────────────
# Transcription helpers
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe(audio: np.ndarray, initial_prompt: str = "") -> list:
    prompt = initial_prompt or settings.MEDICAL_VOCAB_PROMPT

    # ── CHANGE 3: NO GLOBAL LOCK + ROUND-ROBIN MODEL ─────────────────────────
    # Before:
    #   with _MODEL_LOCK:
    #       seg_gen, _ = _MODEL.transcribe(...)
    # After:
    #   model = _get_next_model()   ← picks next model in rotation
    #   seg_gen, _ = model.transcribe(...)   ← no lock; models are independent
    model = _get_next_model()
    seg_gen, _ = model.transcribe(
        audio,
        language=settings.WHISPER_LANGUAGE or None,
        condition_on_previous_text=True,
        initial_prompt=prompt,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,  # lower gap needed to split words
            "speech_pad_ms":           150,  # small pad so VAD accepts tightly cut chunks without dropping them
            "threshold":               0.05, # very sensitive to catch all quiet speech
            "min_speech_duration_ms":  50,   # catch fast tiny words like "ok" or "yes"
        },
        beam_size=1,  # Greedy decoding (massively faster on CPU)
        no_speech_threshold=0.45,
        compression_ratio_threshold=2.4,
        temperature=0.0,
        patience=1.0,
        log_prob_threshold=-1.0,
    )
    return list(seg_gen)

def _build_prompt(emitted_text: str) -> str:
    tail = emitted_text[-200:].strip() if emitted_text else ""
    return (tail + "\n" + settings.MEDICAL_VOCAB_PROMPT) if tail \
        else settings.MEDICAL_VOCAB_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _strip_hallucinations(text: str) -> str:
    if not text:
        return text
    text = _FILLER_RE.sub(r"\1", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    filtered, seen = [], set()
    for s in sentences:
        if _HALLUCINATION_RE.fullmatch(s.strip()):
            continue
        key = re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        if key and key in seen:
            continue
        seen.add(key)
        filtered.append(s)
    while len(filtered) >= 2:
        last = re.sub(r"[^a-z0-9 ]", "", filtered[-1].lower()).strip()
        prev = re.sub(r"[^a-z0-9 ]", "", filtered[-2].lower()).strip()
        if last and last == prev:
            filtered.pop()
        else:
            break
    return " ".join(filtered).strip()

def _is_garbage(text: str) -> bool:
    if not text or not text.strip():
        return True
    s = text.strip()
    if len(s) <= 1:
        return True
    if re.fullmatch(r"[^a-zA-Z0-9]+", s):
        return True
    if _HALLUCINATION_RE.fullmatch(s):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Background worker pool
# ─────────────────────────────────────────────────────────────────────────────

# ── CHANGE 4: REPLACE SINGLE SEQUENTIAL WORKER WITH A POOL ───────────────────
# Before:
#   class _SequentialWorker:          (1 thread, 1 queue)
#   _worker = _SequentialWorker()
#
# After:
#   class _WorkerPool:                (NUM_WORKERS threads, shared queue)
#   _worker_pool = _WorkerPool(NUM_WORKERS)
#
# Each thread in the pool picks jobs from the shared queue independently,
# so NUM_WORKERS jobs can execute simultaneously.

import queue as _q_mod

class _WorkerPool:
    """
    Thread pool backed by a single shared queue.
    NUM_WORKERS threads drain the queue concurrently so that multiple
    transcription jobs run in parallel instead of one-at-a-time.
    """
    def __init__(self, num_workers: int):
        self._q = _q_mod.Queue()
        for i in range(num_workers):
            t = threading.Thread(
                target=self._loop,
                daemon=True,
                name=f"WhisperWorker-{i}",
            )
            t.start()
        print(f"[gateway] worker pool started: {num_workers} threads")

    def submit(self, fn, *args):
        self._q.put((fn, args))

    def _loop(self):
        while True:
            fn, args = self._q.get()
            try:
                fn(*args)
            except Exception:
                import traceback
                traceback.print_exc()

_worker = _WorkerPool(NUM_WORKERS)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket & HTTP endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_sessions[session_id] = {
        "websocket":        websocket,
        "transcript_queue": asyncio.Queue(),
        "full_text":        [],
    }
    proc_task = asyncio.create_task(_accumulate_and_dispatch(session_id))
    session_tasks[session_id] = proc_task
    consumer  = asyncio.create_task(_send_transcripts(websocket, session_id))
    try:
        await asyncio.gather(proc_task, consumer)
    except WebSocketDisconnect:
        print(f"[{session_id}] client disconnected")
    finally:
        await _cleanup_session(session_id)

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "active_sessions": len(active_sessions),
        "num_workers": NUM_WORKERS,
    })

frontend_dir = Path(__file__).resolve().parents[2] / "public"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Transcript sender
# ─────────────────────────────────────────────────────────────────────────────

async def _send_transcripts(websocket: WebSocket, session_id: str):
    transcript_dir = Path(__file__).resolve().parents[2] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    try:
        tq = active_sessions[session_id]["transcript_queue"]
        while True:
            data = await tq.get()
            if data.get("type") == "final":
                active_sessions[session_id]["full_text"].append(data.get("text", ""))
            if data.get("type") == "end":
                final_text = active_sessions[session_id].get("final_transcript", "")
                if not final_text:
                    final_text = " ".join(active_sessions[session_id]["full_text"]).strip()
                path = transcript_dir / "results" / f"{session_id}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(final_text + "\n", encoding="utf-8")
                data["transcript_file"] = str(path)
            await websocket.send_json(data)
            if data.get("type") == "end":
                break
    except Exception as exc:
        print(f"[{session_id}] send_transcripts: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline — Strict Chunking (No Overlap Deduplication)
# ─────────────────────────────────────────────────────────────────────────────

class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.log_path = Path(__file__).resolve().parents[2] / "transcripts" / "debug" / f"{session_id}_debug.tsv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("T+ Audio Arrived (s)\tT+ Sent Worker (s)\tT+ Worker Done (s)\tT+ Results UI (s)\tAudio Len (s)\tProcess Time (s)\tType\tText / Event\n")

    def log_event(self, t_sent: float, t_done: float, audio_len: float, process_time: float, event_type: str, text: str):
        t_arrived = max(0.0, t_sent - audio_len)
        t_emitted = time.time() - self.start_time
        with open(self.log_path, "a", encoding="utf-8") as f:
            safe_text = str(text).replace("\n", " ").replace("\t", " ")
            f.write(f"{t_arrived:.2f}\t{t_sent:.2f}\t{t_done:.2f}\t{t_emitted:.2f}\t{audio_len:.2f}\t{process_time:.2f}\t{event_type}\t{safe_text}\n")

    def log_error(self, err_msg: str):
        elapsed = time.time() - self.start_time
        with open(self.log_path, "a", encoding="utf-8") as f:
            safe_err = str(err_msg).replace("\n", " ").replace("\t", " ")
            f.write(f"{elapsed:.2f}s\t-\t-\t-\tCRITICAL ERROR\t{safe_err}\n")


@dataclass
class _Word:
    word: str
    start: float
    end: float

async def _accumulate_and_dispatch(session_id: str):
    """
    ARCHITECTURE CHANGE: Split into two cooperating coroutines.

    OLD (broken under load):
        One coroutine doing BOTH WebSocket reading AND dispatch decisions
        using ws.receive(timeout=0.05) busy-poll.
        Problem: with 46 sessions x 20 polls/sec = 920 event-loop wake-ups/sec.
        The event loop falls behind; the 50ms window misses arriving WebSocket
        frames; the session sees a spurious disconnect -> closes early -> client
        times out with a partial or empty transcript.

    NEW (fixed):
        _ws_reader  -- pure blocking await ws.receive(); never times out.
                       Pushes (pcm_bytes | "END") into an asyncio.Queue.
        _dispatcher -- wakes only on a TRANSCRIBE_EVERY_SEC ticker OR when
                       _ws_reader enqueues new data. Makes all dispatch
                       decisions from the queue.
        The two tasks run concurrently inside asyncio.gather().
        The event loop is free between real I/O events -> no missed frames.
    """
    tq   = active_sessions[session_id]["transcript_queue"]
    ws   = active_sessions[session_id]["websocket"]
    loop = asyncio.get_event_loop()

    logger = SessionLogger(session_id)
    audio  = AudioBuffer()
    emitted_transcript: str = ""

    # Signal queue: _ws_reader -> _dispatcher
    # Each item is either raw bytes (PCM) or the string sentinel "END".
    _incoming: asyncio.Queue = asyncio.Queue()

    worker_busy  = False
    result_event = asyncio.Event()

    last_transcribe_time          = 0.0
    samples_since_last_transcribe = 0

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _commit_final(text: str):
        nonlocal emitted_transcript
        text = _strip_hallucinations(_clean(text))
        if _is_garbage(text):
            return
        emitted_transcript = (emitted_transcript + " " + text).strip()
        loop.call_soon_threadsafe(
            lambda t=text: tq.put_nowait({"type": "final", "text": t})
        )
        print(f"[{session_id}] FINAL: '{text[:120]}'")

    # -------------------------------------------------------------------------
    # Thread callbacks (all logic identical to original)
    # -------------------------------------------------------------------------

    def _on_rolling_result(window_audio: np.ndarray, prompt: str, t_sent: float):
        nonlocal worker_busy
        try:
            t0 = time.monotonic()
            segments = _transcribe(window_audio, prompt)
            process_time = time.monotonic() - t0
            t_done = time.time() - logger.start_time

            words = []
            for seg in segments:
                seg_words = getattr(seg, "words", None)
                if seg_words:
                    for w in seg_words:
                        words.append(_Word(w.word, w.start, w.end))
                else:
                    words.append(_Word(seg.text, seg.start, seg.end))

            buffer_duration = len(window_audio) / SR

            if not words:
                if buffer_duration > 1.0:
                    cut_time = buffer_duration - 0.3
                    loop.call_soon_threadsafe(lambda c=cut_time: audio.consume(c))
                loop.call_soon_threadsafe(
                    lambda: tq.put_nowait({"type": "partial", "text": ""})
                )
                return

            last_end      = words[-1].end
            silence_after = buffer_duration - last_end

            # CONDITION 1: Natural silence gap
            if silence_after >= SILENCE_TAIL_SEC:
                commit_text = "".join(w.word for w in words)
                _commit_final(commit_text)
                logger.log_event(t_sent, t_done, buffer_duration, process_time, "FINAL (Sil-Gap)", commit_text)
                keep_silence = min(silence_after / 2.0, 0.4)
                cut_time     = min(last_end + keep_silence, buffer_duration)
                loop.call_soon_threadsafe(lambda c=cut_time: audio.consume(c))
                loop.call_soon_threadsafe(
                    lambda: tq.put_nowait({"type": "partial", "text": ""})
                )
                return

            # CONDITION 2: Buffer too large - force commit
            if buffer_duration >= MAX_AUDIO_SEC:
                max_gap      = 0.0
                best_cut_idx = -1
                for i in range(len(words) - 2, -1, -1):
                    gap = words[i+1].start - words[i].end
                    if gap >= 0.3:
                        max_gap      = gap
                        best_cut_idx = i
                        break

                if best_cut_idx >= 0 and max_gap >= 0.3:
                    commit_words = words[:best_cut_idx+1]
                    commit_text  = "".join(w.word for w in commit_words)
                    _commit_final(commit_text)
                    logger.log_event(t_sent, t_done, buffer_duration, process_time, "FINAL (Max-Gap)", commit_text)
                    cut_time = min(words[best_cut_idx].end + (max_gap / 2.0), buffer_duration)
                    loop.call_soon_threadsafe(lambda c=cut_time: audio.consume(c))
                    partial_text = "".join(w.word for w in words[best_cut_idx+1:])
                    partial_text = _strip_hallucinations(_clean(partial_text))
                    if not _is_garbage(partial_text):
                        logger.log_event(t_sent, t_done, buffer_duration, process_time, "PARTIAL (split)", partial_text)
                        loop.call_soon_threadsafe(
                            lambda t=partial_text: tq.put_nowait({"type": "partial", "text": t})
                        )
                    return
                else:
                    half        = max(1, len(words) // 2)
                    commit_text = "".join(w.word for w in words[:half])
                    _commit_final(commit_text)
                    logger.log_event(t_sent, t_done, buffer_duration, process_time, "FINAL (Half-Cut)", commit_text)
                    if half < len(words):
                        cut_time = (words[half-1].end + words[half].start) / 2.0
                    else:
                        cut_time = words[-1].end + 0.1
                    cut_time = min(cut_time, buffer_duration)
                    loop.call_soon_threadsafe(lambda c=cut_time: audio.consume(c))
                    partial_text = "".join(w.word for w in words[half:])
                    partial_text = _strip_hallucinations(_clean(partial_text))
                    if not _is_garbage(partial_text):
                        loop.call_soon_threadsafe(
                            lambda t=partial_text: tq.put_nowait({"type": "partial", "text": t})
                        )
                    return

            # CONDITION 3: Stream partial
            partial_text = "".join(w.word for w in words)
            partial_text = _strip_hallucinations(_clean(partial_text))
            if not _is_garbage(partial_text):
                logger.log_event(t_sent, t_done, buffer_duration, process_time, "PARTIAL", partial_text)
                loop.call_soon_threadsafe(
                    lambda t=partial_text: tq.put_nowait({"type": "partial", "text": t})
                )

        except Exception as exc:
            logger.log_error(f"Rolling Result Crash: {exc}")
            import traceback; traceback.print_exc()
        finally:
            worker_busy = False
            loop.call_soon_threadsafe(result_event.set)

    def _on_tail_result(tail_audio: np.ndarray, prompt: str, t_sent: float):
        nonlocal worker_busy
        try:
            if len(tail_audio) == 0:
                return
            t0          = time.monotonic()
            padded_tail = np.pad(tail_audio, (0, SR))
            segments    = _transcribe(padded_tail, prompt)
            process_dur = time.monotonic() - t0
            t_done      = time.time() - logger.start_time
            words = []
            for seg in segments:
                seg_words = getattr(seg, "words", None)
                if seg_words:
                    for w in seg_words:
                        words.append(_Word(w.word, w.start, w.end))
                else:
                    words.append(_Word(seg.text, seg.start, seg.end))
            if words:
                commit_text = "".join(w.word for w in words)
                _commit_final(commit_text)
                logger.log_event(t_sent, t_done, len(tail_audio)/SR, process_dur, "FINAL (Tail-Pass)", commit_text)
        except Exception as exc:
            logger.log_error(f"Tail Result Crash: {exc}")
            import traceback; traceback.print_exc()
        finally:
            worker_busy = False
            loop.call_soon_threadsafe(result_event.set)

    # -------------------------------------------------------------------------
    # COROUTINE 1: pure WebSocket reader
    # Never uses a timeout -- just awaits the next frame and enqueues it.
    # The event loop only wakes this coroutine when real data arrives,
    # not on a 50ms tick. This eliminates the event-loop overload that
    # caused spurious disconnects at 46+ concurrent sessions.
    # -------------------------------------------------------------------------

    async def _ws_reader():
        try:
            while True:
                msg = await ws.receive()
                if msg.get("bytes"):
                    await _incoming.put(msg["bytes"])
                elif msg.get("text") == "END":
                    await _incoming.put("END")
                    return
        except WebSocketDisconnect:
            await _incoming.put("END")
        except Exception as exc:
            logger.log_error(f"ws_reader crash: {exc}")
            await _incoming.put("END")

    # -------------------------------------------------------------------------
    # COROUTINE 2: dispatcher
    # Drains _incoming to keep the audio buffer full, then on a
    # TRANSCRIBE_EVERY_SEC cadence decides whether to submit a job.
    # Uses asyncio.wait_for with a short timeout purely as a ticker --
    # it does NOT affect whether audio bytes are received (_ws_reader handles that).
    # -------------------------------------------------------------------------

    async def _dispatcher():
        nonlocal worker_busy, last_transcribe_time, samples_since_last_transcribe
        is_closed = False

        while not is_closed:
            # Drain everything currently in the queue without blocking long.
            while True:
                try:
                    item = _incoming.get_nowait()
                    if item == "END":
                        is_closed = True
                        break
                    pcm = (
                        np.frombuffer(item, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    audio.append(pcm)
                    samples_since_last_transcribe += pcm.size
                except asyncio.QueueEmpty:
                    break

            # If nothing was queued, wait up to TRANSCRIBE_EVERY_SEC for new
            # data before looping again (acts as a dispatch ticker).
            if not is_closed:
                try:
                    item = await asyncio.wait_for(
                        _incoming.get(), timeout=TRANSCRIBE_EVERY_SEC
                    )
                    if item == "END":
                        is_closed = True
                    else:
                        pcm = (
                            np.frombuffer(item, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        audio.append(pcm)
                        samples_since_last_transcribe += pcm.size
                except asyncio.TimeoutError:
                    pass  # ticker fired -- fall through to dispatch check

            if result_event.is_set():
                result_event.clear()

            now          = time.monotonic()
            enough_audio = audio.get_duration() >= (MIN_START_SAMPLES / SR)
            time_ok      = (now - last_transcribe_time) >= TRANSCRIBE_EVERY_SEC
            enough_new   = samples_since_last_transcribe >= MIN_NEW_SAMPLES

            if enough_audio and time_ok and enough_new and not worker_busy and not is_closed:
                window_audio = audio.get_audio()
                if window_audio.size > 0:
                    prompt                        = _build_prompt(emitted_transcript)
                    worker_busy                   = True
                    last_transcribe_time          = now
                    samples_since_last_transcribe = 0
                    result_event.clear()
                    t_sent = time.time() - logger.start_time
                    _worker.submit(_on_rolling_result, window_audio, prompt, t_sent)

        # END received: flush in-flight job, then run tail pass

        if worker_busy:
            print(f"[{session_id}] waiting for in-flight job ...")
            try:
                await asyncio.wait_for(result_event.wait(), timeout=800.0)
            except asyncio.TimeoutError:
                pass
            result_event.clear()

        tail_audio = audio.get_audio()
        if len(tail_audio) > 0:
            prompt      = _build_prompt(emitted_transcript)
            worker_busy = True
            result_event.clear()
            t_sent = time.time() - logger.start_time
            _worker.submit(_on_tail_result, tail_audio, prompt, t_sent)
            try:
                await asyncio.wait_for(result_event.wait(), timeout=800.0)
            except asyncio.TimeoutError:
                pass

        final_text = emitted_transcript.strip()
        if session_id in active_sessions:
            active_sessions[session_id]["final_transcript"] = final_text
        if final_text:
            await tq.put({"type": "final_complete", "text": final_text})
        await tq.put({"type": "end"})
        print(f"[{session_id}] Transcription complete. session={audio.total_session_samples/SR:.1f}s")
        t_end = time.time() - logger.start_time
        logger.log_event(
            t_end, t_end, audio.total_session_samples / SR, 0.0,
            "SESSION END", f"Emitted Words: {len(final_text.split())}"
        )

    # Run both coroutines concurrently
    try:
        await asyncio.gather(_ws_reader(), _dispatcher())
    except Exception as exc:
        logger.log_error(f"accumulate_and_dispatch crash: {exc}")
        print(f"[{session_id}] loop crashed: {exc}")
        import traceback; traceback.print_exc()
    finally:
        if session_id in active_sessions:
            try:
                active_sessions[session_id]["transcript_queue"].put_nowait({"type": "end"})
            except Exception:
                pass


async def _cleanup_session(session_id: str):
    print(f"[{session_id}] cleaning up")
    if session_id in session_tasks:
        session_tasks[session_id].cancel()
        del session_tasks[session_id]
    if session_id in active_sessions:
        del active_sessions[session_id]