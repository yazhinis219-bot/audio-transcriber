import os
import time
import uuid
import asyncio
import json
import numpy as np
import logging
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.transcriber import WhisperTranscriber
from core.punctuator import fix_punctuation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model choice ──────────────────────────────────────────────────────────────
# small.en is ~4× more accurate than base.en with ~2× slower inference.
# On a modern CPU this is ~800-1200ms per utterance — acceptable for
# a consultation transcriber where accuracy matters more than speed.
# If you have a GPU, inference drops to ~200-400ms.
transcriber = WhisperTranscriber(
    model_name="small.en",
    no_speech_threshold=0.7,    # raised: reject more silence/noise as non-speech
    logprob_threshold=-0.6      # raised: reject low-confidence tokens earlier
)

pool = ThreadPoolExecutor(max_workers=2)   # small.en is heavier; 2 workers is enough

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    pool.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/consultation/{session_id}")
async def websocket_transcribe(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.send_json({"type": "connected", "session_id": session_id})

    SAMPLE_RATE        = 16000
    MAX_BUFFER_S       = 30
    # Fire rolling inference less often — small.en needs time, and accuracy
    # benefits from seeing more audio context per pass.
    PROCESS_INTERVAL_S = 0.6
    # Confirm words only when they are well inside the buffer, not at the edge
    # where Whisper is still unstable.
    CONTEXT_MARGIN_S   = 1.2
    MAX_PENDING_WORDS  = 25
    SILENCE_FLUSH_MS   = 600
    # Energy gates
    SILENCE_ENERGY     = 0.003   # chunks below this = silence
    # Minimum mean energy of the WHOLE speech buffer before we bother calling
    # Whisper.  Prevents transcribing near-silent "breaths" between sentences.
    MIN_BUFFER_ENERGY  = 0.004

    audio_buffer        = np.array([], dtype=np.float32)
    session_texts       = []
    chunk_counter       = 0
    last_process_time   = time.time()
    last_confirmed_end  = 0.0
    buffer_offset_s     = 0.0
    process_lock        = asyncio.Lock()
    silence_start: float | None = None

    pending_tail_words: list  = []
    pending_tail_ts:    float = 0.0
    pending_tail_sent:  float = 0.0

    confirmed_words_log: list[str] = []
    DEDUP_WINDOW = 15

    os.makedirs(r"d:\dummy\transcripts", exist_ok=True)
    telemetry_path  = rf"d:\dummy\transcripts\telemetry_{session_id}.tsv"
    transcript_path = rf"d:\dummy\transcripts\transcript_{session_id}.txt"
    try:
        if not os.path.exists(telemetry_path):
            with open(telemetry_path, "w", encoding="utf-8") as f:
                f.write("UtteranceID\tAudioSentTS\tAudioReceivedTS\tAudioProcessedTS\tAudioSentToUITS\tLatencyMS\tText\n")
    except Exception:
        pass

    def fmt_ts(ms_val):
        return datetime.fromtimestamp(ms_val / 1000.0).strftime('%H:%M:%S.%f')[:-3]

    def has_enough_speech(audio: np.ndarray) -> bool:
        """
        Return True only if the audio contains enough energy to be real speech.
        Prevents Whisper from transcribing near-silent buffers (breath, hum, etc.)
        which is a primary cause of hallucinated words.
        """
        if len(audio) == 0:
            return False
        energy = float(np.mean(audio ** 2))
        if energy < MIN_BUFFER_ENERGY:
            logger.info(f"[GATE] buffer energy {energy:.5f} below threshold, skipping Whisper")
            return False
        return True

    def strip_repeated_prefix(new_words: list) -> list:
        """Drop leading words that duplicate the tail of confirmed_words_log."""
        if not confirmed_words_log or not new_words:
            return new_words
        tail      = confirmed_words_log[-DEDUP_WINDOW:]
        new_texts = [w.word.strip().lower() for w in new_words]
        best_cut  = 0
        for length in range(1, min(len(tail), len(new_texts)) + 1):
            if tail[-length:] == new_texts[:length]:
                best_cut = length
        if best_cut:
            logger.info(f"[DEDUP] dropped {best_cut} repeated word(s): {new_texts[:best_cut]}")
        return new_words[best_cut:]

    def clamp_to_audio(words: list, audio_duration_s: float,
                       tolerance_s: float = 0.1) -> list:
        """
        Reject words whose start timestamp is beyond the actual audio length.
        Whisper hallucinates tokens past the end of the buffer — this kills them.
        """
        limit = audio_duration_s + tolerance_s
        kept  = [w for w in words if w.start <= limit]
        dropped = len(words) - len(kept)
        if dropped:
            logger.info(f"[CLAMP] dropped {dropped} word(s) past audio end "
                        f"({audio_duration_s:.2f}s): "
                        f"{[w.word.strip() for w in words[len(kept):]]}")
        return kept

    def log_confirmed(words: list):
        confirmed_words_log.extend(w.word.strip().lower() for w in words)
        if len(confirmed_words_log) > 300:
            del confirmed_words_log[:150]

    async def emit_confirmed(text: str, ts_sent: float, inference_ms: float = 0,
                             words: list | None = None):
        nonlocal chunk_counter
        if not text.strip():
            return
        chunk_counter += 1
        now_ms = time.time() * 1000
        uid    = str(uuid.uuid4())
        payload = {
            "type": "transcript",
            "text": text,
            "utterance_id": uid,
            "chunk_index": chunk_counter,
            "is_final": True,
            "inference_ms": inference_ms,
            "timestamps": {
                "audio_sent_ts":       ts_sent,
                "audio_received_ts":   ts_sent,
                "audio_processed_ts":  now_ms,
                "audio_sent_to_ui_ts": now_ms,
            },
        }
        try:
            await websocket.send_json(payload)
        except Exception:
            pass
        session_texts.append(text)
        if words:
            log_confirmed(words)
        try:
            latency = int(now_ms - ts_sent)
            with open(telemetry_path, "a", encoding="utf-8") as f:
                f.write(f"{uid}\t{fmt_ts(ts_sent)}\t{fmt_ts(ts_sent)}\t"
                        f"{fmt_ts(now_ms)}\t{fmt_ts(now_ms)}\t{latency}\t{text}\n")
        except Exception:
            pass

    async def flush_tail():
        nonlocal pending_tail_words, pending_tail_ts, pending_tail_sent, last_confirmed_end
        if not pending_tail_words:
            return
        new_tail = [
            w for w in pending_tail_words
            if (buffer_offset_s + w.start) >= last_confirmed_end - 0.05
        ]
        new_tail = strip_repeated_prefix(new_tail)
        if new_tail:
            tail_text = "".join(w.word for w in new_tail).strip()
            tail_text = fix_punctuation(tail_text, is_final=True)
            if tail_text.strip():
                await emit_confirmed(tail_text, pending_tail_sent, words=new_tail)
                last_confirmed_end = buffer_offset_s + new_tail[-1].end
        pending_tail_words = []
        pending_tail_ts    = 0.0
        pending_tail_sent  = 0.0

    async def process_buffer(ts_sent: float, force_flush: bool = False):
        nonlocal audio_buffer, last_confirmed_end, buffer_offset_s
        nonlocal pending_tail_words, pending_tail_ts, pending_tail_sent

        if process_lock.locked() and not force_flush:
            return

        async with process_lock:

            if force_flush:
                await flush_tail()

            if len(audio_buffer) < SAMPLE_RATE * 0.3 and not force_flush:
                return

            buf_copy = audio_buffer.copy()

            # ── Energy gate: don't call Whisper on near-silent buffers ────────
            if not has_enough_speech(buf_copy) and not force_flush:
                return

            buf_duration_s = len(buf_copy) / SAMPLE_RATE

            try:
                loop   = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    pool,
                    lambda: transcriber.transcribe(
                        buf_copy,
                        utterance_id=str(uuid.uuid4()),
                        is_final=force_flush,
                        blocking=True,
                    ),
                )
            except Exception as e:
                logger.error(f"transcribe error: {e}")
                return

            if not result or not getattr(result, "words", None) or not result.words:
                if force_flush:
                    audio_buffer    = np.array([], dtype=np.float32)
                    buffer_offset_s += buf_duration_s
                return

            words = result.words

            def abs_end(w):   return buffer_offset_s + w.end
            def abs_start(w): return buffer_offset_s + w.start

            # ── Force-flush path ─────────────────────────────────────────────
            if force_flush:
                words     = clamp_to_audio(words, buf_duration_s)
                new_words = [w for w in words if abs_start(w) >= last_confirmed_end - 0.05]
                new_words = strip_repeated_prefix(new_words)
                if new_words:
                    full_text = "".join(w.word for w in new_words).strip()
                    full_text = fix_punctuation(full_text, is_final=True)
                    if full_text.strip():
                        await emit_confirmed(full_text, ts_sent, result.inference_ms,
                                             words=new_words)
                        last_confirmed_end = abs_end(new_words[-1])

                keep_s       = 0.5
                keep_samples = int(keep_s * SAMPLE_RATE)
                if len(audio_buffer) > keep_samples:
                    discarded_s      = (len(audio_buffer) - keep_samples) / SAMPLE_RATE
                    audio_buffer     = audio_buffer[-keep_samples:]
                    buffer_offset_s += discarded_s
                else:
                    audio_buffer    = np.array([], dtype=np.float32)
                    buffer_offset_s += buf_duration_s
                return

            # ── Rolling path ─────────────────────────────────────────────────
            words = clamp_to_audio(words, buf_duration_s)

            safe_boundary = buffer_offset_s + buf_duration_s - CONTEXT_MARGIN_S

            confirm_up_to = -1
            for i, w in enumerate(words):
                if abs_end(w) <= safe_boundary:
                    confirm_up_to = i
                else:
                    break

            if len(words) > MAX_PENDING_WORDS and confirm_up_to < MAX_PENDING_WORDS - 1:
                confirm_up_to = MAX_PENDING_WORDS - 1

            if confirm_up_to >= 0:
                confirmed_words = words[: confirm_up_to + 1]
                new_words = [
                    w for w in confirmed_words
                    if abs_start(w) >= last_confirmed_end - 0.05
                ]
                new_words = strip_repeated_prefix(new_words)
                if new_words:
                    confirmed_text = "".join(w.word for w in new_words).strip()
                    confirmed_text = fix_punctuation(confirmed_text, is_final=False)
                    if confirmed_text.strip():
                        await emit_confirmed(confirmed_text, ts_sent, result.inference_ms,
                                             words=new_words)
                        last_confirmed_end = abs_end(new_words[-1])

                # No buffer trimming — preserves acoustic context for next pass
                tail_words = words[confirm_up_to + 1:]
                if tail_words:
                    tail_text = "".join(w.word for w in tail_words).strip()
                    tail_text = fix_punctuation(tail_text, is_final=False)
                    try:
                        await websocket.send_json({"type": "partial", "text": tail_text})
                    except Exception:
                        pass
                    if not pending_tail_words:
                        pending_tail_ts   = time.time()
                        pending_tail_sent = ts_sent
                    pending_tail_words = tail_words
                else:
                    pending_tail_words = []
                    pending_tail_ts    = 0.0

            else:
                partial_text = "".join(w.word for w in words).strip()
                partial_text = fix_punctuation(partial_text, is_final=False)
                try:
                    await websocket.send_json({"type": "partial", "text": partial_text})
                except Exception:
                    pass
                if words:
                    if not pending_tail_words:
                        pending_tail_ts   = time.time()
                        pending_tail_sent = ts_sent
                    pending_tail_words = words

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg:
                raw_bytes = msg["bytes"]
                ts_sent   = time.time() * 1000

                pcm_i16   = np.frombuffer(raw_bytes, dtype=np.int16)
                pcm_float = pcm_i16.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, pcm_float])

                max_samples = SAMPLE_RATE * MAX_BUFFER_S
                if len(audio_buffer) > max_samples:
                    overflow         = len(audio_buffer) - max_samples
                    buffer_offset_s += overflow / SAMPLE_RATE
                    audio_buffer     = audio_buffer[-max_samples:]

                chunk_energy = float(np.mean(pcm_float ** 2))

                if chunk_energy < SILENCE_ENERGY:
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) * 1000 >= SILENCE_FLUSH_MS:
                        if pending_tail_words and not process_lock.locked():
                            await flush_tail()
                        await process_buffer(ts_sent, force_flush=True)
                        silence_start = None
                        continue
                else:
                    silence_start = None

                now = time.time()
                if (now - last_process_time) >= PROCESS_INTERVAL_S:
                    last_process_time = now
                    if not process_lock.locked():
                        asyncio.create_task(process_buffer(ts_sent))

            elif "text" in msg:
                try:
                    data     = json.loads(msg["text"])
                    msg_type = data.get("type")
                    if msg_type == "stop":
                        await flush_tail()
                        await process_buffer(time.time() * 1000, force_flush=True)
                        break
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {traceback.format_exc()}")
    finally:
        if pending_tail_words:
            try:
                await flush_tail()
            except Exception:
                pass
        if session_texts:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(" ".join(session_texts))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)