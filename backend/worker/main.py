"""
backend/worker/main.py  —  MULTI-MODEL POOL v6 (SCALABLE)
===========================================================

Upgraded from single-model to a pool of 3 independent WhisperModel instances.
Each model gets a fixed slice of CPU threads (2 threads each) so that all three
can run concurrently without starving each other.

Key changes vs v5:
- NUM_WORKERS = 3  →  3 real model instances (was 1)
- cpu_threads = 2  →  each model owns 2 threads (was all threads to 1 model)
- model_pool  →  list of 3 WhisperModel objects (was list of 1)

Everything else (model size, device, compute type, warmup) is unchanged.
"""

import os
import sys
import numpy as np
from faster_whisper import WhisperModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_settings

settings = get_settings()

SR          = settings.AUDIO_SAMPLE_RATE

# ── CHANGE 1: Set number of parallel workers / models ────────────────────────
NUM_WORKERS = 4   # 3 independent model instances → 3 concurrent transcriptions

# ── CHANGE 2: Limit CPU threads per model ────────────────────────────────────
# Before: one model consumed ALL cpu threads → no room to scale
# After:  each model gets 2 threads → 3 models can run in parallel
CPU_THREADS_PER_MODEL = 2

print(
    f"[worker] Loading {NUM_WORKERS} × faster-whisper '{settings.WHISPER_MODEL_SIZE}' "
    f"on {settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE} "
    f"with {CPU_THREADS_PER_MODEL} CPU threads each ..."
)

# ── CHANGE 3: Build model pool with NUM_WORKERS independent instances ─────────
# Before: model_pool = [_model]   (single model)
# After:  model_pool = [model1, model2, model3]
model_pool: list[WhisperModel] = []
for i in range(NUM_WORKERS):
    m = WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        cpu_threads=CPU_THREADS_PER_MODEL,   # ← limited per model (was all threads)
        num_workers=1,
    )
    model_pool.append(m)
    print(f"[worker]   model[{i}] loaded")

print(f"[worker] {NUM_WORKERS} models loaded — ready for parallel transcription")

# ── Warmup all models ─────────────────────────────────────────────────────────
# Identical warmup call as before; applied to each model in the pool.
print("[worker] Warming up all models ...")
_noise = (np.random.randn(SR * 2).astype(np.float32) * 0.05)
for i, m in enumerate(model_pool):
    list(m.transcribe(
        _noise,
        language="en",
        beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=False,
        word_timestamps=False,
        temperature=0.0,
    ))
    print(f"[worker]   model[{i}] warmed up")

print("[worker] all models warmed up — ready for parallel transcription cycles.")
