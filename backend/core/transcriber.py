"""
NHS Real-Time Transcription — Faster-Whisper Engine
====================================================
Two inference modes:
  transcribe_fast()  — beam_size=1, for rolling partial windows
  transcribe_final() — beam_size=3, for complete final transcription

Model is loaded once per worker process and is thread-safe via CTranslate2.
cpu_threads=2, num_workers=1 → balanced with external ThreadPoolExecutor(2).
"""

import os
import time
import logging
import numpy as np

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from faster_whisper import WhisperModel
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text:          str
    utterance_id:  str
    language:      str
    duration_ms:   float
    inference_ms:  float
    is_final:      bool
    no_speech_prob: float = 0.0
    avg_logprob:   float = 0.0
    words:         list  = field(default_factory=list)


# Whisper hallucinations on silence — filtered post-inference
_SUPPRESS_PHRASES = {
    "thank you.", "thank you for watching.", "thanks.",
    "subscribe.", "like and subscribe.", "you", "you.", "bye.", "bye bye.",
    "i'm sorry.", "please.", "okay.", "...", "mm-hmm.", "mm-hmm",
    "yeah.", "hmm.", "hmm", "oh.", "ah.", "uh.",
}


class WhisperTranscriber:
    """
    Thread-safe Whisper transcriber.
    One instance per uvicorn worker process.
    """

    def __init__(
        self,
        model_name:          str   = "small.en",
        device:              Optional[str] = None,
        no_speech_threshold: float = 0.55,
        logprob_threshold:   float = -0.7,
        cpu_threads:         int   = 2,
        num_workers:         int   = 1,
    ):
        if device:
            self.device = device
        elif _has_torch and __import__("torch").cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.compute_type        = "float16" if self.device == "cuda" else "int8"
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold   = logprob_threshold

        logger.info(
            f"Loading Whisper '{model_name}' on {self.device}/{self.compute_type} "
            f"cpu_threads={cpu_threads} num_workers={num_workers}"
        )
        t0 = time.time()
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Whisper '{model_name}' loaded in {elapsed:.0f}ms")

    # ── Internal shared inference logic ───────────────────────────────────────
    def _run(
        self,
        audio: np.ndarray,
        beam_size: int,
        is_final: bool,
        utterance_id: str = "",
    ) -> Optional[TranscriptResult]:

        if len(audio) < 1600:   # < 0.1s
            return None

        energy = float(np.mean(audio ** 2))
        if energy < 0.00005:
            logger.debug(f"[{utterance_id}] Dropped (energy={energy:.6f})")
            return None

        t0 = time.time()
        try:
            seg_gen, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt="Doctor, patient, clinical consultation.",
                word_timestamps=True,
                no_speech_threshold=self.no_speech_threshold,
                log_prob_threshold=self.logprob_threshold,
            )
            segments = list(seg_gen)
        except Exception as exc:
            logger.error(f"[{utterance_id}] Whisper error: {exc}")
            return None

        inference_ms = (time.time() - t0) * 1000
        duration_ms  = len(audio) / 16000 * 1000

        if not segments:
            return None

        # Aggregate across segments
        words         = []
        no_speech_sum = 0.0
        logprob_sum   = 0.0
        for seg in segments:
            if getattr(seg, "words", None):
                words.extend(seg.words)
            no_speech_sum += getattr(seg, "no_speech_prob", 0.0)
            logprob_sum   += getattr(seg, "avg_logprob", 0.0)

        n = len(segments)
        no_speech_prob = no_speech_sum / n
        avg_logprob    = logprob_sum   / n

        # Silence / low-confidence rejection
        if no_speech_prob > self.no_speech_threshold:
            logger.debug(f"[{utterance_id}] Silence (no_speech={no_speech_prob:.2f})")
            return None
        if avg_logprob < self.logprob_threshold and not is_final:
            logger.debug(f"[{utterance_id}] Low-conf (logprob={avg_logprob:.2f})")
            return None

        text = " ".join(s.text for s in segments).strip()

        # Hallucination filter
        if text.lower().strip(".").strip() in _SUPPRESS_PHRASES:
            logger.debug(f"[{utterance_id}] Suppressed hallucination: '{text}'")
            return None
        if len(text.strip()) <= 1:
            return None

        # Compression ratio — hallucination guard for finals
        if is_final:
            for seg in segments:
                if getattr(seg, "compression_ratio", 0) > 2.8:
                    logger.debug(f"[{utterance_id}] Hallucination by compression ratio")
                    return None

        logger.debug(
            f"[{utterance_id}] '{text[:60]}...' "
            f"ns={no_speech_prob:.2f} lp={avg_logprob:.2f} "
            f"inf={inference_ms:.0f}ms"
        )

        return TranscriptResult(
            text=text,
            utterance_id=utterance_id,
            language=info.language,
            duration_ms=duration_ms,
            inference_ms=inference_ms,
            is_final=is_final,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            words=words,
        )

    # ── Public API ─────────────────────────────────────────────────────────────
    def transcribe_fast(
        self,
        audio: np.ndarray,
        beam_size: int = 1,
        utterance_id: str = "",
    ) -> Optional[TranscriptResult]:
        """Fast partial inference: beam_size=1, for rolling windows during streaming."""
        return self._run(audio, beam_size=beam_size, is_final=False, utterance_id=utterance_id)

    def transcribe_final(
        self,
        audio: np.ndarray,
        beam_size: int = 3,
        utterance_id: str = "",
    ) -> Optional[TranscriptResult]:
        """Accurate final inference: beam_size=3, on full accumulated audio."""
        return self._run(audio, beam_size=beam_size, is_final=True, utterance_id=utterance_id)

    def transcribe(
        self,
        audio: np.ndarray,
        utterance_id: str = "",
        is_final: bool = True,
        blocking: bool = True,
    ) -> Optional[TranscriptResult]:
        """Legacy compatibility wrapper."""
        beam = 3 if is_final else 1
        return self._run(audio, beam_size=beam, is_final=is_final, utterance_id=utterance_id)
