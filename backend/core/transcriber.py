"""
NHS Real-Time Transcription - Faster-Whisper Engine
Wraps faster-whisper with NHS-optimised settings.
Model is loaded once and utilizes CTranslate2's native multi-threading.
"""

import time
import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TranscriptResult:
    text: str
    utterance_id: str
    language: str
    duration_ms: float
    inference_ms: float
    is_final: bool
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    words: list = None


# Tokens that Whisper hallucinates during silence/noise — suppress them
NHS_SUPPRESS_TOKENS = [
    # Common Whisper hallucinations on silence
    " Thank you.", " Thanks.", " Thank you for watching.",
    " Subscribe.", " Like and subscribe.",
    " you", " You", " Bye.", " Bye bye.",
    " I'm sorry.", " Please.", " Okay.",
    " ...", "...", " Mm-hmm.", " mm-hmm", " Mm-hmm", " mm-hmm.",
    " Yeah.", " yeah.", " Hmm.", " hmm", " hmm.", " Oh.",
    " Ah.", " ah.", " Uh.", " uh."
]


class WhisperTranscriber:
    """
    Transcribes a single AudioChunk using faster-whisper.

    Key NHS-specific settings:
    - language="en" — no language detection overhead
    - no_speech_threshold: chunks below this prob are silently dropped
    - condition_on_previous_text=False: prevents context bleed between utterances
    """

    def __init__(
        self,
        model_name: str = "base.en",
        device: Optional[str] = None,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -0.85,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold

        logger.info(f"Loading faster-whisper model '{model_name}' on {self.device} with {self.compute_type}...")
        t0 = time.time()
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
            num_workers=4
        )
        elapsed = (time.time() - t0) * 1000
        logger.info(f"faster-whisper model '{model_name}' loaded in {elapsed:.0f}ms")

        # In faster-whisper, suppress_tokens expects a list of IDs.
        # But we can also just use the string text to drop hallucinations manually because faster-whisper's 
        # API for suppress_tokens requires exploring the specific tokenizer IDs of faster-whisper.
        # For safety, we will just filter the text strings post-transcription.
        self._suppress_phrases = [s.strip() for s in NHS_SUPPRESS_TOKENS]

    def transcribe(self, audio_float32: np.ndarray, utterance_id: str = "", is_final: bool = True, blocking: bool = True) -> Optional[TranscriptResult]:
        """
        Transcribe float32 audio at 16kHz.
        Returns None if audio is classified as silence/noise.
        (blocking is ignored for faster-whisper because we use native num_workers queue)
        """
        if len(audio_float32) < 1600:  # < 0.1s — skip
            return None

        # Energy threshold lowered to allow extremely quiet speech (RMS ~ 0.01)
        energy = np.mean(audio_float32**2)
        if energy < 0.0001:
            logger.debug(f"[{utterance_id}] Dropped due to low audio energy: {energy:.5f}")
            return None

        t0 = time.time()

        try:
            segments_gen, info = self.model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt="Doctor, patient, diagnosis.",
                word_timestamps=True
            )
            
            # evaluate generator
            segments = list(segments_gen)
        except Exception as e:
            logger.error(f"Whisper decode error: {e}")
            return None

        inference_ms = (time.time() - t0) * 1000
        duration_ms = len(audio_float32) / 16000 * 1000

        if not segments:
            return None

        words = []
        for seg in segments:
            if getattr(seg, 'words', None):
                words.extend(seg.words)

        # Calculate avg no_speech_prob and avg_logprob across segments
        no_speech_prob = sum([seg.no_speech_prob for seg in segments]) / len(segments)
        avg_logprob = sum([seg.avg_logprob for seg in segments]) / len(segments)

        if no_speech_prob > self.no_speech_threshold:
            logger.debug(
                f"[{utterance_id}] Rejected as silence "
                f"(no_speech_prob={no_speech_prob:.2f})"
            )
            return None

        text = " ".join([seg.text for seg in segments]).strip()

        # Reject very low probability transcripts (noise, garbled)
        if avg_logprob < self.logprob_threshold:
            logger.debug(
                f"[{utterance_id}] Rejected low-confidence "
                f"(avg_logprob={avg_logprob:.2f})"
            )
            return None

        # Reject hallucinated repetitive garbage
        if hasattr(info, 'compression_ratio') and info.compression_ratio > 2.4:
            logger.debug(f"[{utterance_id}] Rejected hallucination (info compression_ratio={info.compression_ratio:.2f})")
            return None
        elif any(getattr(seg, 'compression_ratio', 0) > 2.4 for seg in segments):
            logger.debug(f"[{utterance_id}] Rejected hallucination (segment compression_ratio > 2.4)")
            return None

        # Filter out common hallucinations explicitly
        if any(text == phrase for phrase in self._suppress_phrases) or any(text == phrase + "." for phrase in self._suppress_phrases):
             # faster-whisper can still hallucinate repetitive garbage. 
             logger.debug(f"[{utterance_id}] Rejected hallucinated suppressed token: '{text}'")
             return None

        # Final guard: empty or single-character results are garbage
        if len(text.strip()) <= 1:
            return None

        logger.debug(
            f"[{utterance_id}] '{text}' "
            f"| no_speech={no_speech_prob:.2f} "
            f"| logprob={avg_logprob:.2f} "
            f"| infer={inference_ms:.0f}ms"
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
