"""
NHS Real-Time Transcription - Audio Preprocessor
Handles VAD, silence detection, chunk assembly and pre-emphasis.
"""

import numpy as np
import webrtcvad
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class SpeechState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    END_OF_UTTERANCE = "end_of_utterance"


@dataclass
class AudioChunk:
    """A processed audio chunk ready for transcription."""
    pcm_float: np.ndarray          # float32 [-1, 1], 16kHz mono
    sample_rate: int = 16000
    is_speech: bool = True
    utterance_id: str = ""
    chunk_index: int = 0
    is_final: bool = False         # True = end of utterance (pause detected)
    context_frames: list = field(default_factory=list)  # trailing context for re-transcription


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_duration_ms: int = 30          # webrtcvad supports 10, 20, 30ms
    vad_aggressiveness: int = 2          # 0=least aggressive, 3=most
    speech_pad_ms: int = 300             # pad N ms before/after speech
    silence_threshold_ms: int = 800      # pause this long = end of utterance
    min_speech_ms: int = 200             # discard shorter than this
    max_utterance_ms: int = 30_000       # force-flush after 30s
    context_keep_ms: int = 1500          # keep last N ms for re-transcription overlap
    pre_emphasis_coef: float = 0.97      # high-pass filter coefficient


class AudioPreprocessor:
    """
    Production-grade audio preprocessor for live transcription.

    Pipeline:
      raw PCM bytes (16kHz, 16-bit, mono)
        → pre-emphasis filter
        → frame-level VAD (WebRTC VAD)
        → speech/silence state machine
        → utterance assembly
        → yields AudioChunk when utterance complete

    Key behaviour:
    - Silence is swallowed — the model never sees silent frames
    - A pause >= silence_threshold_ms emits the current utterance as final
    - Last context_keep_ms of audio is prepended to the NEXT utterance
      so Whisper can re-transcribe the boundary accurately
    - Short noise bursts (<min_speech_ms) are rejected
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.cfg = config or VADConfig()
        self._vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)

        self._frame_bytes = int(
            self.cfg.sample_rate * self.cfg.frame_duration_ms / 1000 * 2
        )  # 2 bytes per 16-bit sample

        # State machine
        self._state = SpeechState.SILENCE
        self._speech_frames: list[bytes] = []
        self._silence_ms: int = 0
        self._speech_ms: int = 0
        self._utterance_count: int = 0

        # Context ring buffer — keeps last N ms for overlap re-transcription
        context_frames = int(self.cfg.context_keep_ms / self.cfg.frame_duration_ms)
        self._context_buffer: deque[bytes] = deque(maxlen=context_frames)

        # Padding buffers
        pad_frames = int(self.cfg.speech_pad_ms / self.cfg.frame_duration_ms)
        self._pre_pad: deque[bytes] = deque(maxlen=pad_frames)

        # Incoming byte accumulator (handles arbitrary-sized network packets)
        self._byte_buf = b""

        logger.info(
            "AudioPreprocessor ready",
            extra={
                "frame_ms": self.cfg.frame_duration_ms,
                "vad_level": self.cfg.vad_aggressiveness,
                "silence_threshold_ms": self.cfg.silence_threshold_ms,
                "context_keep_ms": self.cfg.context_keep_ms,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, raw_bytes: bytes) -> list[AudioChunk]:
        """
        Feed arbitrary-length raw PCM bytes (16-bit, 16kHz, mono, little-endian).
        Returns a list of AudioChunks ready for transcription (may be empty).
        """
        self._byte_buf += raw_bytes
        ready_chunks: list[AudioChunk] = []

        while len(self._byte_buf) >= self._frame_bytes:
            frame = self._byte_buf[: self._frame_bytes]
            self._byte_buf = self._byte_buf[self._frame_bytes :]
            chunk = self._process_frame(frame)
            if chunk is not None:
                ready_chunks.append(chunk)

        return ready_chunks

    def flush(self) -> Optional[AudioChunk]:
        """Force-emit whatever speech is buffered (call on stream close)."""
        if self._speech_frames:
            return self._emit_utterance(is_final=True)
        return None

    def reset(self):
        """Reset all state (new session)."""
        self._state = SpeechState.SILENCE
        self._speech_frames.clear()
        self._silence_ms = 0
        self._speech_ms = 0
        self._byte_buf = b""
        self._context_buffer.clear()
        self._pre_pad.clear()

    def commit_audio(self, ms: float):
        """Removes the oldest `ms` milliseconds from the current accumulating buffers."""
        total_frames_to_remove = int(ms / self.cfg.frame_duration_ms)
        
        # Remove from context buffer first
        removed_from_context = 0
        while len(self._context_buffer) > 0 and removed_from_context < total_frames_to_remove:
            self._context_buffer.popleft()
            removed_from_context += 1
            
        remaining = total_frames_to_remove - removed_from_context
        if remaining > 0 and remaining < len(self._speech_frames):
            self._speech_frames = self._speech_frames[remaining:]
            self._speech_ms -= remaining * self.cfg.frame_duration_ms
        elif remaining >= len(self._speech_frames):
            self._speech_frames = []
            self._speech_ms = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_frame(self, frame: bytes) -> Optional[AudioChunk]:
        """Process one VAD frame. Returns AudioChunk if an utterance is complete."""

        # Pre-emphasis on float representation (before VAD which uses raw PCM)
        samples_i16 = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        emphasized = self._pre_emphasis(samples_i16)
        frame_emphasized = (emphasized.clip(-32768, 32767).astype(np.int16)).tobytes()

        # WebRTC VAD decision
        try:
            is_speech = self._vad.is_speech(frame_emphasized, self.cfg.sample_rate)
            # Strict energy gate disabled/reduced to prevent masking quiet speakers
            energy = np.mean(samples_i16**2)
            if energy < 10_000:  # RMS ~100
                is_speech = False
        except Exception:
            is_speech = False

        result = None

        if is_speech:
            self._silence_ms = 0
            self._speech_ms += self.cfg.frame_duration_ms

            if self._state == SpeechState.SILENCE:
                # Transition: silence → speech
                # Prepend pre-roll padding frames for natural sentence start
                self._speech_frames = list(self._pre_pad) + [frame]
                self._state = SpeechState.SPEECH
            else:
                self._speech_frames.append(frame)

            # Guard: force-flush if utterance is too long
            if self._speech_ms >= self.cfg.max_utterance_ms:
                logger.warning("Max utterance length reached, force-flushing")
                result = self._emit_utterance(is_final=True)

        else:
            # Silence frame
            self._pre_pad.append(frame)  # keep in pre-roll ring

            if self._state == SpeechState.SPEECH:
                self._silence_ms += self.cfg.frame_duration_ms
                # Keep appending frames during padding window
                self._speech_frames.append(frame)

                if self._silence_ms >= self.cfg.silence_threshold_ms:
                    # Pause is long enough → end of utterance
                    if self._speech_ms >= self.cfg.min_speech_ms:
                        result = self._emit_utterance(is_final=True)
                    else:
                        # Too short — discard (noise burst)
                        logger.debug(
                            f"Discarding short burst: {self._speech_ms}ms"
                        )
                        self._reset_speech()

        return result

    def _emit_utterance(self, is_final: bool) -> Optional[AudioChunk]:
        """Assemble all buffered speech frames into a single AudioChunk."""
        if not self._speech_frames:
            return None

        # Build context prefix (last N frames from previous utterance)
        context = list(self._context_buffer)

        # Combine context + current speech
        all_frames = context + self._speech_frames
        raw_pcm = b"".join(all_frames)

        # Convert to float32 [-1, 1]
        samples_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
        float_audio = samples_i16.astype(np.float32) / 32768.0

        self._utterance_count += 1
        chunk = AudioChunk(
            pcm_float=float_audio,
            sample_rate=self.cfg.sample_rate,
            is_speech=True,
            utterance_id=f"utt_{self._utterance_count:06d}",
            chunk_index=self._utterance_count,
            is_final=is_final,
            context_frames=context,
        )

        # Save last N frames into context buffer for next utterance
        for f in self._speech_frames[-len(self._context_buffer.maxlen and self._context_buffer or []):]:
            pass
        # Simple: save last context_keep_ms worth of frames
        frames_to_keep = int(self.cfg.context_keep_ms / self.cfg.frame_duration_ms)
        keep_frames = self._speech_frames[-frames_to_keep:] if frames_to_keep > 0 else []
        self._context_buffer.clear()
        for f in keep_frames:
            self._context_buffer.append(f)

        self._reset_speech()
        return chunk

    def _reset_speech(self):
        self._speech_frames = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._state = SpeechState.SILENCE
        self._pre_pad.clear()

    @property
    def current_buffer_float(self) -> Optional[np.ndarray]:
        """Returns the currently accumulating speech frames as a float32 array, or None if silent."""
        if self._state != SpeechState.SPEECH or not self._speech_frames:
            return None
        context = list(self._context_buffer)
        all_frames = context + self._speech_frames
        raw_pcm = b"".join(all_frames)
        samples_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
        return samples_i16.astype(np.float32) / 32768.0

    @staticmethod
    def _pre_emphasis(samples: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Simple first-order high-pass filter. Boosts high freq for better VAD."""
        emphasized = np.empty_like(samples)
        emphasized[0] = samples[0]
        emphasized[1:] = samples[1:] - coef * samples[:-1]
        return emphasized
