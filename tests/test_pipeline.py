"""
NHS Transcription — Pipeline Integration Test
Tests the full audio → VAD → queue → Whisper path
without needing a browser or WebSocket connection.

Usage:
    python tests/test_pipeline.py                    # uses a generated sine tone
    python tests/test_pipeline.py path/to/audio.wav  # uses a real WAV file
"""

import sys
import os
import time
import numpy as np
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_pipeline")


def generate_test_audio(duration_s: float = 5.0, sample_rate: int = 16000) -> bytes:
    """
    Generate synthetic speech-like audio: bursts of sine tone separated by silence.
    Returns raw 16-bit PCM bytes.
    """
    samples_total = int(duration_s * sample_rate)
    audio = np.zeros(samples_total, dtype=np.float32)

    # Add three "speech" bursts at different times
    for start_s, dur_s, freq in [(0.3, 1.2, 300), (2.0, 0.8, 350), (3.5, 1.0, 320)]:
        start = int(start_s * sample_rate)
        end = int((start_s + dur_s) * sample_rate)
        t = np.linspace(0, dur_s, end - start)
        burst = 0.4 * np.sin(2 * np.pi * freq * t)
        # Add harmonics to make it more speech-like
        burst += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        burst += 0.08 * np.sin(2 * np.pi * freq * 3 * t)
        audio[start:end] = burst

    # Add light background noise
    audio += 0.005 * np.random.randn(samples_total)

    # Convert to int16 PCM
    pcm_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm_i16.tobytes()


def load_wav(path: str, target_sr: int = 16000) -> bytes:
    """Load a WAV file and resample to 16kHz mono if needed."""
    import soundfile as sf
    import scipy.signal as sps

    data, sr = sf.read(path, dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)  # stereo → mono
    if sr != target_sr:
        n_out = int(len(data) * target_sr / sr)
        data = sps.resample(data, n_out).astype(np.int16)
    return data.tobytes()


def test_audio_preprocessor(raw_pcm: bytes):
    """Test VAD and chunking without Whisper."""
    from core.audio_processor import AudioPreprocessor, VADConfig

    logger.info("=" * 50)
    logger.info("TEST 1: AudioPreprocessor (VAD + chunking)")
    logger.info("=" * 50)

    cfg = VADConfig(
        silence_threshold_ms=600,
        min_speech_ms=150,
        context_keep_ms=800,
    )
    proc = AudioPreprocessor(config=cfg)

    # Feed audio in 50ms chunks (simulates network packets)
    frame_size = int(16000 * 0.05 * 2)  # 50ms in bytes
    chunks_emitted = []

    t0 = time.time()
    offset = 0
    while offset < len(raw_pcm):
        pkt = raw_pcm[offset : offset + frame_size]
        offset += frame_size
        audio_chunks = proc.feed(pkt)
        for chunk in audio_chunks:
            chunks_emitted.append(chunk)
            logger.info(
                f"  Chunk emitted: {chunk.utterance_id} | "
                f"duration={len(chunk.pcm_float)/16000:.2f}s | "
                f"final={chunk.is_final}"
            )

    # Flush
    final = proc.flush()
    if final:
        chunks_emitted.append(final)
        logger.info(f"  Flush chunk: {final.utterance_id} | final={final.is_final}")

    elapsed = (time.time() - t0) * 1000
    logger.info(f"\nTotal chunks emitted: {len(chunks_emitted)} in {elapsed:.0f}ms")
    return chunks_emitted


def test_transcriber(chunks):
    """Test Whisper transcription on the emitted chunks."""
    from core.transcriber import WhisperTranscriber

    logger.info("=" * 50)
    logger.info("TEST 2: WhisperTranscriber")
    logger.info("=" * 50)

    model_name = os.getenv("WHISPER_MODEL", "base.en")
    logger.info(f"Loading model: {model_name}")

    transcriber = WhisperTranscriber(
        model_name=model_name,
        no_speech_threshold=0.6,
        logprob_threshold=-1.2,
    )

    for chunk in chunks:
        t0 = time.time()
        result = transcriber.transcribe(
            audio_float32=chunk.pcm_float,
            utterance_id=chunk.utterance_id,
            is_final=chunk.is_final,
        )
        elapsed = (time.time() - t0) * 1000

        if result:
            logger.info(f"  [{chunk.utterance_id}] → '{result.text}' ({elapsed:.0f}ms)")
        else:
            logger.info(f"  [{chunk.utterance_id}] → (silence/rejected) ({elapsed:.0f}ms)")





if __name__ == "__main__":
    # Determine audio source
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        logger.info(f"Loading audio from: {sys.argv[1]}")
        raw_pcm = load_wav(sys.argv[1])
    else:
        logger.info("Using generated synthetic audio (provide a WAV path for real speech)")
        raw_pcm = generate_test_audio(duration_s=6.0)

    logger.info(f"Audio: {len(raw_pcm) / 2 / 16000:.1f}s at 16kHz")

    # Run tests
    chunks = test_audio_preprocessor(raw_pcm)

    if chunks:
        test_transcriber(chunks)
    else:
        logger.warning("No chunks emitted — check VAD settings or audio content")



    logger.info("\nAll tests complete.")
