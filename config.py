"""
config.py  —  v5 ultra-low latency configuration (NO REDIS)
========================================================

Enhanced configuration for ultra-low latency transcription without Redis dependencies.
Optimized for 100ms processing cycles with enhanced word boundary protection.

Key changes:
- Added ultra-low latency timing parameters
- Enhanced VAD settings for faster response
- Removed Redis-related configurations
- Added word boundary protection settings
"""

import os
from functools import lru_cache


class Settings:
    """Ultra-low latency application configuration."""

    APP_NAME: str = "NHS Realtime Consult"
    ENV: str = os.getenv("APP_ENV", "local")

    # ── Whisper / transcription ──────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small.en")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # or "cuda"
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")
    WHISPER_VAD_FILTER: bool = os.getenv("WHISPER_VAD_FILTER", "true").lower() in {"1", "true", "yes"}
    
    # Ultra-low latency beam size
    WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
    
    # Enhanced medical vocabulary for ultra-low latency processing
    MEDICAL_VOCAB_PROMPT: str = os.getenv(
        "MEDICAL_VOCAB_PROMPT",
        "Doctor patient consultation. NHS UK. "

        # Symptoms
        "headache, migraine, nausea, vomiting, dizziness, fatigue, fever, chills, "
        "shortness of breath, chest pain, palpitations, cough, wheezing, sore throat, "
        "abdominal pain, diarrhoea, constipation, bloating, heartburn, back pain, "
        "joint pain, muscle ache, rash, swelling, numbness, tingling, blurred vision, "
        "weight loss, weight gain, loss of appetite, insomnia, anxiety, depression, "

        # Conditions / Diagnoses
        "hypertension, hypotension, diabetes mellitus, type 2 diabetes, asthma, COPD, "
        "pneumonia, bronchitis, urinary tract infection, UTI, anaemia, hypothyroidism, "
        "hyperthyroidism, atrial fibrillation, heart failure, angina, myocardial infarction, "
        "stroke, TIA, epilepsy, dementia, Alzheimer's, Parkinson's, multiple sclerosis, "
        "rheumatoid arthritis, osteoarthritis, osteoporosis, fibromyalgia, gout, "
        "chronic kidney disease, liver disease, irritable bowel syndrome, IBS, "
        "Crohn's disease, ulcerative colitis, psoriasis, eczema, cellulitis, "
        "deep vein thrombosis, DVT, pulmonary embolism, sepsis, anaphylaxis, "

        # Medications
        "paracetamol, ibuprofen, aspirin, amoxicillin, penicillin, metformin, "
        "atorvastatin, simvastatin, ramipril, lisinopril, amlodipine, bisoprolol, "
        "atenolol, warfarin, apixaban, rivaroxaban, omeprazole, lansoprazole, "
        "salbutamol, prednisolone, levothyroxine, sertraline, fluoxetine, citalopram, "
        "diazepam, lorazepam, codeine, morphine, tramadol, gabapentin, pregabalin, "
        "methotrexate, hydroxychloroquine, adalimumab, insulin, dapagliflozin, "

        # Procedures / Clinical terms
        "blood pressure, heart rate, oxygen saturation, SpO2, ECG, X-ray, MRI, CT scan, "
        "ultrasound, biopsy, endoscopy, colonoscopy, spirometry, blood test, urine test, "
        "full blood count, FBC, HbA1c, cholesterol, creatinine, eGFR, liver function, "
        "thyroid function, INR, referral, outpatient, inpatient, discharge, triage, "
        "GP, consultant, physiotherapy, dietitian, pharmacist, prescription, dosage, "
        "once daily, twice daily, three times a day, with food, contraindication, "
        "side effects, allergy, anaphylaxis, informed consent, follow-up, sick note."
    )

    # ── Audio streaming ───────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    MAX_SESSION_MINUTES: int = int(os.getenv("MAX_SESSION_MINUTES", "60"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))

    # ── Balanced latency settings ───────────────────────────────────────────────────
    # TRANSCRIBE_EVERY_MS: Balanced latency - transcribe every 500ms for accuracy
    TRANSCRIBE_EVERY_MS: int = int(os.getenv("TRANSCRIBE_EVERY_MS", "500"))
    
    # MIN_TRANSCRIBE_WINDOW_MS: Minimum audio needed for meaningful transcription
    MIN_TRANSCRIBE_WINDOW_MS: int = int(os.getenv("MIN_TRANSCRIBE_WINDOW_MS", "300"))
    
    # Word boundary protection settings - optimized for accuracy
    SILENCE_TAIL_SEC: float = float(os.getenv("SILENCE_TAIL_SEC", "0.8"))
    
    # Memory buffer settings
    MAX_BUFFER_SEC: float = float(os.getenv("MAX_BUFFER_SEC", "20.0"))
    
    # VAD settings optimized for ultra-low latency
    VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "400"))

    # ── Single-worker sequential pipeline ───────────────────────────────────────────────
    NUM_WORKERS: int = 1

    # Rolling window settings
    SEGMENT_WINDOW_SEC: float = float(os.getenv("SEGMENT_WINDOW_SEC", "12.0"))
    SEGMENT_OVERLAP_SEC: float = float(os.getenv("SEGMENT_OVERLAP_SEC", "3.0"))
    MIN_CHUNK_TO_START_SEC: float = float(os.getenv("MIN_CHUNK_TO_START_SEC", "1.0"))

    # Live display settings
    LIVE_WINDOW_SEC: float = float(os.getenv("LIVE_WINDOW_SEC", "10.0"))

    # ── Queue limits (local only, no Redis) ───────────────────────────────────────────────
    MAX_AUDIO_QUEUE_CHUNKS: int = int(os.getenv("MAX_AUDIO_QUEUE_CHUNKS", "400"))
    MAX_TRANSCRIPT_QUEUE_MESSAGES: int = int(os.getenv("MAX_TRANSCRIPT_QUEUE_MESSAGES", "2000"))
    AUDIO_PUT_TIMEOUT_SEC: float = float(os.getenv("AUDIO_PUT_TIMEOUT_SEC", "2.0"))
    MAX_CONNECTIONS_PER_WORKER: int = int(os.getenv("MAX_CONNECTIONS_PER_WORKER", "2000"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
