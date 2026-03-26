import re
import logging

logger = logging.getLogger(__name__)

def load_punctuation_model():
    """Stub for an AI punctuation model. Using heuristic regex instead for speed."""
    logger.info("Using heuristic regex punctuator for lowest latency.")
    return True

def build_paragraph(sentences):
    """Joins sentences into a paragraph."""
    return " ".join(sentences)

def fix_punctuation(text: str, is_final: bool = True) -> str:
    """Fixes dropped boundary punctuation from Whisper chunks."""
    if not text:
        return text
        
    text = text.strip()
    
    if len(text) > 0:
        pass # Let Whisper handle casing naturally
        
    # Pre-process lowercase 'i' and 'ok' to uppercase 'I' and 'OK' so regex works
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\b[oO][kK]\b', 'OK', text)
    text = re.sub(r'\bdoctor\b', 'Doctor', text, flags=re.IGNORECASE)
    
    # Avoid naive regex that injects periods before proper nouns.
    
    # Ensure trailing punctuation for final sentences
    if is_final and not re.search(r'[.!?]$', text) and len(text) > 2:
        # Don't punctuate if it ends abruptly with a dash
        if not text.endswith('-'):
            text += "."
            
    # Clean up double punctuation
    text = re.sub(r'\.\.', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r',\.', '.', text)
    
    return text
