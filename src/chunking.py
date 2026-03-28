"""
Chunking Module
Splits long documents into smaller overlapping chunks for better retrieval.
"""

import re
from typing import List, Dict, Any
import yaml
import os

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = config["chunk_size"]           # e.g. 500 characters
CHUNK_OVERLAP = config["chunk_overlap"]     # e.g. 100 characters
MIN_CHUNK_LENGTH = config["min_chunk_length"]

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex (simple but effective)."""
    # This regex tries to split on . ! ? followed by space or newline
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def create_chunks(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split document text into overlapping chunks with better sentence handling.
    """
    if not text or len(text) < MIN_CHUNK_LENGTH:
        return []

    # Use a simple character-based sliding window with overlap (more reliable for now)
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        
        # Try to cut at sentence boundary if possible
        if end < len(text):
            # Look for sentence end (. ! ?) in last 100 chars
            last_period = text.rfind('.', start, end)
            last_exclamation = text.rfind('!', start, end)
            last_question = text.rfind('?', start, end)
            cut_point = max(last_period, last_exclamation, last_question)
            
            if cut_point > start + 50:  # only use if it's meaningful
                end = cut_point + 1
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunk_id = f"{metadata['filename']}_chunk_{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_length": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                }
            })
            chunk_index += 1
        
        # Move start forward with overlap
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break

    return chunks

# Simple test function
if __name__ == "__main__":
    print("Chunking module loaded successfully!")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")