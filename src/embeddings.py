"""
Embeddings Module
Uses Sentence-BERT to convert text chunks into dense vector embeddings.
"""

import os
import yaml
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import torch, json

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Force consistent model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model once when the module is imported (efficient)
print(f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print("✅ Embedding model loaded successfully!\n")

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a single piece of text."""
    if not text or not text.strip():
        return np.zeros(config["embedding_dim"], dtype=np.float32)
    
    # SentenceTransformer returns numpy array by default
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def generate_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add embedding vectors to each chunk.
    Returns the same chunks list but with 'embedding' key added.
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk["text"])
        chunk["embedding"] = embedding.tolist()   # store as Python list for JSON saving
        
        if i % 10 == 0 and i > 0:   # progress feedback
            print(f"  → Processed {i}/{len(chunks)} chunks")
    
    print("✅ All embeddings generated successfully!")
    return chunks

# Simple test function
if __name__ == "__main__":
    print("Embeddings module loaded successfully!")
    print(f"Model: {EMBEDDING_MODEL} | Dimension: {config['embedding_dim']}")

def save_chunks_with_embeddings(chunks: List[Dict[str, Any]], output_filename: str = "chunks_with_embeddings.json"):
    """Save chunks with embeddings to data/chunks/ folder."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chunks")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Create a clean copy for saving (embeddings as list)
    save_data = []
    for chunk in chunks:
        save_data.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": chunk["embedding"]
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(save_data)} chunks with embeddings to: {output_path}")
    return output_path