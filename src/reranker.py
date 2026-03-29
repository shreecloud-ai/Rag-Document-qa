"""
Reranker Module
Re-ranks retrieved chunks using a cross-encoder for better relevance scoring.
"""

import os
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import yaml

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

RERANK_TOP_K = config.get("rerank_top_k", 10)

# Load reranker model (only once)
print("Loading reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
print("✅ Reranker model loaded successfully!\n")

def rerank_chunks(query: str, retrieved_chunks: List[Dict], top_k: int = None) -> List[Dict]:
    """
    Re-rank chunks using cross-encoder.
    Returns the top-K most relevant chunks with new scores.
    """
    if top_k is None:
        top_k = RERANK_TOP_K
    
    if not retrieved_chunks:
        return []
    
    # Prepare pairs: (query, chunk_text)
    pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
    
    # Get relevance scores
    scores = reranker_model.predict(pairs)
    
    # Add scores to chunks and sort
    for chunk, score in zip(retrieved_chunks, scores):
        chunk = chunk.copy()
        chunk["rerank_score"] = float(score)
    
    # Sort by rerank_score (higher = more relevant)
    reranked = sorted(retrieved_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    
    return reranked[:top_k]

# Simple test function
if __name__ == "__main__":
    print("Reranker module loaded successfully!")