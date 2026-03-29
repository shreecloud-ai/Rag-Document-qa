"""
Hybrid Retriever Module
Combines FAISS (semantic) + BM25 (keyword) using Reciprocal Rank Fusion (RRF).
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Import our existing modules
from src.faiss_store import load_chunks_with_embeddings, load_faiss_index, search_faiss
from src.bm25_store import build_bm25_index, search_bm25

# Load configuration
import yaml
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

TOP_K = config["top_k"]
HYBRID_ALPHA = config.get("hybrid_alpha", 0.7)   # Weight for FAISS (0.0 = pure BM25, 1.0 = pure FAISS)

# Global variables (loaded once)
model = SentenceTransformer(config["embedding_model"])
faiss_index = None
bm25_index = None
chunks = None

def initialize_retriever():
    """Load FAISS, BM25, and chunks once."""
    global faiss_index, bm25_index, chunks
    
    chunks_path = "data/chunks/sample_chunks_final.json"
    
    # Load chunks and FAISS
    chunks, embeddings = load_chunks_with_embeddings(chunks_path)
    faiss_index = load_faiss_index("faiss_index.bin")
    
    # Build BM25
    bm25_index, _ = build_bm25_index(chunks_path)
    
    print("✅ Hybrid retriever initialized successfully!")
    print(f"   • FAISS weight (alpha): {HYBRID_ALPHA}")
    print(f"   • Total chunks available: {len(chunks)}")
    return chunks

def reciprocal_rank_fusion(faiss_results: List[int], bm25_results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Combine FAISS and BM25 rankings using Reciprocal Rank Fusion."""
    scores = {}
    
    # Score from FAISS (semantic) - using indices
    for rank, idx in enumerate(faiss_results):
        if idx not in scores:
            scores[idx] = 0.0
        scores[idx] += 1.0 / (rank + 60)   # RRF formula
    
    # Score from BM25 (keyword)
    for rank, chunk in enumerate(bm25_results):
        # Get chunk_index safely
        idx = chunk["metadata"].get("chunk_index")
        if idx is None:
            continue
        if idx not in scores:
            scores[idx] = 0.0
        scores[idx] += 1.0 / (rank + 60)
    
    # Sort by combined RRF score (highest first)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top-K chunks with full data
    results = []
    for idx, score in sorted_items[:top_k]:
        # Find the full chunk
        chunk = next((c for c in chunks if c["metadata"].get("chunk_index") == idx), None)
        if chunk is None:
            continue
        chunk_copy = chunk.copy()
        chunk_copy["hybrid_score"] = round(score, 4)
        results.append(chunk_copy)
    
    return results

def hybrid_search(query: str, top_k: int = None) -> List[Dict]:
    """Main hybrid search function."""
    if top_k is None:
        top_k = TOP_K
    
    if faiss_index is None or bm25_index is None:
        initialize_retriever()
    
    # 1. FAISS search (semantic)
    query_embedding = model.encode(query)
    faiss_indices = search_faiss(faiss_index, query_embedding, top_k=top_k*2)  # get more candidates
    
    # 2. BM25 search (keyword)
    bm25_results = search_bm25(bm25_index, chunks, query, top_k=top_k*2)
    
    # 3. Combine with RRF
    final_results = reciprocal_rank_fusion(faiss_indices, bm25_results, top_k)
    
    return final_results

# Simple test
if __name__ == "__main__":
    print("Hybrid retriever module loaded successfully!")