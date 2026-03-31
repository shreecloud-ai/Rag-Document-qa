"""
Hybrid Retriever Module
Combines FAISS + BM25 using Reciprocal Rank Fusion (RRF)
"""

import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Import our modules
from .faiss_store import load_chunks_with_embeddings, load_faiss_index, search_faiss
from .bm25_store import build_bm25_index, search_bm25

# Load config
import yaml
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

TOP_K = config.get("top_k", 5)
HYBRID_ALPHA = 0.8

# Global variables
model = None
faiss_index = None
bm25_index = None
chunks = None

def initialize_retriever():
    """Reload chunks every time to pick up new uploads."""
    global model, faiss_index, bm25_index, chunks
    
    chunks_path = "data/chunks/all_chunks.json"
    
    if not os.path.exists(chunks_path):
        print("⚠️ No chunks file found. Upload a document first.")
        chunks = []
        faiss_index = None
        bm25_index = None
        model = SentenceTransformer(config["embedding_model"])
        return []

    print(f"🔄 Reloading latest chunks from: {chunks_path}")
    
    # Force reload every time
    chunks, _ = load_chunks_with_embeddings(chunks_path)
    faiss_index = load_faiss_index("faiss_index.bin")
    bm25_index, _ = build_bm25_index(chunks_path)
    model = SentenceTransformer(config["embedding_model"])
    
    print(f"✅ Loaded {len(chunks)} chunks successfully. Ready for questions.")
    return chunks

def hybrid_search(query: str, top_k: int = None) -> List[Dict]:
    if top_k is None:
        top_k = TOP_K
    
    if faiss_index is None or bm25_index is None or model is None:
        initialize_retriever()
    
    # 1. FAISS search (GET SCORES)
    query_embedding = model.encode(query)
    distances, indices = faiss_index.search(
        np.array([query_embedding]).astype("float32"),
        top_k * 2
    )
    
    faiss_scores = {}
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            # Convert distance → similarity
            score = 1 / (1 + distances[0][i])
            faiss_scores[idx] = score

    # 2. BM25 search (WITH SCORES)
    bm25_results = search_bm25(bm25_index, chunks, query, top_k=top_k*2)
    
    bm25_scores = {}
    for rank, chunk in enumerate(bm25_results):
        idx = chunk["metadata"]["chunk_index"]
        bm25_scores[idx] = 1 / (rank + 1)

    # 3. Normalize scores
    def normalize(scores_dict):
        if not scores_dict:
            return scores_dict
        max_score = max(scores_dict.values())
        return {k: v / max_score for k, v in scores_dict.items()}
    
    faiss_scores = normalize(faiss_scores)
    bm25_scores = normalize(bm25_scores)

    # 4. Weighted fusion (BETTER than RRF)
    alpha = HYBRID_ALPHA  # 0.8
    final_scores = {}
    
    all_indices = set(faiss_scores) | set(bm25_scores)
    
    for idx in all_indices:
        final_scores[idx] = (
            alpha * faiss_scores.get(idx, 0) +
            (1 - alpha) * bm25_scores.get(idx, 0)
        )

    # 5. Fast lookup map
    chunk_map = {
        c["metadata"]["chunk_index"]: c
        for c in chunks
    }

    # 6. Sort results
    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in sorted_items[:top_k]:
        if idx in chunk_map:
            chunk_copy = chunk_map[idx].copy()
            chunk_copy["hybrid_score"] = round(score, 4)
            results.append(chunk_copy)

    return results

def reciprocal_rank_fusion(faiss_results: List[int], bm25_results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Combine rankings using Reciprocal Rank Fusion."""
    scores = {}
    
    # FAISS scores
    for rank, idx in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + 60)
    
    # BM25 scores
    for rank, chunk in enumerate(bm25_results):
        idx = chunk["metadata"].get("chunk_index")
        if idx is not None:
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + 60)
    
    # Sort and return top chunks
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in sorted_items[:top_k]:
        chunk = next((c for c in chunks if c["metadata"].get("chunk_index") == idx), None)
        if chunk:
            chunk_copy = chunk.copy()
            chunk_copy["hybrid_score"] = round(score, 4)
            results.append(chunk_copy)
    
    return results

# Simple test
if __name__ == "__main__":
    print("Hybrid retriever module loaded successfully!")