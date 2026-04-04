"""
Hybrid Retriever - Minimal & Robust Version
"""

import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from .faiss_store import load_chunks_with_embeddings, load_faiss_index, search_faiss
from .bm25_store import build_bm25_index, search_bm25
from .reranker import rerank_chunks

import yaml
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

TOP_K = config.get("top_k", 5)

# Globals
model = None
faiss_index = None
bm25_index = None
chunks = None

def initialize_retriever():
    """Force reload on every call to pick up new uploads."""
    global model, faiss_index, bm25_index, chunks
    
    chunks_path = "data/chunks/all_chunks.json"
    
    if not os.path.exists(chunks_path):
        print("⚠️ No chunks file found. Please upload a document first.")
        chunks = []
        return []

    print(f"🔄 Reloading latest chunks from: {chunks_path}")

    chunks, _ = load_chunks_with_embeddings(chunks_path)
    faiss_index = load_faiss_index("faiss_index.bin")
    bm25_index, _ = build_bm25_index(chunks_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"✅ Loaded {len(chunks)} chunks successfully. Ready for questions.")
    return chunks

def hybrid_search(query: str, top_k: int = None) -> List[Dict]:
    """Pure reranker version for testing (no FAISS/BM25 fusion)."""
    if top_k is None:
        top_k = TOP_K
    
    print(f"\n🔍 Query: '{query}'")
    
    if model is None:
        initialize_retriever()
    
    if not chunks:
        print("   No chunks available.")
        return []

    # Use all chunks as candidates and let reranker decide
    reranked = rerank_chunks(query, chunks, top_k=top_k)
    
    final_indices = [r["metadata"].get("chunk_index") for r in reranked]
    print(f"   Final reranked order: {final_indices}")
    
    return reranked