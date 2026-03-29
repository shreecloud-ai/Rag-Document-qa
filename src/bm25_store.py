"""
BM25 Keyword Store Module
Builds and searches a BM25 index for keyword-based retrieval.
"""

import os
import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import yaml
import numpy as np

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def load_chunks_for_bm25(json_path: str) -> tuple:
    """Load chunks and prepare text for BM25."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # BM25 works on tokenized text
    corpus = [chunk["text"].lower().split() for chunk in chunks]
    return chunks, corpus

def build_bm25_index(chunks_path: str = "data/chunks/sample_chunks_final.json"):
    """Build BM25 index from chunks."""
    chunks, corpus = load_chunks_for_bm25(chunks_path)
    
    bm25 = BM25Okapi(corpus)
    
    print(f"✅ BM25 index built with {len(chunks)} documents")
    return bm25, chunks

def search_bm25(bm25: BM25Okapi, chunks: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """Search BM25 and return top-K chunks with scores."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top indices sorted by score (highest first)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()          # copy to avoid modifying original
        chunk["bm25_score"] = float(scores[idx])
        chunk["rank"] = int(np.where(top_indices == idx)[0][0] + 1)
        results.append(chunk)
    
    return results