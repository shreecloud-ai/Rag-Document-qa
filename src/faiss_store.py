"""
FAISS Vector Store Module
Builds, saves, loads, and searches a FAISS index using chunk embeddings.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import yaml

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

def load_chunks_with_embeddings(json_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load chunks and convert their embeddings back to numpy array."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Extract embeddings as numpy array (shape: num_chunks x embedding_dim)
    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    
    return chunks, embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a simple Flat FAISS index (exact search - good for small-medium data)."""
    dimension = embeddings.shape[1]   # 384 in our case
    
    # Create index
    index = faiss.IndexFlatL2(dimension)   # L2 = Euclidean distance
    index.add(embeddings)                  # Add all vectors to the index
    
    print(f"✅ FAISS index built with {index.ntotal} vectors (dim={dimension})")
    return index

def save_faiss_index(index: faiss.Index, index_name: str = "faiss_index.bin"):
    """Save FAISS index to disk."""
    index_path = os.path.join(INDEX_DIR, index_name)
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to: {index_path}")
    return index_path

def load_faiss_index(index_name: str = "faiss_index.bin"):
    index_path = os.path.join(INDEX_DIR, index_name)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    print(f"✅ FAISS index loaded with {index.ntotal} vectors")

    return index

def search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
    """Search FAISS index and return indices of top-K most similar chunks."""
    query_embedding = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    
    return indices[0].tolist()   # return list of chunk indices

# Simple test function
if __name__ == "__main__":
    print("FAISS store module loaded successfully!")
    print(f"Index will be saved in: {INDEX_DIR}")