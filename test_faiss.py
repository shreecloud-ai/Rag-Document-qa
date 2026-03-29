# test_faiss.py - Build FAISS index and test similarity search
from src.ingestion import ingest_document
from src.chunking import create_chunks
from src.embeddings import generate_embeddings_for_chunks, save_chunks_with_embeddings
from src.faiss_store import (
    load_chunks_with_embeddings,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    search_faiss
)
import os
import numpy as np
from sentence_transformers import SentenceTransformer

print("🚀 Starting FAISS Index Build & Search Test...\n")

# Option 1: Use existing saved chunks (recommended)
chunks_path = "data/chunks/sample_chunks_final.json"

if os.path.exists(chunks_path):
    print("Using existing chunks with embeddings...")
    chunks, embeddings = load_chunks_with_embeddings(chunks_path)
else:
    print("Creating fresh chunks + embeddings...")
    ingested = ingest_document("data/documents/sample.txt")
    chunks = create_chunks(ingested['text'], ingested['metadata'])
    chunks = generate_embeddings_for_chunks(chunks)
    save_chunks_with_embeddings(chunks, "sample_chunks_final.json")
    chunks, embeddings = load_chunks_with_embeddings("data/chunks/sample_chunks_final.json")

# Build FAISS index
print("\nStep 1: Building FAISS index...")
index = build_faiss_index(embeddings)

# Save the index
print("\nStep 2: Saving FAISS index...")
save_faiss_index(index, "faiss_index.bin")

# Load the index back (to test save/load)
print("\nStep 3: Loading FAISS index...")
loaded_index = load_faiss_index("faiss_index.bin")

# Test searches with 5 sample questions
print("\nStep 4: Running sample searches...\n")

test_questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain hybrid retrieval",
    "What is Sentence-BERT?",
    "Tell me about confidence scoring in RAG"
]

# Load model for query embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

for i, question in enumerate(test_questions, 1):
    print(f"Question {i}: {question}")
    
    # Generate embedding for the question
    query_embedding = model.encode(question)
    
    # Search FAISS
    top_indices = search_faiss(loaded_index, query_embedding, top_k=3)
    
    print(f"Top {len(top_indices)} matching chunk indices: {top_indices}")
    
    for rank, idx in enumerate(top_indices):
        chunk = chunks[idx]
        print(f"  Rank {rank+1}: Chunk {idx} | Score distance based")
        print(f"  Text preview: {chunk['text'][:150]}...\n")
    
    print("-" * 80 + "\n")

print("✅ FAISS build and search test completed successfully!")
