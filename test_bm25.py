# test_bm25.py - Test BM25 keyword search and compare with FAISS
from src.bm25_store import build_bm25_index, search_bm25
import os

print("🚀 Starting BM25 Test...\n")

# Build BM25 index
chunks_path = "data/chunks/sample_chunks_final.json"
bm25, chunks = build_bm25_index(chunks_path)

print("\nStep 1: BM25 index built successfully!\n")

# Test the same 5 questions we used for FAISS
test_questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain hybrid retrieval",
    "What is Sentence-BERT?",
    "Tell me about confidence scoring in RAG"
]

print("Step 2: Running BM25 searches...\n")

for i, question in enumerate(test_questions, 1):
    print(f"Question {i}: {question}")
    
    results = search_bm25(bm25, chunks, question, top_k=3)
    
    for result in results:
        print(f"  Rank {result['rank']}: Chunk {result['metadata']['chunk_index']} | BM25 Score: {result['bm25_score']:.4f}")
        print(f"  Preview: {result['text'][:120]}...\n")
    
    print("-" * 80 + "\n")

print("✅ BM25 test completed!")
print("Now we can compare how FAISS (semantic) and BM25 (keyword) behave differently.")