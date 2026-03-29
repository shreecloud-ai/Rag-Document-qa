# test_hybrid.py - Test Hybrid Retrieval (FAISS + BM25)
from src.hybrid_retriever import initialize_retriever, hybrid_search

print("🚀 Starting Hybrid Retrieval Test...\n")

# Initialize once
chunks = initialize_retriever()

print("\n" + "="*80)
print("HYBRID SEARCH RESULTS (FAISS + BM25 combined)")
print("="*80 + "\n")

test_questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain hybrid retrieval",
    "What is Sentence-BERT?",
    "Tell me about confidence scoring in RAG"
]

for i, question in enumerate(test_questions, 1):
    print(f"Question {i}: {question}")
    
    results = hybrid_search(question, top_k=3)
    
    for rank, chunk in enumerate(results, 1):
        print(f"  Rank {rank}: Chunk {chunk['metadata']['chunk_index']} | Hybrid Score: {chunk['hybrid_score']:.4f}")
        print(f"  Preview: {chunk['text'][:150]}...\n")
    
    print("-" * 80 + "\n")

print("✅ Hybrid retrieval test completed!")
print("You can now see how combining both retrievers usually gives better results.")