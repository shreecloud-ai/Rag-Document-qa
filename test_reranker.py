# test_reranker.py - Test Hybrid Retrieval + Reranking
from src.hybrid_retriever import initialize_retriever, hybrid_search
from src.reranker import rerank_chunks

print("🚀 Starting Hybrid + Reranker Test...\n")

# Initialize retriever
initialize_retriever()

test_questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain hybrid retrieval",
    "What is Sentence-BERT?",
    "Tell me about confidence scoring in RAG"
]

for i, question in enumerate(test_questions, 1):
    print(f"\nQuestion {i}: {question}")
    
    # 1. Hybrid retrieval (gets candidates)
    hybrid_results = hybrid_search(question, top_k=6)
    
    print(f"   Hybrid retrieved {len(hybrid_results)} candidates")
    
    # 2. Rerank them
    reranked = rerank_chunks(question, hybrid_results, top_k=3)
    
    print("   After Reranking:")
    for rank, chunk in enumerate(reranked, 1):
        score = chunk.get("rerank_score", 0)
        print(f"     Rank {rank}: Chunk {chunk['metadata']['chunk_index']} | Rerank Score: {score:.4f}")
        print(f"     Preview: {chunk['text'][:140]}...\n")
    
    print("-" * 80)

print("✅ Hybrid + Reranker test completed!")