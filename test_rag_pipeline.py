# test_rag_pipeline.py - First End-to-End RAG Test!
from src.hybrid_retriever import initialize_retriever, hybrid_search
from src.reranker import rerank_chunks
from src.llm_chain import create_rag_chain, format_context

print("🚀 Starting Full RAG Pipeline Test...\n")

# 1. Initialize retriever (loads FAISS + BM25)
initialize_retriever()

# 2. Create the RAG chain
rag_chain = create_rag_chain()

test_questions = [
    "What is RAG and how does it work?",
    "What is FAISS used for?",
    "Explain hybrid retrieval in this system",
    "What is Sentence-BERT and why do we use it?",
    "Tell me about confidence scoring in RAG systems"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}: {question}")
    print(f"{'='*80}\n")

    # Step A: Hybrid Retrieval
    hybrid_results = hybrid_search(question, top_k=6)
    
    # Step B: Rerank the results
    reranked_chunks = rerank_chunks(question, hybrid_results, top_k=4)
    
    print(f"Retrieved and reranked {len(reranked_chunks)} relevant chunks\n")

    # Step C: Generate answer with Gemini
    result = rag_chain({
        "question": question,
        "retrieved_chunks": reranked_chunks
    })

    # Display the answer
    print("🤖 Gemini Answer:")
    print(result["answer"])
    print("\n" + "-"*60)

    # Show sources
    print("📚 Sources used:")
    for j, chunk in enumerate(result["sources"], 1):
        print(f"  {j}. Chunk {chunk['metadata']['chunk_index']} | Text preview: {chunk['text'][:100]}...")

    print("\n" + "="*80)

print("\n✅ Full RAG Pipeline test completed!")
print("You just ran your first complete RAG system: Retrieval → Reranking → LLM Answer!")
