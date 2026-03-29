"""
End-to-End RAG Pipeline
Orchestrates the full flow: Retrieval → Reranking → LLM → Confidence Scoring
"""

from src.hybrid_retriever import initialize_retriever, hybrid_search
from src.reranker import rerank_chunks
from src.llm_chain import create_rag_chain
from src.confidence import calculate_confidence
from typing import Dict, Any

# Initialize retriever once when module is imported
initialize_retriever()

# Create the RAG chain once
rag_chain = create_rag_chain()

def answer_question(question: str) -> Dict[str, Any]:
    """
    Main function: Ask a question → Get answer with confidence score.
    """
    print(f"🔍 Processing question: {question}")

    # 1. Hybrid Retrieval
    hybrid_results = hybrid_search(question, top_k=8)

    # 2. Rerank
    reranked_chunks = rerank_chunks(question, hybrid_results, top_k=5)

    # 3. Generate answer
    llm_result = rag_chain({
        "question": question,
        "retrieved_chunks": reranked_chunks
    })

    answer = llm_result["answer"]
    sources = llm_result["sources"]

    # 4. Calculate confidence
    confidence_result = calculate_confidence(reranked_chunks, answer, question)

    final_result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": confidence_result["score"],
        "flagged": confidence_result["flagged"],
        "reason": confidence_result["reason"],
        "details": confidence_result["details"]
    }

    # Log to review queue if flagged
    if confidence_result["flagged"]:
        from src.confidence import log_to_review_queue
        log_to_review_queue(final_result, question, answer)

    # Print summary
    print(f"🤖 Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print(f"📊 Confidence: {confidence_result['score']:.3f} | Flagged: {confidence_result['flagged']}")
    if confidence_result["flagged"]:
        print(f"⚠️  Reason: {confidence_result['reason']}")

    return final_result