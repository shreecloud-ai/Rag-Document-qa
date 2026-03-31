"""
RAG Pipeline Orchestrator
Single clean entry point for the entire RAG system.
"""

from src.hybrid_retriever import initialize_retriever, hybrid_search
from src.reranker import rerank_chunks
from src.llm_chain import create_rag_chain
from src.confidence import calculate_confidence, log_to_review_queue
from typing import Dict, Any

# Initialize once (but will be refreshed after upload)
initialize_retriever()
rag_chain = create_rag_chain()

def answer_question(question: str, show_details: bool = False) -> Dict[str, Any]:
    """Main public function with better error handling."""
    try:
        print(f"🔍 Question: {question}")

        # 1. Retrieval
        hybrid_results = hybrid_search(question, top_k=8)
        reranked_chunks = rerank_chunks(question, hybrid_results, top_k=5)

        # 2. Generate answer
        llm_result = rag_chain({
            "question": question,
            "retrieved_chunks": reranked_chunks
        })

        answer = llm_result["answer"]
        sources = llm_result["sources"]

        # 3. Confidence
        confidence_result = calculate_confidence(reranked_chunks, answer, question)

        final_result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence_result["score"],
            "flagged": confidence_result["flagged"],
            "reason": confidence_result["reason"],
            "details": confidence_result.get("details", {})
        }

        if confidence_result["flagged"]:
            from src.confidence import log_to_review_queue
            log_to_review_queue(final_result, question, answer)

        # Print summary
        print(f"🤖 Answer: {answer[:250]}{'...' if len(answer) > 250 else ''}")
        print(f"📊 Confidence: {confidence_result['score']:.3f} | Flagged: {confidence_result['flagged']}")

        return final_result

    except Exception as e:
        print(f"❌ Error in answer_question: {str(e)}")
        # Return safe fallback answer
        return {
            "question": question,
            "answer": "Sorry, I encountered an error while processing your question. Please try again.",
            "sources": [],
            "confidence": 0.0,
            "flagged": True,
            "reason": f"Internal error: {str(e)}",
            "details": {}
        }