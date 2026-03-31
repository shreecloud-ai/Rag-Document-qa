"""
Confidence Scoring Module
Calculates confidence score for RAG answers and flags low-confidence responses.
"""

import re
from typing import List, Dict, Any
import yaml
import os
import csv
from datetime import datetime

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

CONFIDENCE_THRESHOLD = config["confidence_threshold"]

def calculate_confidence(retrieved_chunks: List[Dict], answer: str, query: str) -> Dict[str, Any]:
    """
    Improved confidence scoring with better heuristics.
    """
    if not retrieved_chunks:
        return {"score": 0.0, "flagged": True, "reason": "No chunks retrieved"}

    # 1. Retrieval strength (based on rerank/hybrid scores)
    top_scores = [chunk.get("rerank_score", chunk.get("hybrid_score", 0)) for chunk in retrieved_chunks[:3]]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
    retrieval_strength = max(0.0, min(1.0, (avg_score + 3) / 8))   # rough normalization

    # 2. Groundedness - check how much of the answer can be found in chunks
    answer_lower = answer.lower()
    chunk_text = " ".join([c["text"].lower() for c in retrieved_chunks])
    
    # Count important words from answer that appear in context
    answer_words = set(re.findall(r'\b\w{4,}\b', answer_lower))
    context_words = set(re.findall(r'\b\w{4,}\b', chunk_text))
    overlap_ratio = len(answer_words & context_words) / max(len(answer_words), 1)
    groundedness = min(1.0, overlap_ratio * 1.5)   # boost a bit

    # 3. Chunk coverage
    coverage = min(1.0, len(retrieved_chunks) / 4.0)

    # 4. Bonus: penalize if answer says "I don't have enough information"
    if "don't have enough" in answer_lower or "insufficient" in answer_lower:
        groundedness *= 0.6

    # Final score
    score = (
        retrieval_strength * config["retrieval_weight"] +
        groundedness * config["groundedness_weight"] +
        coverage * config["coverage_weight"]
    )
    score = max(0.0, min(1.0, score))

    flagged = score < CONFIDENCE_THRESHOLD

    reason = []
    if retrieval_strength < 0.5:
        reason.append("Weak retrieval")
    if groundedness < 0.6:
        reason.append("Low groundedness (possible hallucination)")
    if coverage < 0.5:
        reason.append("Limited chunk coverage")
    if flagged and not reason:
        reason.append("Overall low confidence")

    return {
        "score": round(score, 3),
        "flagged": flagged,
        "reason": ", ".join(reason) if reason else "High confidence",
        "details": {
            "retrieval_strength": round(retrieval_strength, 3),
            "groundedness": round(groundedness, 3),
            "coverage": round(coverage, 3)
        }
    }
def log_to_review_queue(result: Dict[str, Any], question: str, answer: str):
    """Log flagged answers to CSV for human review."""
    review_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(review_dir, exist_ok=True)
    review_path = os.path.join(review_dir, "review_queue.csv")
    
    file_exists = os.path.exists(review_path)
    
    with open(review_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "confidence", "reason", "status"])
        
        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer[:500],           # truncate long answers
            result["confidence"],
            result["reason"],
            "pending"
        ])
    
    print(f"📝 Flagged answer logged to review queue → {review_path}")