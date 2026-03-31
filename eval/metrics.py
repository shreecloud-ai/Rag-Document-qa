"""
Evaluation Metrics for RAG System
Calculates Hit@K and MRR
"""

import json
import os
from typing import List, Dict

def calculate_hit_at_k(retrieved_chunk_ids: List[int], relevant_chunk_id: int, k: int = 3) -> bool:
    """Check if relevant chunk is in top-K retrieved results."""
    return relevant_chunk_id in retrieved_chunk_ids[:k]

def calculate_mrr(retrieved_chunk_ids: List[int], relevant_chunk_id: int) -> float:
    """Calculate Mean Reciprocal Rank."""
    try:
        rank = retrieved_chunk_ids.index(relevant_chunk_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def run_evaluation(eval_data: List[Dict], top_k: int = 3):
    """Run full evaluation and print results."""
    hit_at_k_count = 0
    mrr_sum = 0.0
    
    for item in eval_data:
        retrieved_ids = item["retrieved_chunk_ids"]
        relevant_id = item["relevant_chunk_id"]
        
        if calculate_hit_at_k(retrieved_ids, relevant_id, top_k):
            hit_at_k_count += 1
        
        mrr_sum += calculate_mrr(retrieved_ids, relevant_id)
    
    total = len(eval_data)
    hit_at_k_score = hit_at_k_count / total
    mrr_score = mrr_sum / total
    
    print("\n" + "="*50)
    print("RAG EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries evaluated : {total}")
    print(f"Hit@{top_k}               : {hit_at_k_score:.4f} (target ≥ 0.91)")
    print(f"MRR                    : {mrr_score:.4f} (target ≥ 0.87)")
    print("="*50)
    
    return {
        "hit_at_k": hit_at_k_score,
        "mrr": mrr_score,
        "total_queries": total
    }

if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully!")