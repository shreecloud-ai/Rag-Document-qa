"""
RAG Evaluation with Failure Analysis (20 queries)
"""

import sys
import os
import json
from src.pipeline import answer_question
from eval.metrics import run_evaluation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def run_benchmark():
    print("🚀 Starting RAG Evaluation (20 queries)...\n")
    
    eval_path = "data/eval/eval_set.json"
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    
    results = []
    failures = []
    
    for i, item in enumerate(eval_data, 1):
        query = item["query"]
        relevant_id = item["relevant_chunk_id"]
        
        print(f"{i:2d}. {query}")
        
        result = answer_question(query)
        
        retrieved_ids = [chunk["metadata"].get("chunk_index", -1) for chunk in result.get("sources", [])]
        
        results.append({
            "query": query,
            "relevant_chunk_id": relevant_id,
            "retrieved_chunk_ids": retrieved_ids
        })
        
        if relevant_id not in retrieved_ids[:3]:
            failures.append({
                "query": query,
                "relevant_id": relevant_id,
                "retrieved": retrieved_ids
            })
    
    metrics = run_evaluation(results)
    
    print(f"\nFailure cases ({len(failures)} queries):")
    for f in failures:
        print(f"   - '{f['query']}' | Relevant: {f['relevant_id']} | Retrieved: {f['retrieved']}")
    
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/detailed_eval.json", "w") as f:
        json.dump({"metrics": metrics, "failures": failures}, f, indent=2)
    
    print("\n✅ Evaluation completed! Results saved.")

if __name__ == "__main__":
    run_benchmark()