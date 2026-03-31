"""
Improved RAG Evaluation with Better Analysis
"""

import sys
import os
import json
from src.pipeline import answer_question
from eval.metrics import run_evaluation

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def run_full_evaluation():
    print("🚀 Starting Improved RAG Evaluation...\n")
    
    eval_path = "data/eval/eval_set.json"
    if not os.path.exists(eval_path):
        print(f"❌ File not found: {eval_path}")
        return
    
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    
    results = []
    print(f"Evaluating {len(eval_data)} queries...\n")
    
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
        
        print(f"   Retrieved: {retrieved_ids} | Relevant: {relevant_id}\n")
    
    metrics = run_evaluation(results)
    
    # Save detailed results
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/detailed_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n📊 Summary:")
    print(f"   Hit@3 : {metrics['hit_at_k']:.4f}")
    print(f"   MRR   : {metrics['mrr']:.4f}")
    print(f"   Results saved to eval/results/")

if __name__ == "__main__":
    run_full_evaluation()