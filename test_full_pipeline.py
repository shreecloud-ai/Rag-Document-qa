# test_full_pipeline.py - Final End-to-End Test
from src.pipeline import answer_question

print("🚀 Running Final End-to-End RAG Pipeline Test...\n")

questions = [
    "What is RAG?",
    "What does FAISS do?",
    "How does hybrid retrieval work?",
    "Tell me about confidence scoring",
    "What is the capital of France?"          # Out-of-scope → should be low confidence
]

for question in questions:
    print(f"\n{'='*80}")
    result = answer_question(question)
    print(f"{'='*80}\n")

print("\n✅ Full pipeline test completed!")