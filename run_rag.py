# run_rag.py - Simple interactive RAG tester
from src.pipeline import answer_question

print("🤖 RAG Document Q&A System")
print("Type 'exit' or 'quit' to stop\n")

while True:
    question = input("\n❓ Ask a question: ").strip()
    
    if question.lower() in ['exit', 'quit', 'q']:
        print("👋 Goodbye!")
        break
    
    if not question:
        continue
        
    try:
        result = answer_question(question, show_details=True)
        print("\n" + "-"*80)
    except Exception as e:
        print(f"❌ Error: {e}")