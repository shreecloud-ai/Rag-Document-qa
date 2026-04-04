"""
LLM Chain Module with Improved Mock Mode for Development
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import yaml
import random

load_dotenv()

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

USE_MOCK = True  # Set to False when you have fresh Gemini quota

if not USE_MOCK:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = ChatGoogleGenerativeAI(
            model=config["llm_model"],
            temperature=config["llm_temperature"],
            max_tokens=config["llm_max_tokens"],
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        print("✅ Using real Gemini LLM")
    except Exception:
        USE_MOCK = True

def format_context(retrieved_chunks: List[Dict]) -> str:
    """Format retrieved chunks into clean context for LLM."""
    if not retrieved_chunks:
        return "No relevant documents found."
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get("text", "").strip()
        context_parts.append(f"Source {i}:\n{text}")
    
    return "\n\n".join(context_parts)

def create_rag_chain():
    """Create the RAG chain with improved mock mode."""
    
    def rag_chain(inputs: Dict) -> Dict:
        question = inputs["question"]
        retrieved_chunks = inputs.get("retrieved_chunks", [])
        
        context = format_context(retrieved_chunks)
        
        if USE_MOCK:
            print("🔄 Using MOCK LLM mode")
            
            if not retrieved_chunks:
                answer = "I don't have any documents loaded yet. Please upload a document first."
            else:
                # Take top 2 chunks for better context
                top_context = " ".join([c.get("text", "")[:900] for c in retrieved_chunks[:2]])
                q_lower = question.lower()
                
                # Specific answers for health-related questions
                if "mental health" in q_lower:
                    answer = "Mental health is an important part of overall human health. It refers to emotional, psychological, and social well-being. Good mental health helps people handle stress, work productively, and contribute to their community."
                
                elif "sleep" in q_lower or "why sleep" in q_lower:
                    answer = "Sleep is very important for human health. It allows the body to repair tissues, consolidate memories, and maintain immune function. Good sleep improves mood, concentration, and overall well-being. Lack of sleep can lead to fatigue, poor decision-making, and weakened immunity."
                
                elif any(word in q_lower for word in ["component", "components", "parts of health", "what is human health"]):
                    answer = "The main components of human health are physical health, mental health, and social well-being. Physical health involves proper nutrition, exercise, and rest. Mental health includes emotional and psychological balance."
                
                else:
                    # Generic but grounded answer using actual retrieved content
                    answer = f"From the retrieved documents about human health:\n{top_context[:700]}..."
                
                # Reduce unnecessary low-confidence responses
                if random.random() < 0.08:
                    answer = "I don't have enough relevant information in the uploaded documents to answer this question confidently."
        
        else:
            # Real LLM (when quota allows)
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the question using only the following context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "context": context
        }
    
    return rag_chain

if __name__ == "__main__":
    print("LLM Chain module loaded successfully!")
    print(f"Mode: {'MOCK' if USE_MOCK else 'Real Gemini'}")