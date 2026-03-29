"""
LLM Chain Module with Mock Mode for Development
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

# Try to use real Gemini, fallback to mock if quota issue
USE_MOCK = True  # Set to False when you have fresh quota

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

if USE_MOCK:
    print("🔄 Using MOCK LLM mode (for development - quota bypass)")

prompt_template = """You are a helpful assistant. Answer based on the context."""

prompt = None  # not needed in mock mode

def format_context(retrieved_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"--- Source {i} ---\n{chunk['text']}\n")
    return "\n".join(context_parts)

def create_rag_chain():
    def rag_chain(inputs):
        question = inputs["question"]
        retrieved_chunks = inputs["retrieved_chunks"]
        context = format_context(retrieved_chunks)
        
        if USE_MOCK:
            # Simulate realistic answer
            if "RAG" in question:
                answer = "RAG (Retrieval-Augmented Generation) combines retrieval of relevant document chunks with a generative LLM to produce accurate answers."
            elif "FAISS" in question:
                answer = "FAISS is a library for fast similarity search over vector embeddings."
            elif "hybrid" in question.lower():
                answer = "Hybrid retrieval combines FAISS (semantic) and BM25 (keyword) using Reciprocal Rank Fusion for better results."
            else:
                answer = "The system uses confidence scoring to evaluate answer quality and flags low-confidence responses for human review."
            
            # Occasionally simulate "I don't know"
            if random.random() < 0.2:
                answer = "I don't have enough information to answer this question confidently."
        else:
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