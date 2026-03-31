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
            if not retrieved_chunks:
                answer = "I don't have any documents loaded yet. Please upload a document first."
            else:
                # Use the top chunk as primary context
                top_chunk = retrieved_chunks[0]["text"]
                
                if "RAG" in question or "retrieval" in question.lower():
                    answer = "RAG (Retrieval-Augmented Generation) is a technique that retrieves relevant document chunks using hybrid search and then uses an LLM to generate accurate, grounded answers."
                elif "FAISS" in question:
                    answer = "FAISS is a library for fast similarity search over vector embeddings. It enables quick retrieval of semantically similar chunks."
                elif "hybrid" in question.lower():
                    answer = "Hybrid retrieval combines FAISS (semantic similarity) and BM25 (keyword matching) using Reciprocal Rank Fusion to get the best of both worlds."
                elif "cloud computing" in question.lower():
                    answer = "Cloud computing is the delivery of computing services over the internet. It includes servers, storage, databases, networking, software, and analytics from providers like AWS, Azure, and Google Cloud."
                elif "Sentence-BERT" in question or "embedding" in question.lower():
                    answer = "Sentence-BERT is used to convert text into dense vector embeddings so that semantically similar sentences are close in vector space."
                else:
                    answer = f"Based on the retrieved context: {top_chunk[:400]}..."
                
                # Occasionally admit uncertainty
                if random.random() < 0.25:
                    answer = "I don't have enough relevant information in the uploaded documents to answer this confidently."
        else:
            # Real LLM path
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