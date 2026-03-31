"""
FastAPI Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="RAG Document Q&A API",
    description="API for the RAG-powered document question answering system",
    version="0.1.0"
)

# CORS settings (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "RAG Document Q&A API is running! 🚀"}

@app.get("/health")
async def health():
    return {"status": "healthy"}