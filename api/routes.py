"""
FastAPI Routes
Defines the API endpoints for our RAG system.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from src.pipeline import answer_question
from .schemas import QueryRequest, QueryResponse, IngestResponse
from src.faiss_store import load_chunks_with_embeddings

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question to the RAG system."""
    try:
        result = answer_question(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            flagged=result["flagged"],
            reason=result["reason"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and incrementally index a document (much faster)."""
    try:
        upload_dir = "data/documents"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📄 Saved uploaded file: {file.filename}")

        # Use incremental indexer
        from src.index_manager import ingest_and_index_new_document
        chunks_created = ingest_and_index_new_document(file_path)

        return IngestResponse(
            message="Document successfully processed and added to the search index!",
            filename=file.filename,
            chunks_created=chunks_created
        )

    except Exception as e:
        print(f"❌ Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))