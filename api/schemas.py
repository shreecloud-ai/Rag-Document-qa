"""
Pydantic Schemas for FastAPI
Defines request and response data shapes.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    flagged: bool
    reason: str
    sources: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int

class ReviewItem(BaseModel):
    id: int
    timestamp: str
    question: str
    answer: str
    confidence: float
    reason: str
    status: str