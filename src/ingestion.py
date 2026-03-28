"""
Document Ingestion Module
Handles loading and text extraction from PDF, TXT, and DOCX files.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Document loaders
from pypdf import PdfReader
from docx import Document as DocxDocument

def load_text_file(file_path: str) -> str:
    """Load plain text file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf_file(file_path: str) -> str:
    """Load PDF file and extract text from all pages."""
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        text += f"\n--- Page {page_num} ---\n{page_text}\n"
    return text.strip()

def load_docx_file(file_path: str) -> str:
    """Load DOCX file and extract text."""
    doc = DocxDocument(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def ingest_document(file_path: str) -> Dict[str, Any]:
    """
    Main function to ingest a single document.
    Returns a dictionary with text and metadata.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.txt':
        text = load_text_file(str(file_path))
    elif file_ext == '.pdf':
        text = load_pdf_file(str(file_path))
    elif file_ext == '.docx':
        text = load_docx_file(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported: .txt, .pdf, .docx")
    
    metadata = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_type": file_ext,
        "total_characters": len(text),
        "ingested_at": __import__('datetime').datetime.now().isoformat()
    }
    
    return {
        "text": text,
        "metadata": metadata
    }

# Simple test function (we'll use this later)
if __name__ == "__main__":
    print("Ingestion module loaded successfully!")
    print("Ready to ingest PDF, TXT, and DOCX files.")
