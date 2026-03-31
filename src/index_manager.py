"""
Index Manager - Proper incremental + reset support
"""

import os
from src.ingestion import ingest_document
from src.chunking import create_chunks
from src.embeddings import generate_embeddings_for_chunks, save_chunks_with_embeddings
from src.faiss_store import load_chunks_with_embeddings, build_faiss_index, save_faiss_index
from src.bm25_store import build_bm25_index

def ingest_and_index_new_document(file_path: str, reset_index: bool = False):
    """Process new document. If reset_index=True, clear previous data."""
    print(f"🚀 Processing: {file_path} | Reset: {reset_index}")

    # 1. Ingest
    ingested = ingest_document(file_path)

    # 2. Chunk
    new_chunks = create_chunks(ingested["text"], ingested["metadata"])

    # 3. Embed
    new_chunks = generate_embeddings_for_chunks(new_chunks)

    # 4. Decide whether to reset or append
    chunks_path = "data/chunks/all_chunks.json"
    
    if reset_index or not os.path.exists(chunks_path):
        all_chunks = new_chunks
        print("🔄 Starting fresh index with this document")
    else:
        existing_chunks, _ = load_chunks_with_embeddings(chunks_path)
        all_chunks = existing_chunks + new_chunks
        print(f"➕ Appended to existing {len(existing_chunks)} chunks")

    # 5. Save
    save_chunks_with_embeddings(all_chunks, "all_chunks.json")

    # 6. Rebuild indexes
    _, embeddings = load_chunks_with_embeddings(chunks_path)
    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index, "faiss_index.bin")

    bm25_index, _ = build_bm25_index(chunks_path)

    print(f"✅ Done! Total chunks now: {len(all_chunks)}")
    return len(new_chunks)