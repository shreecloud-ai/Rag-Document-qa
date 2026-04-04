"""
Index Manager - Incremental indexing for fast uploads
"""

import os
from src.ingestion import ingest_document
from src.chunking import create_chunks
from src.embeddings import generate_embeddings_for_chunks, save_chunks_with_embeddings
from src.faiss_store import load_chunks_with_embeddings, build_faiss_index, save_faiss_index
from src.bm25_store import build_bm25_index

def ingest_and_index_new_document(file_path: str):
    """Process only the new document and append to existing index."""
    print(f"🚀 Processing new document: {file_path}")

    # 1. Ingest text
    ingested = ingest_document(file_path)
    print(f"✅ Text extracted ({ingested['metadata']['total_characters']} characters)")

    # 2. Chunk only the new document
    new_chunks = create_chunks(ingested["text"], ingested["metadata"])
    print(f"✅ Created {len(new_chunks)} new chunks")

    # 3. Generate embeddings for new chunks only
    new_chunks = generate_embeddings_for_chunks(new_chunks)

    # 4. Load existing chunks if any
    chunks_path = "data/chunks/all_chunks.json"
    if os.path.exists(chunks_path):
        existing_chunks, _ = load_chunks_with_embeddings(chunks_path)
        all_chunks = existing_chunks + new_chunks
        print(f"✅ Appended to existing {len(existing_chunks)} chunks")
    else:
        all_chunks = new_chunks

    # 5. Save combined chunks
    save_chunks_with_embeddings(all_chunks, "all_chunks.json")

    # 6. Rebuild indexes (still full rebuild for reliability - we can optimize further later)
    _, embeddings = load_chunks_with_embeddings(chunks_path)
    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index, "faiss_index.bin")

    bm25_index, _ = build_bm25_index(chunks_path)

    print(f"✅ Successfully added {len(new_chunks)} chunks. Total chunks now: {len(all_chunks)}")
    return len(new_chunks)