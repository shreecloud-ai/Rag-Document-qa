# test_embeddings.py - Test full ingestion → chunking → embeddings pipeline
from src.ingestion import ingest_document
from src.chunking import create_chunks
from src.embeddings import generate_embeddings_for_chunks
import json
import os

print("🚀 Starting full embedding test...\n")

# 1. Ingest document
doc_path = "data/documents/sample_rag.txt"
print("Step 1: Ingesting document...")
ingested = ingest_document(doc_path)
print(f"✅ Loaded: {ingested['metadata']['filename']} ({ingested['metadata']['total_characters']} chars)\n")

# 2. Create chunks
print("Step 2: Creating chunks...")
chunks = create_chunks(ingested['text'], ingested['metadata'])
print(f"✅ Created {len(chunks)} chunks\n")

# 3. Generate embeddings
print("Step 3: Generating embeddings...")
chunks_with_embeddings = generate_embeddings_for_chunks(chunks)

# 4. Show sample embedding
if chunks_with_embeddings:
    sample_chunk = chunks_with_embeddings[0]
    embedding = sample_chunk["embedding"]
    
    print("\n✅ Sample Embedding (first 10 values):")
    print(embedding[:10])
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding type: {type(embedding[0])}")

# 5. Save everything to JSON (with embeddings)
output_path = "data/chunks/sample_chunks_with_embeddings.json"
os.makedirs("data/chunks", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)

print(f"\n✅ All chunks with embeddings saved to: {output_path}")
print(f"Total chunks processed: {len(chunks_with_embeddings)}")