# test_save_embeddings.py - Test saving embeddings cleanly
from src.ingestion import ingest_document
from src.chunking import create_chunks
from src.embeddings import generate_embeddings_for_chunks, save_chunks_with_embeddings

print("🚀 Testing clean embedding save...\n")

# 1. Ingest + Chunk + Embed
doc_path = "data/documents/sample.txt"
ingested = ingest_document(doc_path)
chunks = create_chunks(ingested['text'], ingested['metadata'])
chunks_with_emb = generate_embeddings_for_chunks(chunks)

# 2. Save
save_chunks_with_embeddings(chunks_with_emb, "sample_chunks_final.json")

print("\n✅ Embedding save test completed!")
print("You can now check the file in data/chunks/")