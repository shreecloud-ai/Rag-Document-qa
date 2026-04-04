# test_chunking.py - Test ingestion + chunking together
from src.ingestion import ingest_document
from src.chunking import create_chunks
import json
import os

# 1. Ingest the document
print("Step 1: Ingesting document...")
doc_path = "data/documents/sample_rag.txt"
ingested = ingest_document(doc_path)

print(f"✅ Loaded: {ingested['metadata']['filename']}")
print(f"Total characters: {ingested['metadata']['total_characters']}\n")

# 2. Create chunks
print("Step 2: Creating chunks...")
chunks = create_chunks(ingested['text'], ingested['metadata'])

print(f"✅ Created {len(chunks)} chunks\n")

# 3. Display the chunks nicely
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} (ID: {chunk['chunk_id']}) ---")
    print(f"Length: {chunk['metadata']['chunk_length']} characters")
    print(f"Text:\n{chunk['text']}")
    print("-" * 80 + "\n")

# 4. Save chunks as JSON for later use
output_path = "data/chunks/sample_chunks.json"
os.makedirs("data/chunks", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"✅ Chunks saved to: {output_path}")