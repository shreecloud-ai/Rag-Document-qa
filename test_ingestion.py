# test_ingestion.py - Quick test for our ingestion module
from src.ingestion import ingest_document
import os

# Make sure we're in the project root
print("Current working directory:", os.getcwd())

# Path to our sample document
doc_path = "data/documents/sample.txt"

# Ingest the document
result = ingest_document(doc_path)

print("\n✅ Ingestion Successful!")
print(f"Filename: {result['metadata']['filename']}")
print(f"File type: {result['metadata']['file_type']}")
print(f"Total characters: {result['metadata']['total_characters']}")
print(f"Ingested at: {result['metadata']['ingested_at']}")

print("\n--- First 300 characters of extracted text ---")
print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])