# 🔍 RAG Document Q&A System

> A production-ready Retrieval-Augmented Generation pipeline for intelligent document question-answering — featuring hybrid retrieval, confidence scoring, and a human review queue.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-v1.0_Working-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Folder Structure](#-folder-structure)
- [Setup Instructions](#-setup-instructions)
- [How to Run](#-how-to-run)
- [How to Use](#-how-to-use)
- [Limitations & Future Improvements](#-limitations--future-improvements)
- [What I Learned](#-what-i-learned)

---

## 📖 Project Description

The **RAG Document Q&A System** is a fully functional, end-to-end Retrieval-Augmented Generation application that enables users to upload documents and ask natural language questions against them. It combines the complementary strengths of **dense vector search** (Sentence-BERT + FAISS) and **sparse keyword search** (BM25) into a hybrid retrieval pipeline, then applies a **cross-encoder reranker** to select the most relevant chunks before generating an answer.

The system includes a **confidence scoring mechanism** that flags uncertain responses into a **human review queue**, ensuring reliability in use cases where accuracy is critical — such as legal documents, research papers, and internal knowledge bases.

This is **Version 1.0** with a mock LLM layer, making it immediately runnable without any paid API keys or GPU infrastructure, while the architecture is fully designed to plug in a real LLM (OpenAI, Mistral, or a local model) with minimal changes.

---

## ✨ Features

- **Multi-format Document Upload** — supports PDF, TXT, and DOCX files
- **Hybrid Retrieval** — combines BM25 (keyword) and Sentence-BERT + FAISS (semantic) search for best-of-both-worlds recall
- **Cross-Encoder Reranking** — a fine-tuned cross-encoder model re-scores candidate chunks for precision before generation
- **Confidence Scoring** — each answer is assigned a confidence score; low-confidence answers are automatically flagged
- **Human Review Queue** — flagged answers are routed to a review queue for manual validation
- **Source Citations** — every answer surfaces the exact document chunk(s) it was derived from
- **Streamlit UI** — clean, interactive frontend for uploading documents and querying
- **FastAPI Backend** — RESTful API with proper request/response schemas
- **Mock LLM Layer** — runs fully without external API keys; swap in any real LLM with one config change
- **Chunk-level Indexing** — documents are split into overlapping chunks for granular retrieval

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive UI for upload and Q&A |
| **Backend** | FastAPI | REST API, routing, business logic |
| **Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) | Dense vector representations of text |
| **Vector Store** | FAISS | Approximate nearest-neighbour search |
| **Sparse Search** | BM25 (rank-bm25) | Keyword-based lexical retrieval |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Precision reranking of retrieved chunks |
| **LLM** | Mock LLM (v1.0) / OpenAI-compatible | Answer generation |
| **Document Parsing** | PyMuPDF, python-docx, built-in | PDF, DOCX, TXT ingestion |
| **Language** | Python 3.10+ | Core runtime |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                       │
│              Upload Docs  │  Ask Questions  │  Review Queue     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (REST)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                          │
│                                                                 │
│   ┌──────────────┐    ┌──────────────────────────────────────┐  │
│   │   /upload    │    │              /query                  │  │
│   │              │    │                                      │  │
│   │ Parse Doc    │    │  ┌─────────────┐  ┌──────────────┐  │  │
│   │ Chunk Text   │    │  │  BM25       │  │  FAISS       │  │  │
│   │ Embed Chunks │    │  │  Retriever  │  │  Retriever   │  │  │
│   │ Index FAISS  │    │  │ (Sparse)    │  │ (Dense)      │  │  │
│   │ Index BM25   │    │  └──────┬──────┘  └──────┬───────┘  │  │
│   └──────────────┘    │         │                │          │  │
│                        │         └────────┬───────┘          │  │
│                        │                  ▼                   │  │
│                        │      ┌───────────────────────┐      │  │
│                        │      │   Hybrid Fusion       │      │  │
│                        │      │  (Reciprocal Rank)    │      │  │
│                        │      └───────────┬───────────┘      │  │
│                        │                  ▼                   │  │
│                        │      ┌───────────────────────┐      │  │
│                        │      │   Cross-Encoder       │      │  │
│                        │      │   Reranker            │      │  │
│                        │      └───────────┬───────────┘      │  │
│                        │                  ▼                   │  │
│                        │      ┌───────────────────────┐      │  │
│                        │      │   LLM Answer Gen      │      │  │
│                        │      │   + Confidence Score  │      │  │
│                        │      └───────────┬───────────┘      │  │
│                        │                  ▼                   │  │
│                        │      ┌───────────────────────┐      │  │
│                        │      │ Confidence < Threshold│      │  │
│                        │      │   → Human Review Queue│      │  │
│                        │      └───────────────────────┘      │  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow Summary:**

1. **Ingestion** — A document is uploaded, parsed by format-specific parsers, split into overlapping text chunks, embedded using Sentence-BERT, indexed in FAISS (dense), and also indexed in a BM25 structure (sparse).
2. **Query** — A user question triggers both BM25 and FAISS retrieval in parallel. Results are fused using Reciprocal Rank Fusion (RRF).
3. **Reranking** — The fused candidate chunks are passed through a cross-encoder reranker for precise relevance scoring.
4. **Generation** — The top-ranked chunks are assembled into a context prompt and sent to the LLM. The response is returned with source citations.
5. **Confidence Check** — If the answer confidence falls below a configurable threshold, it is automatically added to the human review queue.

---

## 📁 Folder Structure

```
rag-document-qa/
│
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── routers/
│   │   ├── upload.py            # Document upload endpoint
│   │   ├── query.py             # Q&A query endpoint
│   │   └── review.py           # Human review queue endpoints
│   ├── services/
│   │   ├── ingestion.py         # Document parsing and chunking
│   │   ├── embedder.py          # Sentence-BERT embedding logic
│   │   ├── retriever.py         # BM25 + FAISS hybrid retrieval
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── llm.py               # LLM interface (mock + real)
│   │   └── confidence.py        # Confidence scoring logic
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── storage/
│   │   ├── faiss_index/         # Persisted FAISS indices
│   │   ├── bm25_index/          # Serialized BM25 indices
│   │   └── review_queue.json    # Persisted human review items
│   └── utils/
│       └── helpers.py           # Shared utility functions
│
├── frontend/
│   └── app.py                   # Streamlit application
│
├── docs/
│   └── architecture.png         # Architecture diagram (optional)
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   └── test_api.py
│
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- `pip` or `conda`
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Frontend
streamlit==1.28.0

# NLP & Embeddings
sentence-transformers==2.2.2
faiss-cpu==1.7.4
rank-bm25==0.2.2
transformers==4.35.0
torch==2.1.0

# Document Parsing
pymupdf==1.23.6
python-docx==1.1.0

# Data Validation
pydantic==2.4.2
pydantic-settings==2.0.3

# Utilities
python-dotenv==1.0.0
httpx==0.25.0
numpy==1.24.4

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.0
```

### 4. Configure Environment Variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

**`.env.example`**

```env
# Application Settings
APP_ENV=development
LOG_LEVEL=INFO

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend
FRONTEND_PORT=8501
BACKEND_URL=http://localhost:8000

# Retrieval Settings
TOP_K_BM25=10
TOP_K_FAISS=10
TOP_K_RERANKED=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Confidence Scoring
CONFIDENCE_THRESHOLD=0.65

# LLM Settings (set to "mock" to use mock LLM)
LLM_PROVIDER=mock
# OPENAI_API_KEY=your-key-here       # Uncomment if using OpenAI
# OPENAI_MODEL=gpt-3.5-turbo        # Uncomment if using OpenAI

# Storage
FAISS_INDEX_DIR=backend/storage/faiss_index
BM25_INDEX_DIR=backend/storage/bm25_index
REVIEW_QUEUE_PATH=backend/storage/review_queue.json
```

---

## 🚀 How to Run

### Start the FastAPI Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`
Interactive API docs (Swagger UI): `http://localhost:8000/docs`

### Start the Streamlit Frontend

Open a new terminal (keep the backend running):

```bash
cd frontend
streamlit run app.py --server.port 8501
```

The UI will open automatically at: `http://localhost:8501`

### Run Tests

```bash
pytest tests/ -v
```

---

## 📋 How to Use

### Step 1 — Upload a Document

1. Open the Streamlit UI at `http://localhost:8501`
2. Navigate to the **"Upload Document"** tab
3. Click **"Browse files"** and select a `.pdf`, `.txt`, or `.docx` file
4. Click **"Upload & Index"** — the system will parse, chunk, embed, and index your document
5. A success message will confirm the number of chunks indexed

### Step 2 — Ask a Question

1. Navigate to the **"Ask a Question"** tab
2. Type your question in the text box (e.g., *"What are the key findings in section 3?"*)
3. Click **"Get Answer"**
4. The system returns:
   - **Answer** — the generated response
   - **Confidence Score** — a 0–1 score indicating answer reliability
   - **Source Citations** — the exact document chunks used to generate the answer

### Step 3 — Human Review Queue

1. Navigate to the **"Review Queue"** tab
2. View all answers that were flagged due to low confidence scores
3. Each item shows the original question, the generated answer, the confidence score, and the source chunks
4. Mark items as **Approved** or **Rejected** to clear them from the queue

### API Usage (Direct)

You can also call the backend API directly:

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -F "file=@your_document.pdf"

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion?"}'

# View review queue
curl -X GET "http://localhost:8000/review"
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

- **Mock LLM** — Version 1.0 uses a rule-based mock LLM. Answers are coherent but not truly generated; this is intentional for zero-dependency local development.
- **Single-document Context** — The retrieval pipeline does not yet support cross-document queries across multiple simultaneously indexed documents.
- **No Authentication** — The API has no auth layer; all endpoints are publicly accessible.
- **In-memory Indices** — FAISS and BM25 indices are rebuilt on restart if persistence is not configured; large document sets may see cold-start delays.
- **No Streaming** — Answers are returned as a single response; streaming tokens is not yet supported.

### Planned Future Improvements

#### 🔌 LLM Integration
- Plug in OpenAI GPT-4o, Mistral, or a local model (Ollama) by switching `LLM_PROVIDER` in `.env`
- Add streaming token output for real-time answer display

#### 📊 Evaluation & Metrics
- Integrate **RAGAS** for automated RAG evaluation: faithfulness, answer relevancy, context precision, and context recall
- Add a metrics dashboard to track retrieval quality over time
- Implement **MRR (Mean Reciprocal Rank)** and **NDCG** for retrieval benchmarking

#### 🐳 Containerization & Deployment
- Add a `Dockerfile` for each service (backend and frontend)
- Create a `docker-compose.yml` for single-command deployment:
  ```yaml
  services:
    backend:
      build: ./backend
      ports: ["8000:8000"]
    frontend:
      build: ./frontend
      ports: ["8501:8501"]
  ```
- Deploy to AWS ECS, Google Cloud Run, or Railway

#### 🧠 Technical Enhancements
- **Multi-document support** with per-document metadata filtering
- **Persistent vector store** migration to Qdrant or Chroma for production scale
- **Query rewriting** using an LLM to rephrase ambiguous questions before retrieval
- **Conversation memory** for multi-turn Q&A sessions
- **Re-ingestion pipeline** for updating documents without full re-indexing
- **OCR support** for scanned PDFs using Tesseract

---

## 🎓 What I Learned

Building this project provided hands-on experience across the full RAG stack. Here are the key takeaways:

**Retrieval Engineering**
- Dense-only retrieval (FAISS) works well for semantically similar queries but struggles with exact keyword matches. Hybrid retrieval with BM25 significantly improves recall for domain-specific terminology and proper nouns.
- Reciprocal Rank Fusion (RRF) is a simple but effective algorithm for merging ranked lists from heterogeneous retrieval systems without needing to tune score normalization.

**Reranking**
- Bi-encoder models (Sentence-BERT) are fast but approximate; cross-encoders are slower but dramatically more precise. The two-stage approach (bi-encoder retrieve → cross-encoder rerank) is the industry-standard pattern for a reason.

**Chunking Strategy Matters**
- Chunk size and overlap have an outsized impact on retrieval quality. Chunks that are too large dilute relevance; chunks that are too small lose context. Overlapping chunks help preserve sentence continuity across boundaries.

**Confidence Scoring**
- Designing a meaningful confidence score is non-trivial. A naive approach (using reranker score alone) produced miscalibrated signals. Combining reranker score, BM25 score, and semantic similarity gives a more robust estimate.

**System Design**
- Separating the backend (FastAPI) from the frontend (Streamlit) via a REST API makes the system more testable, extensible, and deployment-friendly — even if it adds some initial complexity.
- Pydantic schemas enforced at the API boundary catch bugs early and serve as living documentation.

**Developer Experience**
- The mock LLM pattern was invaluable for building and testing the full pipeline without API costs or latency during development. Always build with a mockable interface.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 Author

Built by Pranav Jadhav — feel free to connect on [LinkedIn](https://linkedin.com/in/pranav-jadhav93) or explore more projects on [GitHub](https://github.com/shreecloud-ai).

---

*If you found this project useful, a ⭐ on GitHub is always appreciated!*
