# PDF-Based RAG Application (Archle Labs Assignment)

A production-ready PDF-based Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **LangChain**, and **ChromaDB**. The system ingests PDF documents, splits them into semantic chunks, generates vector embeddings, stores them in a persistent vector database, executes hybrid retrieval with reranking, and synthesizes grounded answers with exact source citations (page numbers and chunk references) exposed via a clean **FastAPI endpoint (`POST /ask`)**.

The application includes a pre-indexed survey paper on Retrieval-Augmented Generation:  
**"Retrieval-Augmented Generation for Large Language Models: A Survey" (`Document (4).pdf` / `sample_document.pdf`)**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack Used](#tech-stack-used)
3. [Setup Instructions](#setup-instructions)
4. [Steps to Run the FastAPI Server](#steps-to-run-the-fastapi-server)
5. [API Endpoint Details](#api-endpoint-details)
6. [Example Questions and Responses](#example-questions-and-responses)
7. [Model Choices and Reasons](#model-choices-and-reasons)
8. [Optional & Bonus Features](#optional--bonus-features)
9. [Limitations of the Project](#limitations-of-the-project)

---

## 1. Project Overview

This project satisfies all requirements for the **Archle Labs PDF-based RAG Application** assignment:
- **PDF Loading & Text Extraction**: Extracts text and per-page metadata from uploaded or bundled PDF documents using `pypdf` and `pymupdf`.
- **Text Chunking**: Implements recursive character splitting with configurable chunk size (500 tokens) and overlap (50 tokens) to preserve semantic continuity across boundaries.
- **Embedding Generation**: Uses local sentence-transformer models (`all-MiniLM-L6-v2`) or OpenAI embeddings to convert text into high-dimensional semantic vectors.
- **Vector Database Storage**: Persists embeddings and metadata into **ChromaDB** with cosine similarity index.
- **Hybrid Retrieval & Reranking**: Executes dense vector similarity search combined with sparse lexical scoring and citation-filtering to eliminate bibliography noise and prioritize core conceptual text.
- **LangGraph Workflow Orchestration**: Employs a compiled **LangGraph `StateGraph`** with conditional branching:
  1. `check_cache`: Sub-millisecond lookup in SQLite persistent disk cache.
  2. `retrieve`: Multi-candidate vector retrieval from ChromaDB.
  3. `rerank`: Hybrid search reranking (dense score + keyword overlap + definition boost - bibliography penalty).
  4. `grade_and_format`: Prompt grading and strict context-grounded prompt compilation.
  5. `generate`: LLM answer synthesis.
  6. `cache_result`: Response persistence.
- **FastAPI Endpoints**: Exposes the required `POST /ask` endpoint and includes an interactive web dashboard for live queries, streaming responses, and document uploads.
- **Source References**: Every response contains exact page numbers, chunk identifiers, excerpts, and similarity scores.

---

## 2. Tech Stack Used

| Component | Technology | Description |
|---|---|---|
| **API Framework** | FastAPI (v0.115+) | High-performance, asynchronous web framework with OpenAPI / Swagger documentation |
| **ASGI Server** | Uvicorn | Lightning-fast ASGI web server implementation |
| **Pipeline & State Graph** | LangGraph & LangChain | Stateful graph orchestrator with conditional edges, state checkpointing, and modular nodes |
| **Vector Database** | ChromaDB (v0.5+) | Persistent embedded vector database utilizing HNSW cosine distance indexing |
| **Embedding Model** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Compact, highly effective 384-dimensional embedding model running locally on CPU/GPU |
| **PDF Extraction** | PyPDF / PyMuPDF | Robust extraction of text streams and page number mapping |
| **Caching Layer** | SQLiteDiskCache | Thread-safe, persistent key-value cache with TTL expiration and hit/miss analytics |
| **Validation & Config** | Pydantic v2 & Pydantic-Settings | Strongly-typed schema validation and `.env` configuration management |
| **Testing** | Pytest & Starlette TestClient | Automated test suite covering unit, integration, and end-to-end RAG workflows |

---

## 3. Setup Instructions

### Prerequisites
- Python 3.10 to 3.14 installed on your system.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ask-your-doc
```

### Step 2: Create and Activate Virtual Environment
```bash
# On Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration (Optional)
The application works **completely out of the box with zero external API keys** using its built-in local extractive engine and local sentence-transformers embeddings.

If you would like to use an external LLM (e.g. OpenAI, Gemini, Groq, or Ollama), copy `.env.example` to `.env`:
```bash
cp .env.example .env    # On Linux/macOS
copy .env.example .env  # On Windows
```

Configure any provider of your choice in `.env`:
```env
# Choose provider: mock, gemini, openai, groq, ollama
LLM_PROVIDER=mock

# If using Gemini:
# GEMINI_API_KEY=your_gemini_api_key_here
# GEMINI_MODEL_NAME=gemini-1.5-flash

# If using OpenAI:
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL_NAME=gpt-4o-mini

# If using Groq:
# GROQ_API_KEY=your_groq_api_key_here
# GROQ_MODEL_NAME=llama-3.1-8b-instant
```

---

## 4. Steps to Run the FastAPI Server

### Option A: Using the Runner Script (Recommended)
```bash
python run.py
```

### Option B: Using Uvicorn CLI
```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

Once started:
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Interactive Web Demo Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 5. API Endpoint Details

### 1. `POST /ask` (Primary Assignment Endpoint)
Answers questions using retrieved context from the PDF document and provides exact page and chunk source citations.

- **URL**: `/ask`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Body
```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

#### Example Response (HTTP 200)
```json
{
  "answer": "Retrieval-Augmented Generation is a technique that combines information retrieval with text generation to enhance Large Language Models (LLMs) by incorporating knowledge from external databases. By retrieving relevant document chunks through semantic similarity calculation, RAG effectively reduces hallucinations, improves factual accuracy, and allows continuous knowledge updates without requiring full model retraining.",
  "sources": [
    {
      "page_number": 1,
      "chunk_id": "f4bbe73e74834bd6",
      "filename": "Document (4).pdf",
      "excerpt": "sive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising...",
      "similarity_score": 0.812
    },
    {
      "page_number": 1,
      "chunk_id": "099bd51cd9533e7a",
      "filename": "Document (4).pdf",
      "excerpt": "Retrieval-Augmented Generation for Large Language Models: A Survey Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia...",
      "similarity_score": 0.810
    }
  ]
}
```

#### Curl Example
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"What is Retrieval-Augmented Generation?\"}"
```

---

### Additional Complementary Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/documents/upload` | `POST` | Upload and dynamically index additional PDF documents into the vector database. |
| `GET /api/documents` | `GET` | List all currently indexed documents, page counts, and chunk totals. |
| `DELETE /api/documents/{doc_id}` | `DELETE` | Delete an indexed document and wipe its vector embeddings from ChromaDB. |
| `POST /api/query` | `POST` | Full synchronous RAG query endpoint with caching and execution time metadata. |
| `POST /api/query/stream` | `POST` | Server-Sent Events (SSE) streaming endpoint for live token emission. |
| `GET /api/cache/stats` | `GET` | Inspect SQLite cache hits, misses, hit rate percentage, and size. |
| `POST /api/cache/clear` | `POST` | Invalidate and purge the query answers cache. |
| `GET /api/health` | `GET` | Service health status, active LLM provider, vector database type, and document counts. |

---

## 6. Example Questions and Responses

All answers below are derived directly from `Document (4).pdf` ("Retrieval-Augmented Generation for Large Language Models: A Survey"):

### Example 1
**Question:**
```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```
**Response:**
```json
{
  "answer": "Retrieval-Augmented Generation is a technique that combines information retrieval with text generation to enhance Large Language Models (LLMs) by incorporating knowledge from external databases. By retrieving relevant document chunks through semantic similarity calculation, RAG effectively reduces hallucinations, improves factual accuracy, and allows continuous knowledge updates without requiring full model retraining.",
  "sources": [
    {
      "page_number": 1,
      "chunk_id": "f4bbe73e74834bd6",
      "filename": "Document (4).pdf",
      "excerpt": "Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases...",
      "similarity_score": 0.812
    }
  ]
}
```

### Example 2
**Question:**
```json
{
  "question": "What are the three main paradigms of RAG discussed in the paper?"
}
```
**Response:**
```json
{
  "answer": "The survey paper categorizes the progression of RAG into three distinct paradigms: Naive RAG, Advanced RAG, and Modular RAG.",
  "sources": [
    {
      "page_number": 1,
      "chunk_id": "f4bbe73e74834bd6",
      "filename": "Document (4).pdf",
      "excerpt": "This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG...",
      "similarity_score": 0.835
    }
  ]
}
```

### Example 3
**Question:**
```json
{
  "question": "What is the tripartite foundation of RAG frameworks?"
}
```
**Response:**
```json
{
  "answer": "The tripartite foundation of RAG frameworks comprises three core components: Retrieval, Generation, and Augmentation techniques.",
  "sources": [
    {
      "page_number": 1,
      "chunk_id": "f4bbe73e74834bd6",
      "filename": "Document (4).pdf",
      "excerpt": "It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the retrieval, the generation and the augmentation techniques...",
      "similarity_score": 0.840
    }
  ]
}
```

---

## 7. Model Choices and Reasons

1. **Embedding Model: `all-MiniLM-L6-v2` (Sentence-Transformers)**
   - **Reason**: `all-MiniLM-L6-v2` provides an optimal balance between execution speed, memory footprint (only 90 MB), and semantic search accuracy. It maps sentences into a 384-dimensional dense vector space and runs efficiently on standard CPU architectures without requiring dedicated GPU acceleration.
   - **Alternative**: Configurable to OpenAI `text-embedding-3-small` or `text-embedding-ada-002` via `.env`.

2. **Vector Store: ChromaDB**
   - **Reason**: ChromaDB is an open-source, embedded vector database with persistent on-disk storage using DuckDB/SQLite and HNSW indexing for nearest neighbor cosine similarity search. It requires no external server setup and offers low-latency retrieval.

3. **Orchestrator: LangGraph StateGraph**
   - **Reason**: LangGraph introduces cyclic and conditional graph workflows with deterministic state transitions. Unlike basic sequential chains, LangGraph allows conditional routing (such as checking the persistent cache before executing heavy vector retrieval and LLM generation), explicit node error recovery, and modular testing of individual pipeline components.

4. **Grounded Generation: Dual Engine (Extractive Grounded Engine + Cloud LLMs)**
   - **Default**: The built-in Local Grounded Extractive Engine synthesizes direct, factual responses directly from verified text chunks without hallucinating and with zero API cost.
   - **Pluggable Cloud LLMs**: The system provides hot-swappable integrations for OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), Groq (`llama-3.1-8b-instant`), and local Ollama (`llama3`).

---

## 8. Optional & Bonus Features

The project includes several optional features from the assignment specification for extra credit:

- ✅ **LangGraph Workflow**: Implemented as a stateful `StateGraph` with explicit nodes (`check_cache`, `retrieve`, `rerank`, `grade_and_format`, `generate`, `cache_result`).
- ✅ **Hybrid Search & Reranking**: Candidate chunks undergo hybrid scoring combining dense vector similarity with sparse lexical matching and bibliography de-noising filters.
- ✅ **Persistent Caching**: SQLite-backed caching with SHA-256 query fingerprinting, TTL expiration, and cache hit/miss statistics (`GET /api/cache/stats`).
- ✅ **Real-Time Streaming**: Server-Sent Events (SSE) streaming endpoint (`POST /api/query/stream`) emitting live tokens.
- ✅ **Interactive Web UI**: Modern single-page application (`templates/index.html` and `static/js/app.js`) designed for recording live video demos.

---

## 9. Limitations of the Project

1. **Scanned Images & Complex Non-Text PDFs**:
   - The current PDF loader is optimized for text-based PDFs. Scanned documents consisting solely of raster images without embedded OCR text layers require an additional OCR engine (such as Tesseract or AWS Textract).
2. **Context Window Boundary for Multi-Hop Inquiries**:
   - For queries requiring reasoning across 20+ disparate sections of a 100-page document simultaneously, top-k retrieval might select only the highest scoring chunks rather than the entire global context. Advanced summarization or tree-based retrieval (RAPTOR) could be used to address this.
3. **Table & Mathematical Formula Extraction**:
   - Complex nested tables and specialized LaTeX mathematical notation may be linearized into plain text during PDF extraction, which can occasionally reduce layout-specific spatial relationships.
4. **Local CPU Inference Latency**:
   - When generating dense vector embeddings for large documents (>100 pages) on low-spec CPUs, the initial one-time indexing phase can take 15–30 seconds. Subsequent queries benefit from persistent caching and pre-computed indexes.

---

## 10. Automated Tests

Run the complete test suite:
```bash
python -m pytest tests/
```

Expected output:
```
======================= 14 passed in 9.24s =======================
```
