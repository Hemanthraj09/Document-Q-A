# Visual Document Q&A

📄 **Ask questions about your documents using AI** — A FastAPI-based RAG pipeline that extracts, embeds, and retrieves document content to answer questions with a local LLM, alongside a visual topic graph powered by D3.js.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-phi3-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)

---

## 🏗️ Architecture

![Architecture Diagram](DocQA-final.png)

The system is split into three parallel flows:

**Ingestion** — PDFs are parsed by PyMuPDF with tiered heading detection (H1/H2/H3 by font size), chunked at 1000 chars with 200-char overlap, embedded using `all-MiniLM-L6-v2`, and stored in ChromaDB.

**RAG Chat** — User questions are embedded with the same model, top-5 chunks are retrieved via cosine similarity, re-ranked with a topic-aware boost, and passed to Ollama (phi3) which streams the response back to the UI via SSE.

**Concept Analysis** — A 4-signal composite score (TF-IDF frequency + section spread + heading weight + PageRank centrality) drives a D3.js force-directed topic graph that renders instantly. A background LLM pass then refines the topics and re-renders the graph via SSE without blocking the user.

---

## ✨ Features

- **📤 PDF Upload** — PyMuPDF extraction with H1/H2/H3 heading detection
- **💬 Streaming Chat** — SSE-streamed responses from local Ollama phi3
- **🔍 Smart Retrieval** — Top-5 cosine similarity + query-aware topic re-ranking
- **🕸️ Topic Graph** — D3.js force-directed graph with tier-based coloring (Core / Supporting / Minor)
- **📊 Bar Chart** — Top 15 topics ranked by composite importance score
- **🔬 LLM Refinement** — Async background topic extraction merges named concepts into the graph
- **⛶ Fullscreen Graph** — Expand the topic graph to full screen for detailed exploration
- **📝 Text Input** — Paste raw text directly without needing a PDF

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) running locally with `phi3` pulled

```bash
ollama pull phi3
```

### 1. Clone & Install

```bash
git clone https://github.com/Hemanthraj09/Document-Q-A.git
cd Document-Q-A
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (optional):

```env
OLLAMA_URL=http://localhost:11434
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Run the Server

```bash
uvicorn main:app --reload
```

### 5. Open the UI

Navigate to **[http://localhost:8000](http://localhost:8000)**

---

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend UI |
| `/health` | GET | Health check (PDF / NLP / RAG status) |
| `/upload-pdf` | POST | PDF extraction + heading detection + RAG ingestion |
| `/ingest` | POST | Text ingestion into RAG pipeline |
| `/analyze` | POST | TF-IDF + Composite Importance Engine |
| `/analyze-llm` | POST | Async LLM topic extraction (SSE) |
| `/chat-stream` | POST | RAG retrieval + Ollama streaming (SSE) |
| `/chat` | POST | Non-streaming fallback (AI Summary) |

---

## 🧠 Concept Importance Score

Each topic is scored using a 4-signal composite:

```
importance = 0.30 × frequency     (normalized TF-IDF)
           + 0.25 × spread        (sections containing term / total sections)
           + 0.20 × heading_weight (H1=1.0, H2=0.67, H3=0.5, body=0)
           + 0.25 × centrality    (PageRank on co-occurrence graph)
```

| Tier | Score | Color |
|------|-------|-------|
| Core | > 0.7 | 🔵 Blue |
| Supporting | 0.4 – 0.7 | 🟡 Yellow |
| Minor | < 0.4 | ⚪ Grey |

---

## 📁 Project Structure

```
Document-Q-A/
├── main.py           # FastAPI backend — all endpoints and pipeline logic
├── index.html        # Frontend — chat UI, D3.js graph, SSE handling
├── requirements.txt  # Python dependencies
├── .env.example      # Environment template
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, D3.js |
| PDF Processing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (in-memory, cosine similarity) |
| LLM | Ollama — phi3:latest (local, ~12 tok/s on RTX 4050) |
| Topic Analysis | scikit-learn TF-IDF + NetworkX PageRank |
| Async HTTP | httpx |

---

## ⚙️ Hardware

Tested on NVIDIA RTX 4050 Laptop (6GB VRAM). phi3 (2.2GB) fits entirely on GPU at ~12.4 tok/s. The embedding model runs on CPU to avoid VRAM contention.

---

## 📝 License

MIT License

## 👤 Author

**Hemanth Raj** — [GitHub](https://github.com/Hemanthraj09)
