# Visual Document Q&A — End-to-End RAG Pipeline

> **Last updated:** 2026-05-01  
> **Version:** 4.1.0 — Concept Importance Engine + LLM Topic Extraction

## Architecture Overview

```
┌──────────────┐     ┌─────────────────────────────────────────────────────┐
│   Frontend   │     │              FastAPI Backend                        │
│  index.html  │────▶│                                                     │
│              │     │  ┌──────────┐  ┌───────────┐  ┌─────────────┐      │
│  • Upload    │     │  │ PyMuPDF  │  │ Sentence  │  │  ChromaDB   │      │
│  • Chat      │     │  │ Extract  │─▶│ Transform │─▶│ Vector      │      │
│  • D3.js     │     │  │ + Heads  │  │ Embed     │  │ Store       │      │
│    Graphs    │     │  └──────────┘  └───────────┘  └──────┬──────┘      │
│              │◀─SSE│                                       │             │
│  Streaming   │     │  ┌──────────┐  ┌───────────┐  ┌──────▼──────┐      │
│  Tokens      │     │  │ Ollama   │◀─│ Augmented │◀─│ Retriever   │      │
│              │     │  │ phi3     │  │ Prompt    │  │ Top-5       │      │
│              │     │  │ (local)  │  │ + Lexo    │  │ + Topic     │      │
│              │     │  └────┬─────┘  └───────────┘  │   Boost     │      │
│              │     │       │                        └─────────────┘      │
│              │     │  ┌────▼──────────────────────────────────────┐      │
│              │◀─SSE│  │  Concept Importance Engine                │      │
│  Graph       │     │  │  TF-IDF → Spread → Heading → PageRank   │      │
│  Update      │     │  │  → Composite Score → Async LLM Refine   │      │
│              │     │  └──────────────────────────────────────────┘      │
└──────────────┘     └─────────────────────────────────────────────────────┘
```

---

## 1. Document Ingestion

### PDF Upload (`POST /upload-pdf`)
1. **Text Extraction** — PyMuPDF extracts text from each page
2. **Tiered Heading Detection** — Font-size analysis relative to body text mode:
   - H1: font > 1.6× body size → 3× weight
   - H2: font > 1.3× body size → 2× weight
   - H3: font > 1.15× body size → 1.5× weight
3. **Chunking** — 1000-char chunks, 200-char overlap, sentence-boundary aware
4. **Embedding** — `all-MiniLM-L6-v2` (384-dim, CPU)
5. **Storage** — ChromaDB in-memory, cosine similarity
6. **Returns** — `doc_id`, `chunks_count`, `headings`, `heading_levels`, full text

### Text Input (`POST /ingest`)
Same pipeline, skips PDF extraction.

---

## 2. RAG Chat (`POST /chat-stream`)

### Retrieval
1. Question embedded with same model
2. Top-5 chunks from ChromaDB (cosine distance)
3. **Score normalization:** `similarity = 1 - (distance / 2)` → [0, 1]
4. **Query-aware topic boost:** +0.05–0.15 for chunks containing top TF-IDF topic terms from the user's question

### Generation
5. Prompt: Lexo system message → conversation history (last 10) → retrieved chunks + question
6. Streamed via SSE from Ollama (`/api/chat`, `stream: true`)
7. ~12 tok/s on RTX 4050

---

## 3. Concept Importance Engine (`POST /analyze`)

### 4-Layer Text Cleaning (before analysis)
| Layer | What It Removes |
|---|---|
| **1. Repeated Line Stripping** | Lines appearing on >30% of pages (headers, footers, running titles) |
| **2. Metadata Regex** | Page numbers, dates, emails, URLs, DOIs, figure/table labels |
| **3. Static Stopwords** | 60+ academic/structural terms: figure, table, chapter, appendix, bibliography... |
| **3b. Dynamic Stopwords** | Top-3 frequent words (4+ letters, 3+ occurrences) from first+last page |
| **4. TF-IDF Frequency Bounds** | `min_df=2, max_df=0.7` for 8+ sections; looser for smaller documents |

### Smart Sectioning
- **Primary:** Split by detected headings (uses heading positions from font-size analysis)
- **Fallback:** If <3 heading-based sections, split into 1500-char fixed chunks
- Guarantees TF-IDF always gets a multi-document corpus

### Composite Importance Score
```
importance = 0.30 × frequency     (normalized TF-IDF)
           + 0.25 × spread        (sections_containing / total_sections)
           + 0.20 × heading_weight (0=body, 0.5=H3, 0.67=H2, 1.0=H1)
           + 0.25 × centrality    (PageRank on co-occurrence graph)
```

### Topic Tiers
| Tier | Score Range | Node Color | Meaning |
|---|---|---|---|
| **Core** | > 0.7 | 🔵 Blue | Central concepts, must-know |
| **Supporting** | 0.4 – 0.7 | 🟡 Yellow | Important context |
| **Minor** | < 0.4 | ⚪ Grey | Peripheral mentions |

### Study Priority
`"Top 5 topics cover ~X% of this document"` — computed as `sum(top5_importance) / sum(all_importance) × 100`

---

## 4. Async LLM Topic Extraction (`POST /analyze-llm`)

**Runs in background after TF-IDF graph is already displayed.**

### Flow
1. Frontend renders TF-IDF graph instantly → badge shows `"20 topics ✨ refining..."`
2. Frontend fires background POST to `/analyze-llm` with document text + existing topics
3. Backend sends first 8K chars to Ollama with domain-adaptive prompt (temperature=0.3)
4. Prompt instructs: "Return ONLY specific named concepts, not generic vocabulary"
5. Response parsed with 2-layer JSON extractor (direct parse → regex fallback)
6. LLM topics merged with existing TF-IDF topics:
   - **Existing match (case-insensitive):** 1.5× score boost + tier upgrade
   - **Partial match (substring):** skipped (already covered)
   - **New topic:** added at score 0.55 (supporting tier)
7. Merged topics (max 25) sent as SSE event → frontend re-renders graph in-place
8. Badge updates to `"25 topics ✨"` — sparkle indicates LLM-refined

### Fault Tolerance
- Malformed JSON → silent fallback to TF-IDF
- Ollama timeout/disconnect → silent fallback
- Frontend badge reverts if LLM fails — user never sees an error

---

## 5. Visual Insights (Frontend)

### D3.js Force-Directed Graph
- **Node size** → composite importance score
- **Node color** → tier (blue/yellow/grey)
- **Edge thickness** → co-occurrence weight
- **Glow effect** → core concepts only
- **Label color** → matches tier
- **Legend** → `● Core  ● Support  ● Minor`
- **Tooltips** → importance %, spread %, centrality %, tier
- **Click** → fills chat input with "What does the document say about X?"

### Animated Bar Chart
- Top 15 topics, horizontal bars, ranked by importance

### AI Summary
- On-demand bullet-point summary via `/chat` endpoint

---

## 6. Data Flow Diagram

```
User uploads PDF
  │
  ▼
PyMuPDF: Extract text + detect tiered headings (H1/H2/H3)
  │
  ├──▶ 4-Layer Cleaning ──▶ Smart Sectioning ──▶ TF-IDF
  │                                                 │
  │                                    ┌────────────┤
  │                                    ▼            ▼
  │                              Spread Score   PageRank
  │                                    │            │
  │                                    ▼            ▼
  │                            Composite Importance Score
  │                              (freq + spread + heading + centrality)
  │                                    │
  │                                    ├──▶ Frontend Graph (instant)
  │                                    │
  │                                    ▼
  │                            Async LLM Extraction
  │                              (phi3, background SSE)
  │                                    │
  │                                    ▼
  │                            Merge + Re-render Graph (10-15s later)
  │
  ▼
Chunk (1000c/200 overlap) ──▶ Embed (MiniLM) ──▶ ChromaDB Store
                                                        │
User asks question ──▶ Embed ──────────────────────────▶│
                                                        ▼
                                                   Retrieve top-5
                                              + topic-aware re-ranking
                                                        │
                                              Lexo prompt + history
                                                        │
                                              Ollama phi3 (stream SSE)
                                                        │
                                                  User sees answer
```

---

## 7. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serve frontend UI |
| `/health` | GET | Health check (PDF/NLP/RAG status) |
| `/upload-pdf` | POST | PDF extraction + heading detection + RAG ingestion |
| `/ingest` | POST | Text ingestion into RAG pipeline |
| `/analyze` | POST | TF-IDF + Composite Importance Engine |
| `/analyze-llm` | POST | Async LLM topic extraction (SSE) |
| `/chat-stream` | POST | RAG retrieval + Ollama streaming (SSE) |
| `/chat` | POST | Non-streaming fallback (AI Summary) |

---

## 8. Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server |
| `pymupdf` (fitz) | PDF extraction + font-size analysis |
| `sentence-transformers` | Embeddings (all-MiniLM-L6-v2) |
| `chromadb` | Vector storage + similarity search |
| `scikit-learn` | TF-IDF topic analysis |
| `networkx` | PageRank centrality computation |
| `httpx` | Async HTTP for Ollama API |
| `python-dotenv` | Environment variables |
| **Ollama** (external) | Local LLM — phi3:latest |

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Composite score over pure TF-IDF** | 4-signal blend prevents generic words from dominating |
| **Heading-based sectioning** | Semantically meaningful sections > arbitrary page splits |
| **1500-char fallback chunks** | Guarantees multi-doc corpus for TF-IDF when headings are absent |
| **Async LLM as enhancement** | TF-IDF graph appears in <1s; LLM refinement is optional upgrade |
| **1.5× boost for LLM matches** | LLM validates domain relevance; existing TF-IDF terms deserve promotion |
| **25 topic cap after merge** | Prevents graph overcrowding while allowing LLM additions |
| **temperature=0.3** | Lower creativity = more consistent JSON output from phi3 |
| **2-layer JSON parser** | phi3 sometimes wraps JSON in markdown; regex fallback catches it |

---

## 10. Hardware Constraints

- **GPU:** NVIDIA RTX 4050 Laptop — 4GB VRAM
- **phi3:** 2.2GB, fits 100% on GPU, ~12.4 tok/s
- **Embedding model:** runs on CPU (no GPU contention)
- **llama3.1 (8B):** does NOT fit — crashes with OOM
