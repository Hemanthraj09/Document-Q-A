# Visual Document Q&A — Pipeline Design

> **Last updated:** 2026-10-10
> **Version:** 5.0.0 — Evidence-backed concept graph, prerequisite learning path, mastery tracking

## Architecture Overview

```
┌──────────────────┐     ┌──────────────────────────────────────────────────────────┐
│   Frontend       │     │                   FastAPI Backend (main.py)              │
│   index.html     │     │                                                          │
│                  │────▶│  Ingestion ─ PyMuPDF ─ clean_pages ─ chunk_pages ─ MiniLM │
│  • Chat + cites  │     │                                         │                │
│  • Concept map   │     │                                     ChromaDB  data/docs  │
│  • Evidence      │◀SSE─│  Chat ─ dense top-20 + BM25 top-20 ─ RRF ─ top-5 ─ phi3   │
│    drawer        │     │                                                          │
│  • Learning path │◀────│  analysis.py ─ sections, sentences, composite score       │
│  • Quiz          │◀SSE─│  LLM refine ─ digest ─ phi3 JSON ─ grounding ─ re-rank    │
│                  │◀────│  learning.py ─ evidence, prerequisite DAG                 │
│                  │◀────│  mastery.py ─ EMA + SM-2, quiz targeting, grading         │
└──────────────────┘     └──────────────────────────────────────────────────────────┘
```

---

## 1. Ingestion (`POST /upload-pdf`, `POST /ingest`)

1. **Extraction** — PyMuPDF page text, plus every text line with its largest font size.
2. **Tiered headings** — relative to the modal (body) font size: H1 ≥ 1.6×, H2 ≥ 1.3×, H3 > 1.15×. Lines must contain a letter and be 4–149 characters.
3. **Cleaning** (`analysis.clean_pages`, one entry per page so page numbers stay valid):
   - Layer 1: lines repeated on ≥ 30% of pages (running headers/footers)
   - Layer 2: page numbers, dates, emails, URLs, DOIs, captions, short numeric labels, table-of-contents and index lines (dot leaders, "(page 3)", or pages where most lines end in a number)
   - Reflow: hard-wrapped lines are rejoined into sentences; hyphenated line breaks are mended
4. **Chunking** (`rag.chunk_pages`) — 1000 chars, 200 overlap, ends snapped to sentence/line boundaries, overlap starts on a sentence boundary when possible. Each chunk records `page_start` / `page_end`.
5. **Embedding** — `all-MiniLM-L6-v2` (384-dim) on CPU.
6. **Storage** — one ChromaDB collection per document (cosine); the document record (pages, headings, chunks) is saved to `data/docs/{doc_id}.json`. A SHA-256 content hash de-duplicates uploads, so re-uploading reopens the same document and its mastery. If the vector index is missing, it is rebuilt from the saved record.
7. **Limits** — 50 MB upload cap; encrypted, corrupt and image-only PDFs get specific error messages.

---

## 2. RAG Chat (`POST /chat-stream`)

1. **Small talk** — only whole-message greetings/thanks get a canned reply; "Define overfitting" is answered normally.
2. **Hybrid retrieval** — dense top-20 (Chroma) and BM25 top-20 (plural-folded tokens) fused with reciprocal rank fusion (k = 60); the best 5 go into the prompt. Fusion changes *which* chunks are used, not just their order.
3. **Context** — each excerpt is labelled `[p. 4]` / `[pp. 4–5]` (or `[Excerpt n]` for text); the system prompt asks Lexo to cite those labels.
4. **Budget** — `num_ctx = 4096`, `num_predict = 768`. History (last 10 messages, question + answer only) is trimmed so system + excerpts + history fit; otherwise Ollama would silently drop the start of the prompt, i.e. the system prompt.
5. **Streaming** — SSE events: `sources` first, then `token`s, then `done`. Ollama errors (model not pulled, server down, timeouts) are turned into user-readable messages.
6. **Exposure** — concepts named in the question are marked "explored" for mastery tracking.

---

## 3. Concept Importance Engine (`POST /analyze`)

### Sections and sentences
- Headings are located **on their own page** (a table of contents can't hijack the boundaries); tiny sections are merged; < 3 sections falls back to 1500-char windows.
- Sentences are indexed with page and section, so every claim in the UI can cite where it came from.

### Candidates
- TF-IDF (1–2 grams, 50 features, adaptive `min_df`/`max_df`) over sections nominates candidates.
- Stopwords: English + static academic/structural words + generic adjectives/verbs + **branding** (words on both the first and last page but nowhere in the body).
- Fragments are dropped: a term whose occurrences are ≥ 70% inside a longer candidate ("descent" in "gradient descent").

### Composite score (`analysis.measure_terms`)
Counts come from word-boundary regex matches (plural- and hyphen-tolerant, never across line breaks), so any term — including LLM suggestions — can be measured.

```
importance = (0.30 × frequency       mean TF-IDF over sections (smoothed idf, l2-normalised rows)
            + 0.25 × spread          sections containing term / total sections
            + 0.20 × heading_weight  max over matching headings: H1 3.0, H2 2.0, H3 1.5, ÷ 3.0
            + 0.25 × centrality)     PageRank on the section co-occurrence graph (edges ≥ 0.05)
            × multipliers            defined explicitly ×1.3 · near figure/table ×1.2 · worked examples ×1.15
                                     (each applied once) · LLM-endorsed ×2
```

Scores are normalised so the top concept is 1.0. Co-occurrence uses one matrix product (`Bᵀ·B` over a binary section×term matrix). Edges with weight ≥ 0.10 are drawn.

| Tier | Score | Colour |
|---|---|---|
| Core | > 0.55 | Blue |
| Supporting | 0.30 – 0.55 | Yellow |
| Minor | ≤ 0.30 | Grey |

**Study priority:** `sum(top-5 scores) / sum(all scores)`.

---

## 4. LLM Refinement (`POST /analyze-llm`, SSE)

1. The graph is already on screen; this runs in the background.
2. **Digest** — outline (H1/H2) plus an evenly-spaced excerpt from every section, within 8000 chars, so the model sees the whole document, not just its first pages.
3. **phi3 in JSON mode** (`format: "json"`, temperature 0.3) returns `{"topics": [...]}`; a regex fallback handles stray prose.
4. **Grounding** — a suggestion that never appears in the text is rejected (and shown to the user as rejected).
5. **Merge** — exact/plural matches endorse the existing concept; a shorter piece of an existing phrase endorses that phrase; new grounded concepts are added using the document's own wording. Everything is re-measured, endorsed concepts get ×2, and the top 25 are kept.
6. Results are cached per document; Ollama failures fall back silently to the TF-IDF graph.

---

## 5. Evidence and Learning Path (`learning.py`)

### Evidence drawer
- **Concept:** score breakdown (each signal × weight, multipliers), up to 2 definitional sentences, up to 4 mentions from different sections, mention/section/page counts, prerequisites and what it unlocks, each with the sentence that justifies it.
- **Link:** number of shared sentences and sections (the edge weight), the sentences that contain both, or paired sentences from shared sections, plus the prerequisite relation if one exists.

### Prerequisite DAG
A concept's **introduction** is its first definitional sentence (else first substantive mention, ≥ 6 words) plus the next two sentences in the same section. Edge *A → B*:

| Signal | Weight | Example |
|---|---|---|
| B's introduction uses A, and A was introduced earlier | +1 per sentence | "Gradient descent … minimizes a **loss function**" |
| A sentence states the dependency | +2 | "B is a type of A", "B relies on A", "A is used by B" |
| B is a more specific A (same head noun) | 1 | "neural network" → "network" |

Minor-tier concepts never gate others. Cycles: repeatedly remove the weakest edge in a cycle (ties: the edge that runs against document order). `transitive_reduction` keeps only direct prerequisites. Order: `lexicographical_topological_sort` with key *(−concepts unlocked, introduction position)*.

---

## 6. Mastery and Quizzes (`mastery.py`)

| Piece | Rule |
|---|---|
| Mastery | EMA: `m ← m + 0.5 × (score − m)`; mastered at ≥ 0.8 |
| Status | unseen → explored (asked about in chat) → learning → mastered |
| Schedule | SM-2: quality = round(score × 5); intervals 1, 6, then × ease; ease ≥ 1.3; failure resets to 1 day |
| Target | maximise `importance × (1 − mastery) × (1 if due else 0.25) × (0.35 if a prerequisite < 0.5 mastery else 1)` |
| Question | phi3 (JSON mode) from the concept's evidence + 3 retrieved chunks; fallback: fill-in-the-blank from a definition sentence |
| Grading | free text: `0.6 × LLM score + 0.4 × similarity`, similarity = clamp((cos − 0.25) / 0.5); cloze: exact (plural-insensitive) match, typo = 0.75; short "I don't know" = 0 |
| Storage | `data/mastery/{doc_id}.json`, atomic writes |

Questions are kept server-side until answered (the client never sees the reference answer early) and can be answered once.

---

## 7. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | UI |
| `/health` | GET | Status incl. Ollama reachability |
| `/upload-pdf` | POST | Ingest a PDF |
| `/ingest` | POST | Ingest text / .txt |
| `/analyze` | POST | Concepts, edges, learning path, mastery (`{doc_id}`) |
| `/analyze-llm` | POST | LLM refinement (SSE) |
| `/chat-stream` | POST | Hybrid RAG chat (SSE) |
| `/documents/{id}/concept` | GET | Concept evidence |
| `/documents/{id}/relation` | GET | Link evidence |
| `/documents/{id}/mastery` | GET | Mastery state |
| `/documents/{id}/quiz` | POST | Next question |
| `/documents/{id}/quiz/{quiz_id}/answer` | POST | Grade + update mastery |

---

## 8. What changed in 5.0 and why

| Problem in 4.x | Fix |
|---|---|
| "Re-ranking" reordered the same 5 chunks that all went to the LLM anyway | Hybrid dense + BM25 retrieval over 20 candidates each, fused with RRF |
| No citations; page markers were embedded as noise | Page-aware chunks, labelled excerpts, cited answers, sources list |
| SSE parser dropped tokens split across network reads | Buffered event parsing for every stream |
| `.txt` uploads were never indexed | `.txt` goes through `/ingest` |
| Greeting filter swallowed short questions ("Define X") | Only whole-message greetings/thanks |
| No `num_ctx`: overflow truncated the system prompt first | `num_ctx` / `num_predict` set; history budgeted |
| CPU-heavy work in `async` handlers froze the server | Threadpool for extraction, embedding, analysis |
| O(terms² × sections) co-occurrence with sparse element access | One matrix product |
| Exam boost compounded per section, then capped → ties at 1.0 | Each multiplier once; normalise instead of capping |
| Headings matched in the table of contents | Headings searched on their own page; TOC/index lines stripped |
| Branding stopwords deleted real topics on the first/last page | Branding must be absent from the body |
| TF-IDF fragments ("descent", "neural") ranked as concepts | Fragment filter |
| LLM saw only the first 8000 chars; hallucinated topics entered the graph | Whole-document digest; grounding check; LLM terms measured like any other |
| Substring matching ("art" in "partial") | Word-boundary, plural-aware term regex |
| State lost on restart; orphaned Chroma collections | Persisted document records, content-hash de-duplication, index rebuild |
| D3 and fonts from CDNs despite "fully local" | D3 served from `static/`, system fonts |
| Embeddings not actually pinned to CPU | `device="cpu"` (configurable) |
| CORS `*` with credentials, `0.0.0.0` bind, raw exception text | Localhost origins, `127.0.0.1`, user-safe error messages |
| No tests | pytest suite for analysis, learning path, mastery, retrieval |

---

## 9. Hardware

- **GPU:** NVIDIA RTX 4050 Laptop — 6 GB VRAM
- **phi3:** 2.2 GB, fits entirely on GPU, ~12 tok/s
- **Embedding model:** CPU (`EMBED_DEVICE=cpu`) to avoid VRAM contention
