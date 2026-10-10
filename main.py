"""
Visual Document Q&A - FastAPI backend.
Local RAG chat with citations, an evidence-backed concept graph, a prerequisite
learning path, and per-concept mastery tracking with quizzes.
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import httpx
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import analysis
import learning
import mastery
import rag

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docqa")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
DOCS_DIR = DATA_DIR / "docs"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "768"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]

CHARS_PER_TOKEN = 3.5  # conservative estimate for English prose
RETRIEVE_K = 5
CANDIDATE_K = 20
HISTORY_MESSAGES = 10
MAX_SESSIONS = 200
MAX_PENDING_QUIZZES = 500

app = FastAPI(title="Visual Document Q&A", version="5.0.0",
              description="Local RAG with an evidence-backed concept graph, learning path and mastery tracking")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

log.info("Loading embedding model all-MiniLM-L6-v2 on %s...", EMBED_DEVICE)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=EMBED_DEVICE)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
mastery_store = mastery.MasteryStore(DATA_DIR / "mastery")

documents: dict[str, dict] = {}
hash_index: dict[str, str] = {}
_analysis_cache: dict[str, dict] = {}
_analysis_locks: dict[str, threading.Lock] = {}
_bm25_cache: dict[str, rag.BM25] = {}
_ingest_lock = threading.Lock()
_locks_guard = threading.Lock()
sessions: OrderedDict[str, dict] = OrderedDict()
pending_quizzes: OrderedDict[str, dict] = OrderedDict()

LEXO_SYSTEM_PROMPT = """You are Lexo, a document analyst. You have one job: help the user understand their document deeply and clearly.
## How you answer
Start with a single direct sentence that answers the question.
Then support it with specific details from the document — not vague summaries, actual content.
Only add connections to other parts of the document if they genuinely add value. Don't force it.
Use **bold** for key terms when first introduced. No headers unless the answer has 3+ clearly separate sections
## Citations
Each document excerpt is labelled in square brackets, like {example}. After a sentence that uses an excerpt, cite it with that exact label in square brackets. Never invent labels.
## Tone
Clear, precise, and human. You can be warm, but don't perform enthusiasm.
No "Great question!", no "Certainly!", no "I'd be happy to".
If a user thanks you or says something conversational, respond naturally — skip the keyword footer for non-content messages.
## Grounding
You answer strictly from the document excerpts provided. If the answer isn't there, say so plainly:
"That's not covered in this document — but you could ask about [related topic that IS in the document]."
Do not speculate or fill gaps from general knowledge unless you explicitly flag it: "(this is general knowledge, not from the document)"
## Length
Match length to complexity. A simple factual question gets 2-3 sentences. A conceptual question gets a structured paragraph or two. Never pad.
## At the end of every content response, add:
📌 **Key terms:** [5–8 comma-separated keywords or short phrases — the core concepts needed to understand this answer]
Skip this footer for greetings, thanks, or conversational messages."""

LLM_TOPIC_PROMPT = """You are analyzing a document to extract its most important concepts.
Extract 15-20 core topics from the outline and excerpts below.
Rules:
- Return ONLY specific named concepts, not generic vocabulary
- Bad examples: "performance", "model", "service", "data", "system", "process", "approach", "method", "high", "large"
- Good examples: domain-specific named concepts, acronyms, proper technical terms, named frameworks, named algorithms, named theories, named events, named entities
- Use the exact wording the document uses, so each topic can be found in the text
- Each topic must be something worth knowing about this document specifically — if it could appear in any document on any topic, it is too generic
Return a JSON object of the form {{"topics": ["topic1", "topic2", ...]}} and nothing else.

Document:
---
{text}
---"""

QUIZ_PROMPT = """You are writing one quiz question to check whether a student understands the concept "{concept}" from their study material.
Use ONLY the context below.
Rules:
- Ask about meaning, purpose, or how it relates to something else in the context — not trivia like page numbers.
- The question must be answerable in 1-3 sentences from the context alone.
- Do not reveal the answer in the question.
Return a JSON object: {{"question": "...", "answer": "a concise model answer grounded in the context"}}

Context:
---
{context}
---"""

GRADE_PROMPT = """You are grading a student's answer against their study material.
Question: {question}
Reference answer: {reference}
Supporting context:
---
{context}
---
Student answer: {answer}

Judge factual correctness against the reference answer and the context. Accept different wording when the meaning is right. Give partial credit for partially correct answers. Ignore spelling and grammar.
Return a JSON object: {{"score": <number from 0 to 1>, "feedback": "<one or two sentences: what was right and what was missing>"}}"""


# ==========================
# === Request models
# ==========================

class DocRequest(BaseModel):
    doc_id: str


class IngestRequest(BaseModel):
    text: str
    filename: str | None = None


class ChatRequest(BaseModel):
    question: str
    doc_id: str
    session_id: str | None = None


class QuizRequest(BaseModel):
    concept: str | None = None


class AnswerRequest(BaseModel):
    answer: str


# ==========================
# === Document storage
# ==========================

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _load_documents() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for path in DOCS_DIR.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("Skipping unreadable document record %s", path.name)
            continue
        documents[doc["doc_id"]] = doc
        hash_index[doc["content_hash"]] = doc["doc_id"]
    log.info("Loaded %d saved documents", len(documents))


_load_documents()


def _get_doc(doc_id: str) -> dict:
    doc = documents.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload it again to continue.")
    return doc


def _index_chunks(doc: dict):
    texts = [c["text"] for c in doc["chunks"]]
    embeddings = embedder.encode(texts, batch_size=32).tolist()
    try:
        chroma_client.delete_collection(doc["doc_id"])
    except Exception:
        pass  # nothing to replace
    collection = chroma_client.create_collection(name=doc["doc_id"], metadata={"hnsw:space": "cosine"})
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(texts))],
        metadatas=[{"chunk_index": i, "page_start": c["page_start"], "page_end": c["page_end"]}
                   for i, c in enumerate(doc["chunks"])],
    )
    return collection


def _collection(doc: dict):
    try:
        return chroma_client.get_collection(doc["doc_id"])
    except Exception:
        log.warning("Vector index for %s missing; rebuilding from the saved document", doc["doc_id"])
        return _index_chunks(doc)


def _ingest(pages: list[str], headings: list[dict], kind: str, filename: str | None,
            content_hash: str) -> tuple[dict, bool]:
    """Index a document, or return the saved copy if this exact content was uploaded before
    (which also restores the learner's mastery progress for it)."""
    with _ingest_lock:
        existing = hash_index.get(content_hash)
        if existing in documents:
            doc = documents[existing]
            _collection(doc)
            return doc, True

        # Retrieval uses the same cleaned text as the analysis: no running headers, page numbers or TOC lines.
        chunks = rag.chunk_pages(analysis.clean_pages(pages))
        if not chunks:
            raise HTTPException(status_code=422, detail="There isn't enough text in this document to index.")
        doc = {
            "doc_id": f"doc_{uuid.uuid4().hex[:12]}",
            "filename": filename,
            "kind": kind,
            "content_hash": content_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
            "headings": headings,
            "chunks": chunks,
        }
        _index_chunks(doc)
        _write_json(DOCS_DIR / f"{doc['doc_id']}.json", doc)
        documents[doc["doc_id"]] = doc
        hash_index[content_hash] = doc["doc_id"]
        log.info("Indexed %s (%s): %d pages, %d chunks", doc["doc_id"], filename, len(pages), len(chunks))
        return doc, False


def _doc_summary(doc: dict, reused: bool) -> dict:
    return {
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "kind": doc["kind"],
        "pages": len(doc["pages"]),
        "chars": sum(len(p) for p in doc["pages"]),
        "headings_count": len(doc["headings"]),
        "chunks_count": len(doc["chunks"]),
        "reused": reused,
    }


def _extract_pdf(content: bytes) -> tuple[list[str], list[dict]]:
    """Page texts plus tiered headings (by font size relative to body text)."""
    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Couldn't open this file as a PDF. It may be corrupted.") from e
    with pdf:
        if pdf.needs_pass:
            raise HTTPException(status_code=400,
                                detail="This PDF is password-protected. Remove the password and upload it again.")
        pages, lines = [], []
        for number, page in enumerate(pdf, start=1):
            pages.append(page.get_text("text"))
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if spans:
                        lines.append((" ".join(s["text"].strip() for s in spans),
                                      max(s.get("size", 0) for s in spans), number))
    return pages, analysis.classify_headings(lines)


# ==========================
# === Analysis cache
# ==========================

def _analysis(doc: dict) -> dict:
    doc_id = doc["doc_id"]
    with _locks_guard:
        lock = _analysis_locks.setdefault(doc_id, threading.Lock())
    with lock:
        state = _analysis_cache.get(doc_id)
        if state is None:
            ctx = analysis.build_context(doc["pages"], doc["headings"], doc["kind"])
            concepts = analysis.analyze_concepts(ctx)
            state = {"ctx": ctx, "concepts": concepts, "llm_refined": False, "llm": None,
                     "path": learning.build_learning_path(ctx, concepts["topics"])}
            _analysis_cache[doc_id] = state
        return state


def _apply_llm_topics(doc: dict, terms: list[str]) -> dict:
    state = _analysis(doc)
    with _analysis_locks[doc["doc_id"]]:
        if not state["llm_refined"]:
            concepts = analysis.merge_llm_topics(state["ctx"], state["concepts"]["topics"], terms)
            state["llm"] = concepts.pop("llm")
            state["concepts"] = concepts
            state["path"] = learning.build_learning_path(state["ctx"], concepts["topics"])
            state["llm_refined"] = True
    return state


def _mastery_payload(doc_id: str, topics: list[dict]) -> dict:
    states = mastery_store.states(doc_id, [t["term"] for t in topics])
    return {
        "concepts": {term: mastery.public_state(s) for term, s in states.items()},
        "summary": mastery.summarize(topics, states),
    }


def _analysis_payload(doc_id: str, state: dict) -> dict:
    topics = state["concepts"]["topics"]
    return {
        "topics": topics,
        "edges": state["concepts"]["edges"],
        "study_priority": state["concepts"]["study_priority"],
        "learning_path": state["path"],
        "mastery": _mastery_payload(doc_id, topics),
        "llm_refined": state["llm_refined"],
        "llm": state["llm"],
    }


# ==========================
# === Retrieval
# ==========================

def _bm25(doc: dict) -> rag.BM25:
    index = _bm25_cache.get(doc["doc_id"])
    if index is None:
        index = rag.BM25([rag.tokenize(c["text"]) for c in doc["chunks"]])
        _bm25_cache[doc["doc_id"]] = index
    return index


def _retrieve(doc: dict, query: str, k: int = RETRIEVE_K) -> list[dict]:
    """Hybrid retrieval: dense (embeddings) and keyword (BM25) candidates fused with reciprocal rank fusion."""
    chunks = doc["chunks"]
    n = min(CANDIDATE_K, len(chunks))
    result = _collection(doc).query(query_embeddings=[embedder.encode(query).tolist()], n_results=n)
    dense = [int(i.split("_")[1]) for i in result["ids"][0]]
    keyword = _bm25(doc).top(rag.tokenize(query), n)
    fused = rag.reciprocal_rank_fusion([dense, keyword])[:k]
    return [{
        "index": idx,
        "text": chunks[idx]["text"],
        "label": rag.source_label(chunks[idx], doc["kind"], idx),
        "page_start": chunks[idx]["page_start"] if doc["kind"] == "pdf" else None,
        "dense_rank": dense.index(idx) + 1 if idx in dense else None,
        "keyword_rank": keyword.index(idx) + 1 if idx in keyword else None,
    } for idx, _ in fused]


def _similarity(a: str, b: str) -> float:
    vectors = embedder.encode([a, b], normalize_embeddings=True)
    return float(vectors[0] @ vectors[1])


# ==========================
# === Ollama
# ==========================

class OllamaError(Exception):
    """An Ollama failure with a message that is safe to show the user."""


def _ollama_error(status_code: int, body: bytes) -> OllamaError:
    try:
        message = json.loads(body).get("error", "")
    except (json.JSONDecodeError, AttributeError):
        message = ""
    if "not found" in message:
        return OllamaError(f"The model '{OLLAMA_MODEL}' isn't installed. Run: ollama pull {OLLAMA_MODEL}")
    log.error("Ollama returned %s: %s", status_code, body[:300])
    return OllamaError(f"The local model returned an error ({status_code}). Check the Ollama logs.")


def _ollama_payload(messages: list[dict], stream: bool, **options) -> dict:
    return {"model": OLLAMA_MODEL, "messages": messages, "stream": stream,
            "options": {"num_ctx": NUM_CTX, **options}}


async def _ollama_chat(messages: list[dict], fmt: str | None = None, timeout: float = 180, **options) -> str:
    payload = _ollama_payload(messages, stream=False, **options)
    if fmt:
        payload["format"] = fmt
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5)) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    if response.status_code != 200:
        raise _ollama_error(response.status_code, response.content)
    return response.json().get("message", {}).get("content", "")


async def _ollama_stream(messages: list[dict]):
    payload = _ollama_payload(messages, stream=True, num_predict=MAX_ANSWER_TOKENS, stop=["\nUser:"])
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5)) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as response:
            if response.status_code != 200:
                raise _ollama_error(response.status_code, await response.aread())
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise OllamaError(f"The local model stopped with an error: {data['error']}")
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    return


def _friendly_llm_error(e: Exception) -> str:
    if isinstance(e, OllamaError):
        return str(e)
    if isinstance(e, httpx.ConnectError):
        return "Ollama is not running. Start it with: ollama serve"
    if isinstance(e, httpx.TimeoutException):
        return "The local model timed out. It may be overloaded; try again."
    log.exception("Unexpected LLM failure", exc_info=e)
    return "Something went wrong while talking to the local model."


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_response(generator) -> StreamingResponse:
    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ==========================
# === Endpoints: documents
# ==========================

@app.get("/")
async def serve_ui():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health_check():
    reachable = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            reachable = (await client.get(f"{OLLAMA_URL}/api/tags")).status_code == 200
    except httpx.HTTPError:
        pass
    return {"status": "healthy", "documents": len(documents), "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL, "ollama_reachable": reachable, "num_ctx": NUM_CTX,
            "embed_device": EMBED_DEVICE}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract a PDF's pages and headings, then index it for retrieval."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported here.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    pages, headings = await run_in_threadpool(_extract_pdf, content)
    if not any(p.strip() for p in pages):
        raise HTTPException(status_code=422, detail="No text could be extracted. This looks like a scanned or "
                                                    "image-only PDF, and OCR isn't supported yet.")
    doc, reused = await run_in_threadpool(_ingest, pages, headings, "pdf", file.filename,
                                          hashlib.sha256(content).hexdigest())
    return _doc_summary(doc, reused)


@app.post("/ingest")
async def ingest_text(request: IngestRequest):
    """Index pasted text or a .txt file."""
    text = request.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Text is too short to index (minimum 20 characters).")
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Text is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    doc, reused = await run_in_threadpool(_ingest, [text], [], "text", request.filename,
                                          hashlib.sha256(text.encode("utf-8")).hexdigest())
    return _doc_summary(doc, reused)


@app.post("/analyze")
def analyze_document(request: DocRequest):
    """Concept graph, learning path and mastery for a document (computed once, then cached)."""
    doc = _get_doc(request.doc_id)
    return _analysis_payload(doc["doc_id"], _analysis(doc))


@app.post("/analyze-llm")
async def analyze_llm_topics(request: DocRequest):
    """Background LLM refinement: the model nominates concepts, the measurement engine scores them (SSE)."""
    doc = _get_doc(request.doc_id)

    async def generate():
        try:
            state = await run_in_threadpool(_analysis, doc)
            if not state["llm_refined"]:
                digest = analysis.build_llm_digest(state["ctx"])
                raw = await _ollama_chat([{"role": "user", "content": LLM_TOPIC_PROMPT.format(text=digest)}],
                                         fmt="json", temperature=0.3)
                terms = analysis.parse_llm_topics(raw)
                if not terms:
                    yield _sse({"status": "fallback", "reason": "The model didn't return a usable topic list."})
                    return
                state = await run_in_threadpool(_apply_llm_topics, doc, terms)
            yield _sse({"status": "success", **_analysis_payload(doc["doc_id"], state)})
        except Exception as e:
            yield _sse({"status": "fallback", "reason": _friendly_llm_error(e)})

    return _sse_response(generate())


@app.get("/documents/{doc_id}/concept")
def concept_details(doc_id: str, term: str):
    """Evidence for one concept: score breakdown, definitions, mentions, prerequisites, mastery."""
    doc = _get_doc(doc_id)
    state = _analysis(doc)
    topic = learning.find_topic(state["concepts"]["topics"], term)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"'{term}' isn't in this document's concept map.")
    details = learning.concept_evidence(state["ctx"], topic, state["path"])
    details["mastery"] = mastery.public_state(mastery_store.states(doc_id, [topic["term"]])[topic["term"]])
    return details


@app.get("/documents/{doc_id}/relation")
def relation_details(doc_id: str, a: str, b: str):
    """Evidence for a graph edge: where the two concepts appear together."""
    doc = _get_doc(doc_id)
    state = _analysis(doc)
    topics = state["concepts"]["topics"]
    ta, tb = learning.find_topic(topics, a), learning.find_topic(topics, b)
    if ta is None or tb is None:
        raise HTTPException(status_code=404, detail="Both concepts must be in this document's concept map.")
    return learning.relation_evidence(state["ctx"], ta["term"], tb["term"], state["path"])


@app.get("/documents/{doc_id}/mastery")
def get_mastery(doc_id: str):
    doc = _get_doc(doc_id)
    return _mastery_payload(doc_id, _analysis(doc)["concepts"]["topics"])


# ==========================
# === Endpoints: chat
# ==========================

def _session(session_id: str | None, doc_id: str) -> dict:
    sid = session_id or str(uuid.uuid4())
    session = sessions.get(sid)
    if session is None or session["doc_id"] != doc_id:
        session = {"id": sid, "doc_id": doc_id, "messages": []}
        sessions[sid] = session
    sessions.move_to_end(sid)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)
    return session


def _remember(session: dict, question: str, answer: str) -> None:
    session["messages"].append({"role": "user", "content": question})
    if answer:
        session["messages"].append({"role": "assistant", "content": answer})
    del session["messages"][:-HISTORY_MESSAGES * 2]


def _record_question_exposure(doc: dict, question: str) -> None:
    state = _analysis_cache.get(doc["doc_id"])
    if state:
        terms = [t["term"] for t in state["concepts"]["topics"] if analysis.term_regex(t["term"]).search(question)]
        if terms:
            mastery_store.record_exposure(doc["doc_id"], terms)


def _chat_messages(doc: dict, session: dict, question: str, sources: list[dict]) -> list[dict]:
    example = "[p. 4]" if doc["kind"] == "pdf" else "[Excerpt 2]"
    system = LEXO_SYSTEM_PROMPT.format(example=example)
    user = (f"Document excerpts:\n\n{rag.build_context_block(sources)}\n\n---\n\n"
            f"User's question: {question}")
    # Ollama silently truncates from the start of the prompt when it overflows num_ctx,
    # which would drop the system prompt first. Budget the history so that never happens.
    budget = int((NUM_CTX - MAX_ANSWER_TOKENS) * CHARS_PER_TOKEN) - len(system) - len(user)
    history = rag.fit_history(session["messages"][-HISTORY_MESSAGES:], max(0, budget))
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """Hybrid retrieval + streamed answer with page citations (SSE)."""
    doc = _get_doc(request.doc_id)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Ask a question first.")
    session = _session(request.session_id, doc["doc_id"])

    if rag.is_small_talk(question):
        reply = rag.small_talk_reply(question)

        async def small_talk():
            yield _sse({"token": reply})
            _remember(session, question, reply)
            yield _sse({"done": True, "session_id": session["id"]})

        return _sse_response(small_talk())

    sources = await run_in_threadpool(_retrieve, doc, question)
    await run_in_threadpool(_record_question_exposure, doc, question)
    messages = _chat_messages(doc, session, question, sources)

    async def generate():
        answer = []
        try:
            yield _sse({"sources": [{k: s[k] for k in ("label", "text", "page_start", "dense_rank", "keyword_rank")}
                                    for s in sources]})
            async for token in _ollama_stream(messages):
                answer.append(token)
                yield _sse({"token": token})
        except Exception as e:
            yield _sse({"error": _friendly_llm_error(e)})
        finally:
            _remember(session, question, "".join(answer))
        yield _sse({"done": True, "session_id": session["id"]})

    return _sse_response(generate())


# ==========================
# === Endpoints: quiz
# ==========================

def _quiz_context(evidence: dict, sources: list[dict]) -> tuple[str, list[dict]]:
    quotes = evidence["definitions"] + evidence["mentions"]
    lines = [f"[{q['label']}] {q['text']}" if q["label"] else q["text"] for q in quotes]
    lines += [f"[{s['label']}] {s['text']}" for s in sources]
    return "\n\n".join(lines)[:3500], quotes[:3]


def _cloze_question(evidence: dict, term: str) -> tuple[str, dict] | None:
    for quote in evidence["definitions"] + evidence["mentions"]:
        question = mastery.make_cloze(quote["text"], term)
        if question:
            return question, quote
    return None


@app.post("/documents/{doc_id}/quiz")
async def next_quiz(doc_id: str, request: QuizRequest):
    """Generate the next quiz question, targeted at an important concept the learner hasn't mastered."""
    doc = _get_doc(doc_id)
    state = await run_in_threadpool(_analysis, doc)
    topics, path = state["concepts"]["topics"], state["path"]
    if not topics:
        raise HTTPException(status_code=422, detail="There isn't enough content in this document to quiz on.")
    states = mastery_store.states(doc_id, [t["term"] for t in topics])

    if request.concept:
        topic = learning.find_topic(topics, request.concept)
        if topic is None:
            raise HTTPException(status_code=404, detail=f"'{request.concept}' isn't in this document's concept map.")
        reason = "You picked this concept"
        gap = mastery.prerequisite_gap(topic["term"], states, path)
        if gap:
            reason += f". Tip: it builds on {', '.join(gap)}, which you haven't mastered yet"
    else:
        term, reason = mastery.pick_quiz_target(topics, states, path, datetime.now(timezone.utc))
        topic = learning.find_topic(topics, term)
    term = topic["term"]

    evidence = await run_in_threadpool(learning.concept_evidence, state["ctx"], topic, path)
    sources = await run_in_threadpool(_retrieve, doc, term, 3)
    context, quotes = _quiz_context(evidence, sources)

    quiz, generated_by, notice = None, "llm", None
    try:
        raw = await _ollama_chat([{"role": "user", "content": QUIZ_PROMPT.format(concept=term, context=context)}],
                                 fmt="json", timeout=120, temperature=0.4)
        quiz = mastery.parse_quiz(raw)
        if quiz:
            quiz.update(kind="short", evidence=quotes)
    except Exception as e:
        notice = _friendly_llm_error(e)
    if quiz is None:
        cloze = _cloze_question(evidence, term)
        if cloze is None:
            raise HTTPException(status_code=422, detail=f"There isn't enough text about '{term}' to write a question.")
        question, quote = cloze
        quiz = {"kind": "cloze", "question": question, "answer": term, "evidence": [quote]}
        generated_by = "cloze"

    quiz_id = uuid.uuid4().hex
    pending_quizzes[quiz_id] = {**quiz, "doc_id": doc_id, "concept": term, "context": context}
    while len(pending_quizzes) > MAX_PENDING_QUIZZES:
        pending_quizzes.popitem(last=False)
    return {
        "quiz_id": quiz_id,
        "concept": term,
        "kind": quiz["kind"],
        "question": quiz["question"],
        "reason": reason,
        "generated_by": generated_by,
        "notice": notice,
        "mastery": mastery.public_state(states[term]),
    }


@app.post("/documents/{doc_id}/quiz/{quiz_id}/answer")
async def answer_quiz(doc_id: str, quiz_id: str, request: AnswerRequest):
    """Grade an answer, update mastery and the review schedule, and show the evidence."""
    quiz = pending_quizzes.get(quiz_id)
    if quiz is None or quiz["doc_id"] != doc_id:
        raise HTTPException(status_code=404, detail="This question has expired. Start a new one.")
    del pending_quizzes[quiz_id]
    answer = request.answer.strip()
    feedback = None

    if not answer or mastery.is_dont_know(answer):
        score, graded_by = 0.0, "no answer"
        feedback = "No problem. Here's what the document says."
    elif quiz["kind"] == "cloze":
        score, graded_by = mastery.grade_exact(answer, quiz["answer"]), "exact match"
    else:
        similarity = mastery.similarity_score(await run_in_threadpool(_similarity, answer, quiz["answer"]))
        llm_score = None
        try:
            prompt = GRADE_PROMPT.format(question=quiz["question"], reference=quiz["answer"],
                                         context=quiz["context"][:2500], answer=answer[:1500])
            llm_score, feedback = mastery.parse_grade(
                await _ollama_chat([{"role": "user", "content": prompt}], fmt="json", timeout=120, temperature=0))
        except Exception as e:
            log.info("LLM grading unavailable, using embedding similarity only: %s", _friendly_llm_error(e))
        score = mastery.blend(llm_score, similarity)
        graded_by = "llm + embedding similarity" if llm_score is not None else "embedding similarity"

    new_state = mastery_store.apply_score(doc_id, quiz["concept"], score)
    topics = (await run_in_threadpool(_analysis, _get_doc(doc_id)))["concepts"]["topics"]
    return {
        "concept": quiz["concept"],
        "score": round(score, 3),
        "verdict": mastery.verdict(score),
        "feedback": feedback,
        "reference_answer": quiz["answer"],
        "evidence": quiz["evidence"],
        "graded_by": graded_by,
        "mastery": mastery.public_state(new_state),
        "summary": _mastery_payload(doc_id, topics)["summary"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
