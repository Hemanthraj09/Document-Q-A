"""
Visual Document Q&A - FastAPI Backend
RAG pipeline: ChromaDB + sentence-transformers + Ollama with streaming
"""

import os
import uuid
import tempfile
import re
import json
import math
from typing import Optional, List, Dict
from collections import Counter

import networkx as nx

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# PDF processing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF upload will not work. Run: pip install pymupdf")

# NLP processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    NLP_SUPPORT = True
except ImportError:
    NLP_SUPPORT = False
    print("Warning: scikit-learn not installed. Document analysis will not work.")

# RAG components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    RAG_SUPPORT = True
except ImportError:
    RAG_SUPPORT = False
    print("Warning: RAG dependencies not installed. Run: pip install sentence-transformers chromadb")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Visual Document Q&A",
    description="RAG-powered document analysis with visual insights and streaming",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:latest")

# Initialize RAG components
embedder = None
chroma_client = None

if RAG_SUPPORT:
    print(f"Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print("RAG components ready")

# In-memory stores
document_store = {}  # doc_id -> {"top_topics": [...], "headings": [...], "chunks_count": int}
session_store = {}   # session_id -> {"doc_id": str, "messages": list}

# Lexo system prompt
LEXO_SYSTEM_PROMPT = """You are Lexo, an enthusiastic and insightful AI document analyst. Your personality:
- You're genuinely excited to help users understand their documents
- You give detailed, well-structured answers with clear explanations
- You use bullet points, examples, and highlights to make information digestible
- You reference specific parts of the document context when answering
- You're conversational and engaging — not robotic or dry
- If you can't find the answer in the provided context, say so honestly but suggest what the user could ask instead

When answering:
1. Start with a brief, direct answer to the question
2. Provide supporting details from the document context provided
3. If relevant, mention connections to other topics in the document
4. Keep your tone warm, helpful, and enthusiastic — like a knowledgeable friend explaining something interesting
5. Use markdown formatting: **bold** for key terms, bullet points for lists, and clear paragraph breaks"""


# ========================
# === Pydantic Models ===
# ========================

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    doc_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class TopicNode(BaseModel):
    term: str
    score: float
    is_heading: bool = False
    spread: float = 0.0       # Fraction of sections containing this topic
    centrality: float = 0.0   # PageRank centrality score
    tier: str = "minor"       # "core", "supporting", "minor"


class TopicEdge(BaseModel):
    source: str
    target: str
    weight: float


class AnalyzeRequest(BaseModel):
    text: str
    headings: Optional[List[str]] = None
    heading_levels: Optional[List[Dict]] = None  # [{"text": str, "level": int}]


class AnalyzeResponse(BaseModel):
    topics: List[TopicNode]
    edges: List[TopicEdge]
    headings: List[str]
    study_priority: str = ""  # "Top 5 topics cover ~X% of this document"


class IngestRequest(BaseModel):
    text: str
    headings: Optional[List[str]] = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks_count: int


# ============================
# === Utility Functions ===
# ============================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries when possible"""
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = text.rfind('. ', start + chunk_size // 2, end)
            last_newline = text.rfind('\n', start + chunk_size // 2, end)
            break_point = max(last_period, last_newline)
            if break_point > start:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


# ====================================
# === Text Cleaning for Analysis ===
# ====================================

# Static stopwords: English defaults + academic/structural noise
STATIC_STOPS = {
    # Common filler
    'page', 'document', 'also', 'may', 'etc', 'using', 'used',
    'based', 'one', 'two', 'three', 'four', 'five', 'new', 'like', 'use', 'make',
    'said', 'would', 'could', 'get', 'got', 'let', 'see', 'way',
    'know', 'need', 'take', 'come', 'go', 'made', 'well', 'just',
    'even', 'back', 'good', 'give', 'much', 'many', 'mr', 'ms', 'dr',
    # Academic / structural noise
    'fig', 'figure', 'table', 'section', 'chapter', 'module', 'ref', 'al',
    'exercise', 'problem', 'example', 'note', 'hint', 'appendix',
    'bibliography', 'reference', 'references', 'index', 'abstract',
    'introduction', 'conclusion', 'acknowledgement', 'acknowledgements',
    'vol', 'volume', 'journal', 'conference', 'proceedings', 'isbn',
    'copyright', 'published', 'publisher', 'press', 'edition',
    'university', 'department', 'faculty', 'institute',
}

# Regex patterns for metadata lines (Layer 2)
_METADATA_PATTERNS = re.compile(
    r'^\s*\d+\s*$'                              # Standalone page numbers
    r'|^\s*-\s*\d+\s*-\s*$'                     # Page nums like "- 3 -"
    r'|^\s*page\s+\d+'
    r'|^\s*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*$'  # Dates: dd/mm/yyyy
    r'|^\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d'
    r'|^\s*\S+@\S+\.\S+\s*$'                    # Email addresses
    r'|^\s*https?://\S+\s*$'                     # URLs
    r'|^\s*(?:doi|DOI)[:\s]'
    r'|^\s*(?:figure|fig|table|chapter|module|exercise)\s+\d',
    re.IGNORECASE | re.MULTILINE
)


def _strip_repeated_lines(pages: List[str], threshold: float = 0.3) -> List[str]:
    """Layer 1: Remove lines that appear verbatim across >30% of pages (headers, footers, etc.)"""
    if len(pages) < 3:
        return pages

    # Count how many pages each normalized line appears in
    line_page_count = Counter()
    for page_text in pages:
        # Unique normalized lines per page
        seen_on_page = set()
        for line in page_text.split('\n'):
            normalized = re.sub(r'[^a-z0-9\s]', '', line.lower()).strip()
            if normalized and len(normalized) > 2:
                seen_on_page.add(normalized)
        for nl in seen_on_page:
            line_page_count[nl] += 1

    # Blacklist lines appearing in >threshold of pages
    min_pages = max(2, int(len(pages) * threshold))
    blacklist = {line for line, count in line_page_count.items() if count >= min_pages}

    # Remove blacklisted lines from each page
    cleaned = []
    for page_text in pages:
        kept_lines = []
        for line in page_text.split('\n'):
            normalized = re.sub(r'[^a-z0-9\s]', '', line.lower()).strip()
            if normalized not in blacklist:
                kept_lines.append(line)
        cleaned.append('\n'.join(kept_lines))

    return cleaned


def _strip_metadata_lines(text: str) -> str:
    """Layer 2: Remove page numbers, dates, emails, URLs, DOIs, and short label lines"""
    lines = text.split('\n')
    kept = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            kept.append(line)
            continue
        # Skip lines matching metadata patterns
        if _METADATA_PATTERNS.match(stripped):
            continue
        # Skip very short lines that look like labels ("Dr. Smith", "Table 4")
        words = stripped.split()
        if len(words) <= 3 and any(c.isdigit() for c in stripped):
            continue
        kept.append(line)
    return '\n'.join(kept)


def _build_dynamic_stops(text: str) -> set:
    """Layer 3 (dynamic): Extract top-5 frequent unigrams from first+last page as branding stopwords"""
    pages = re.split(r'---\s*Page\s+\d+\s*---', text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        return set()

    # Use first and last page text
    edge_text = pages[0]
    if len(pages) > 1:
        edge_text += ' ' + pages[-1]

    # Tokenize and count, excluding existing static stops and short words
    base_stops = ENGLISH_STOP_WORDS.union(STATIC_STOPS)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', edge_text.lower())
    word_counts = Counter(w for w in words if w not in base_stops)

    # Return top 3 most frequent edge-page words (only if they appear 3+ times)
    return {w for w, c in word_counts.most_common(3) if c >= 3}


def clean_text_for_analysis(text: str) -> str:
    """Apply all cleaning layers to document text before TF-IDF analysis"""
    # Layer 1: Strip repeated lines across pages
    pages = re.split(r'(---\s*Page\s+\d+\s*---)', text)
    # Reconstruct page contents (odd indices are delimiters)
    page_contents = []
    for i, part in enumerate(pages):
        if not re.match(r'---\s*Page\s+\d+\s*---', part):
            page_contents.append(part)

    if len(page_contents) > 2:
        page_contents = _strip_repeated_lines(page_contents)

    cleaned = '\n\n'.join(page_contents)

    # Layer 2: Strip metadata lines
    cleaned = _strip_metadata_lines(cleaned)

    return cleaned


def _get_all_stops(text: str) -> list:
    """Build complete stopword list: ENGLISH_STOP_WORDS + static + dynamic"""
    dynamic = _build_dynamic_stops(text)
    return list(ENGLISH_STOP_WORDS.union(STATIC_STOPS).union(dynamic))


def _split_into_sections(text: str, headings: List[str] = None) -> List[str]:
    """Split text into sections for TF-IDF. Primary: by headings. Fallback: fixed 1500-char chunks."""
    sections = []

    # Primary: split by detected headings
    if headings and len(headings) >= 2:
        # Escape headings for regex and build split pattern
        heading_positions = []
        text_lower = text.lower()
        for h in headings:
            h_lower = h.lower().strip()
            idx = text_lower.find(h_lower)
            if idx >= 0:
                heading_positions.append(idx)

        if len(heading_positions) >= 2:
            heading_positions.sort()
            # Add start and end boundaries
            boundaries = [0] + heading_positions + [len(text)]
            for i in range(len(boundaries) - 1):
                section = text[boundaries[i]:boundaries[i+1]].strip()
                if len(section) > 30:
                    sections.append(section)

    # Fallback: if heading-based split gave < 3 sections, use fixed-size chunks
    if len(sections) < 3:
        sections = []
        start = 0
        while start < len(text):
            end = min(start + 1500, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                sections.append(chunk)
            start = end

    return sections if sections else [text]


def extract_top_topics(text: str, headings: List[str] = None) -> List[str]:
    """Quick TF-IDF to get top topic terms for retrieval boosting"""
    if not NLP_SUPPORT or len(text.strip()) < 50:
        return []

    # Clean text before analysis
    cleaned = clean_text_for_analysis(text)

    sections = _split_into_sections(cleaned, headings)

    all_stops = _get_all_stops(text)

    n_sections = len(sections)
    if n_sections < 3:
        min_df, max_df = 1, 1.0
    elif n_sections < 8:
        min_df, max_df = 1, 0.95
    else:
        min_df, max_df = 2, 0.7

    try:
        vectorizer = TfidfVectorizer(
            stop_words=all_stops,
            max_features=20,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        tfidf_matrix = vectorizer.fit_transform(sections)
        feature_names = list(vectorizer.get_feature_names_out())
        scores = tfidf_matrix.mean(axis=0).A1

        topic_scores = {}
        for term, score in zip(feature_names, scores):
            if score > 0:
                topic_scores[term] = score

        # Heading boost: terms in headings get 2x weight
        if headings:
            heading_terms = set()
            for h in headings:
                for w in h.lower().split():
                    if len(w) > 2:
                        heading_terms.add(w)
            for term in topic_scores:
                if any(w in heading_terms for w in term.lower().split()):
                    topic_scores[term] *= 2.0

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return [t[0] for t in sorted_topics]
    except Exception:
        return []


def ingest_document(text: str, headings: List[str] = None) -> tuple:
    """Chunk, embed, and store document in ChromaDB. Returns (doc_id, chunks_count)"""
    if not RAG_SUPPORT:
        raise HTTPException(status_code=503, detail="RAG components not available. Install: pip install sentence-transformers chromadb")

    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    # Chunk the document
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to index")

    # Embed all chunks
    embeddings = embedder.encode(chunks).tolist()

    # Store in ChromaDB with cosine similarity
    collection = chroma_client.create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i} for i in range(len(chunks))]
    )

    # Extract top topics for retrieval boosting
    top_topics = extract_top_topics(text, headings)

    # Store metadata
    document_store[doc_id] = {
        "top_topics": top_topics,
        "headings": headings or [],
        "chunks_count": len(chunks)
    }

    return doc_id, len(chunks)


# ========================
# === API Endpoints ===
# ========================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Visual Document Q&A</h1><p>UI not found. Please ensure index.html exists.</p>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pdf_support": PDF_SUPPORT,
        "nlp_support": NLP_SUPPORT,
        "rag_support": RAG_SUPPORT,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, detect headings, and ingest into RAG pipeline"""

    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=503,
            detail="PDF support not available. Install PyMuPDF: pip install pymupdf"
        )

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Read PDF content
        content = await file.read()

        # Create temp file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Extract text from PDF
            doc = fitz.open(tmp_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            # Heading extraction with tiered levels using font size analysis
            headings = []
            heading_levels = []  # [{"text": str, "level": 1|2|3}]
            try:
                all_font_sizes = []
                text_with_sizes = []

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_dict = page.get_text("dict")
                    for block in page_dict.get("blocks", []):
                        if block.get("type") != 0:  # Skip image blocks
                            continue
                        for line in block.get("lines", []):
                            line_text = ""
                            max_size = 0
                            for span in line.get("spans", []):
                                span_text = span.get("text", "").strip()
                                if span_text:
                                    line_text += span_text + " "
                                    size = span.get("size", 0)
                                    max_size = max(max_size, size)
                                    all_font_sizes.append(size)
                            line_text = line_text.strip()
                            if line_text and max_size > 0:
                                text_with_sizes.append((line_text, max_size))

                if all_font_sizes:
                    # Find body size (mode) and heading thresholds
                    size_counts = Counter(round(s, 1) for s in all_font_sizes)
                    body_size = size_counts.most_common(1)[0][0]
                    # Tiered thresholds: H1 > 1.6x, H2 > 1.3x, H3 > 1.15x body
                    h1_threshold = body_size * 1.6
                    h2_threshold = body_size * 1.3
                    h3_threshold = body_size * 1.15

                    for item_text, size in text_with_sizes:
                        if size > h3_threshold and 3 < len(item_text) < 150:
                            headings.append(item_text)
                            if size >= h1_threshold:
                                heading_levels.append({"text": item_text, "level": 1})
                            elif size >= h2_threshold:
                                heading_levels.append({"text": item_text, "level": 2})
                            else:
                                heading_levels.append({"text": item_text, "level": 3})
            except Exception:
                pass  # Heading extraction is best-effort

            total_pages = len(doc)
            doc.close()
            full_text = "\n\n".join(text_parts)

            # Build response
            result = {
                "filename": file.filename,
                "pages": total_pages,
                "text": full_text,
                "chars": len(full_text),
                "headings": headings,
                "heading_levels": heading_levels,
            }

            if not full_text.strip():
                result["warning"] = "No text could be extracted. This may be a scanned or image-based PDF."
            elif len(full_text) < 50:
                result["warning"] = "Very little text was extracted. The PDF may contain mostly images."

            # Ingest into RAG pipeline
            if RAG_SUPPORT and full_text.strip():
                doc_id, chunks_count = ingest_document(full_text, headings)
                result["doc_id"] = doc_id
                result["chunks_count"] = chunks_count

            return result

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_text_endpoint(request: IngestRequest):
    """Ingest plain text into the RAG pipeline"""
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short to index")

    doc_id, chunks_count = ingest_document(request.text, request.headings)
    return IngestResponse(doc_id=doc_id, chunks_count=chunks_count)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(request: AnalyzeRequest):
    """Analyze document text to extract topics, relationships, and insights"""

    if not NLP_SUPPORT:
        raise HTTPException(
            status_code=503,
            detail="NLP support not available. Install scikit-learn: pip install scikit-learn"
        )

    text = request.text
    headings = request.headings or []

    print(f"[ANALYZE] Raw text length: {len(text)} chars")

    # Need minimum content for analysis
    if not text or len(text.strip()) < 50:
        print("[ANALYZE] ABORT: text too short (<50 chars)")
        return AnalyzeResponse(topics=[], edges=[], headings=headings)

    # Clean text: strip repeated lines, metadata, page numbers
    cleaned = clean_text_for_analysis(text)
    print(f"[ANALYZE] Text after cleaning (repeated lines + metadata strip): {len(cleaned)} chars (removed {len(text) - len(cleaned)})")

    # Split into sections by headings, fallback to 1500-char chunks
    sections = _split_into_sections(cleaned, headings)

    print(f"[ANALYZE] Number of sections passed to TF-IDF: {len(sections)}")
    for i, sec in enumerate(sections[:3]):
        print(f"[ANALYZE]   Section {i}: {len(sec)} chars, preview: {sec[:80]!r}...")

    # Build stopword list: English + static + dynamic (from first/last page)
    dynamic = _build_dynamic_stops(text)
    print(f"[ANALYZE] Dynamic stopwords identified: {dynamic}")
    all_stops = list(ENGLISH_STOP_WORDS.union(STATIC_STOPS).union(dynamic))

    n_sections = len(sections)
    if n_sections < 3:
        min_df, max_df = 1, 1.0
    elif n_sections < 8:
        min_df, max_df = 1, 0.95
    else:
        min_df, max_df = 2, 0.7
    print(f"[ANALYZE] min_df={min_df}, max_df={max_df} (sections: {n_sections})")

    try:
        vectorizer = TfidfVectorizer(
            stop_words=all_stops,
            max_features=50,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )

        tfidf_matrix = vectorizer.fit_transform(sections)
        feature_names = list(vectorizer.get_feature_names_out())

        print(f"[ANALYZE] TF-IDF vocabulary size after fitting: {len(vectorizer.vocabulary_)}")
        print(f"[ANALYZE] Top 20 terms from TF-IDF: {feature_names[:20]}")

    except ValueError as e:
        print(f"[ANALYZE] TF-IDF FAILED with ValueError: {e}")
        return AnalyzeResponse(topics=[], edges=[], headings=headings)

    # === STEP 1: Base TF-IDF frequency scores ===
    overall_scores = tfidf_matrix.mean(axis=0).A1
    freq_scores = {}
    for term, score in zip(feature_names, overall_scores):
        if score > 0:
            freq_scores[term] = score

    if not freq_scores:
        print("[ANALYZE] ABORT: no terms with score > 0")
        return AnalyzeResponse(topics=[], edges=[], headings=headings)

    # Normalize frequency to [0, 1]
    max_freq = max(freq_scores.values())
    norm_freq = {t: s / max_freq for t, s in freq_scores.items()}

    # === STEP 2: Spread Score ===
    term_to_idx = {name: i for i, name in enumerate(feature_names)}
    spread_scores = {}
    for term in freq_scores:
        idx = term_to_idx[term]
        sections_with_term = sum(1 for s in range(tfidf_matrix.shape[0]) if tfidf_matrix[s, idx] > 0)
        spread_scores[term] = sections_with_term / n_sections

    print(f"[ANALYZE] Spread scores computed for {len(spread_scores)} terms")

    # === STEP 3: Tiered Heading Weights ===
    heading_levels = request.heading_levels or []
    # Build term -> max heading weight mapping
    h_weight_map = {}  # term -> heading weight (0 if not in heading)
    level_multipliers = {1: 3.0, 2: 2.0, 3: 1.5}  # H1=3x, H2=2x, H3=1.5x

    for hl in heading_levels:
        h_text = hl.get("text", "").lower()
        h_level = hl.get("level", 3)
        multiplier = level_multipliers.get(h_level, 1.5)
        for term in freq_scores:
            term_words = term.lower().split()
            if any(w in h_text for w in term_words if len(w) > 2) or term.lower() in h_text:
                h_weight_map[term] = max(h_weight_map.get(term, 0), multiplier)

    # Fallback: flat 2x for untiered headings
    if not heading_levels and headings:
        heading_text_lower = ' '.join(headings).lower()
        for term in freq_scores:
            term_words = term.lower().split()
            if any(w in heading_text_lower for w in term_words if len(w) > 2):
                h_weight_map[term] = max(h_weight_map.get(term, 0), 2.0)

    # Normalize heading weights to [0, 1]: 0 = not in heading, 1 = in H1
    max_hw = max(level_multipliers.values())  # 3.0
    norm_heading = {t: h_weight_map.get(t, 0) / max_hw for t in freq_scores}

    heading_terms = set()
    for h in headings:
        for w in h.lower().split():
            if len(w) > 2:
                heading_terms.add(w)

    print(f"[ANALYZE] Heading weights: {len(h_weight_map)} terms boosted")

    # === STEP 4: Co-occurrence edges + PageRank Centrality ===
    G = nx.Graph()
    for term in freq_scores:
        G.add_node(term)

    edges = []
    for t1 in freq_scores:
        for t2 in freq_scores:
            if t1 >= t2:
                continue
            if t1 not in term_to_idx or t2 not in term_to_idx:
                continue
            idx1, idx2 = term_to_idx[t1], term_to_idx[t2]
            co_occur = sum(
                1 for s in range(tfidf_matrix.shape[0])
                if tfidf_matrix[s, idx1] > 0 and tfidf_matrix[s, idx2] > 0
            )
            if co_occur > 0:
                weight = co_occur / n_sections
                if weight >= 0.05:
                    G.add_edge(t1, t2, weight=weight)
                    if weight >= 0.1:
                        edges.append(TopicEdge(source=t1, target=t2, weight=round(weight, 3)))

    # Compute PageRank
    try:
        centrality = nx.pagerank(G, weight='weight')
    except Exception:
        centrality = {t: 1.0 / len(freq_scores) for t in freq_scores}

    # Normalize centrality to [0, 1]
    max_cent = max(centrality.values()) if centrality else 1.0
    norm_centrality = {t: centrality.get(t, 0) / max_cent for t in freq_scores}

    print(f"[ANALYZE] PageRank computed: {len(centrality)} nodes, {G.number_of_edges()} edges")

    # === STEP 5: Composite Importance Score ===
    # importance = 0.30*frequency + 0.25*spread + 0.20*heading + 0.25*centrality
    composite = {}
    for term in freq_scores:
        composite[term] = (
            0.30 * norm_freq.get(term, 0) +
            0.25 * spread_scores.get(term, 0) +
            0.20 * norm_heading.get(term, 0) +
            0.25 * norm_centrality.get(term, 0)
        )

    # Sort by composite score, take top 20
    sorted_topics_pre = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    
    # Duplicate topic normalization: group by stem
    def simple_stem(word):
        word = word.lower()
        for suffix in ['ing', 'tion', 'ions', 'ings', 'ers', 'ies', 'es', 's']:
            if word.endswith(suffix) and len(word) - len(suffix) > 3:
                return word[:-len(suffix)]
        return word

    stems = {}
    for topic, score in sorted_topics_pre:
        stem = simple_stem(topic)
        if stem in stems:
            # keep whichever is longer (more specific)
            if len(topic) > len(stems[stem]):
                stems[stem] = topic
        else:
            stems[stem] = topic
    deduplicated_topics = list(stems.values())
    
    # Filter composite to deduplicated topics
    composite = {k: v for k, v in composite.items() if k in deduplicated_topics}
    
    # Exam signal boost
    for term in composite:
        exam_signal_count = 0
        for sec in sections:
            sec_lower = sec.lower()
            if term.lower() in sec_lower:
                if sec_lower.startswith("definition:") or sec_lower.startswith("key definition") or "defined as" in sec_lower:
                    composite[term] *= 1.3
                    exam_signal_count += 1
                if "figure" in sec_lower or "table" in sec_lower or "diagram" in sec_lower:
                    composite[term] *= 1.2
                    exam_signal_count += 1
                if "example:" in sec_lower or "case study" in sec_lower or "for example" in sec_lower:
                    composite[term] *= 1.15
                    exam_signal_count += 1
        
        if exam_signal_count > 3:
            composite[term] *= 1.1
            
        # Cap the maximum importance at 1.0 after all boosts to keep normalization intact
        composite[term] = min(composite[term], 1.0)
        
    sorted_topics = sorted(composite.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"[ANALYZE] Composite scores computed. Top 5: {[(t, round(s, 3)) for t, s in sorted_topics[:5]]}")

    if not sorted_topics:
        print("[ANALYZE] ABORT: no topics after composite scoring")
        return AnalyzeResponse(topics=[], edges=[], headings=headings)

    # Normalize composite to [0, 1]
    max_composite = sorted_topics[0][1]

    topics = []
    for term, score in sorted_topics:
        norm_score = round(score / max_composite, 3) if max_composite > 0 else 0
        is_heading = any(w in heading_terms for w in term.lower().split())
        # Tier classification
        if norm_score > 0.55:
            tier = "core"
        elif norm_score > 0.3:
            tier = "supporting"
        else:
            tier = "minor"

        topics.append(TopicNode(
            term=term,
            score=norm_score,
            is_heading=is_heading,
            spread=round(spread_scores.get(term, 0), 3),
            centrality=round(norm_centrality.get(term, 0), 3),
            tier=tier
        ))

    # Study Priority: Top 5 importance as proportion of total
    total_importance = sum(topic.score for topic in topics)
    top5_importance = sum(topic.score for topic in sorted(topics, key=lambda x: x.score, reverse=True)[:5])
    if total_importance == 0:
        print("Warning: total_importance is 0")
        coverage = 0
        study_priority = "Top 5 topics cover ~0% of this document"
    else:
        coverage = (top5_importance / total_importance) * 100
        study_priority = f"Top 5 topics cover ~{round(coverage)}% of this document"

    print(f"[ANALYZE] Final: {len(topics)} topics, {len(edges)} edges, priority: {study_priority}")

    return AnalyzeResponse(
        topics=topics,
        edges=edges,
        headings=headings,
        study_priority=study_priority
    )


# ==========================================
# === Async LLM Topic Extraction (SSE) ===
# ==========================================

LLM_TOPIC_PROMPT = """You are analyzing a document to extract its most important concepts.
Extract exactly 15-20 core topics from the text below.
Rules:
- Return ONLY specific named concepts, not generic vocabulary
- Bad examples: "performance", "model", "service", "data", "system", "process", "approach", "method", "high", "large"
- Good examples: domain-specific named concepts, acronyms, proper technical terms, named frameworks, named algorithms, named theories, named events, named entities
- Each topic must be something worth knowing about this document specifically — if it could appear in any document on any topic, it is too generic
- If the document is technical: return specific technologies, protocols, algorithms, architectures
- If the document is scientific: return specific phenomena, theories, methods, named experiments
- If the document is legal/business: return specific clauses, frameworks, regulations, named processes
- If the document is historical/general: return specific events, figures, movements, named concepts
- Return a JSON array only, no explanation, no markdown, no backticks
["topic1", "topic2", ...]

Document text:
---
{text}
---"""


class LLMTopicRequest(BaseModel):
    text: str
    existing_topics: List[dict] = []  # Current TF-IDF topics [{term, score, ...}]
    existing_edges: List[dict] = []   # Current edges


@app.post("/analyze-llm")
async def analyze_llm_topics(request: LLMTopicRequest):
    """Async LLM topic extraction — returns SSE stream that emits merged topics when ready"""

    async def generate():
        try:
            # Truncate text for LLM context window (phi3 handles ~4K tokens well)
            doc_text = request.text[:8000]
            prompt = LLM_TOPIC_PROMPT.format(text=doc_text)

            print(f"[LLM-TOPICS] Sending {len(doc_text)} chars to Ollama...")

            # Call Ollama non-streaming for JSON response
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "stop": ["User:", "Question:"]
                        }
                    }
                )
                data = response.json()
                raw_answer = data.get("message", {}).get("content", "")

            print(f"[LLM-TOPICS] Raw response: {raw_answer[:300]}")

            # Parse JSON safely
            llm_topics = _parse_llm_topics(raw_answer)

            if not llm_topics:
                print("[LLM-TOPICS] No valid topics parsed, falling back")
                yield f"data: {json.dumps({'status': 'fallback', 'reason': 'No valid JSON from LLM'})}\n\n"
                return

            print(f"[LLM-TOPICS] Parsed {len(llm_topics)} topics: {llm_topics[:5]}...")

            # Merge with existing TF-IDF topics
            existing = request.existing_topics
            existing_edges = request.existing_edges
            merged = _merge_llm_topics(existing, llm_topics, existing_edges)

            yield f"data: {json.dumps({'status': 'success', 'topics': merged['topics'], 'edges': merged['edges'], 'study_priority': merged['study_priority']})}\n\n"

        except httpx.ConnectError:
            print("[LLM-TOPICS] Ollama not reachable")
            yield f"data: {json.dumps({'status': 'fallback', 'reason': 'Ollama not running'})}\n\n"
        except httpx.TimeoutException:
            print("[LLM-TOPICS] Ollama timeout")
            yield f"data: {json.dumps({'status': 'fallback', 'reason': 'LLM timeout'})}\n\n"
        except Exception as e:
            print(f"[LLM-TOPICS] Error: {e}")
            yield f"data: {json.dumps({'status': 'fallback', 'reason': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def _parse_llm_topics(raw: str) -> List[str]:
    """Safely extract a JSON array of topic strings from LLM output"""
    # Try direct parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if isinstance(t, str) and len(t.strip()) > 1]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from markdown/wrapper text
    match = re.search(r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if isinstance(t, str) and len(t.strip()) > 1]
        except json.JSONDecodeError:
            pass

    print(f"[LLM-TOPICS] Failed to parse JSON from: {raw[:200]}")
    return []


def _merge_llm_topics(existing_topics: List[dict], llm_topics: List[str], existing_edges: List[dict]) -> dict:
    """Merge LLM-extracted topics with existing TF-IDF topics"""
    # Build lookup of existing topics (case-insensitive)
    existing_map = {}
    for t in existing_topics:
        existing_map[t['term'].lower()] = t

    merged = list(existing_topics)  # Start with all existing
    new_count = 0

    for llm_term in llm_topics:
        llm_lower = llm_term.lower().strip()

        # Check if already exists (case-insensitive)
        if llm_lower in existing_map:
            # Boost existing topic's score by 2x (LLM validated)
            for i, t in enumerate(merged):
                if t['term'].lower() == llm_lower:
                    merged[i] = dict(t)
                    merged[i]['score'] = min(1.0, round(t['score'] * 2.0, 3))
                    # Upgrade tier if boosted above threshold
                    if merged[i]['score'] > 0.55:
                        merged[i]['tier'] = 'core'
                    elif merged[i]['score'] > 0.3:
                        merged[i]['tier'] = 'supporting'
                    break
            continue

        # Check for partial match (substring)
        partial_match = False
        for existing_lower in existing_map:
            if llm_lower in existing_lower or existing_lower in llm_lower:
                partial_match = True
                break

        if partial_match:
            continue

        # New topic from LLM — add with moderate score
        new_count += 1
        merged.append({
            'term': llm_term,
            'score': 0.55,  # Supporting tier
            'is_heading': False,
            'spread': 0.0,
            'centrality': 0.0,
            'tier': 'supporting'
        })

    # Re-sort by score and take top 25 (expanded from 20)
    merged.sort(key=lambda t: t['score'], reverse=True)
    merged = merged[:25]

    # Recalculate study priority
    total_importance = sum(t['score'] for t in merged)
    top5_importance = sum(t['score'] for t in merged[:5])
    coverage_pct = round(top5_importance / total_importance * 100) if total_importance > 0 else 0
    study_priority = f"Top 5 topics cover ~{coverage_pct}% of this document"

    print(f"[LLM-TOPICS] Merged: {len(merged)} topics ({new_count} new from LLM), priority: {study_priority}")

    return {
        'topics': merged,
        'edges': existing_edges,  # Keep existing edges (LLM topics won't have co-occurrence data)
        'study_priority': study_priority
    }


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """RAG-powered chat with streaming response via Server-Sent Events"""

    if not RAG_SUPPORT:
        raise HTTPException(status_code=503, detail="RAG components not available")

    session_id = request.session_id or str(uuid.uuid4())
    doc_id = request.doc_id

    if not doc_id or doc_id not in document_store:
        raise HTTPException(status_code=400, detail="No document indexed. Upload a document first.")

    # Initialize session if new
    if session_id not in session_store:
        session_store[session_id] = {
            "doc_id": doc_id,
            "messages": []
        }
    elif session_store[session_id].get("doc_id") != doc_id:
        session_store[session_id]["doc_id"] = doc_id
        session_store[session_id]["messages"] = []

    # 0. Check for greeting / non-question
    question_words = ["what", "how", "why", "where", "when", "explain", "tell", "describe"]
    question_lower = request.question.lower().strip()
    words = question_lower.split()
    
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "morning", "good morning", "good evening", "good afternoon"]
    q_clean = "".join(c for c in question_lower if c.isalnum() or c.isspace()).strip()
    
    is_greeting = any(q_clean == g for g in greetings)
    is_short_non_question = len(words) < 4 and not any(qw in question_lower for qw in question_words)
    
    if is_greeting or is_short_non_question:
        greeting_response = "Hey! I'm Lexo, your document analyst. Ask me anything about the document you've uploaded!"
        
        async def generate_greeting():
            yield f"data: {json.dumps({'token': greeting_response})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
            
            session_store[session_id]["messages"].append({"role": "user", "content": request.question})
            session_store[session_id]["messages"].append({"role": "assistant", "content": greeting_response})
            
        return StreamingResponse(
            generate_greeting(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # 1. Embed the question
    query_embedding = embedder.encode(request.question).tolist()

    # 2. Retrieve top-5 chunks from ChromaDB
    try:
        collection = chroma_client.get_collection(name=doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Document collection not found")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    # 3. Normalize cosine distance to similarity score [0, 1]
    # ChromaDB cosine distance range: [0, 2] → similarity: [1, 0]
    similarities = [1 - (d / 2) for d in distances]

    # 4. Query-aware topic boost
    doc_data = document_store[doc_id]
    top_topics = doc_data.get("top_topics", [])
    question_lower = request.question.lower()
    query_has_topic = any(topic in question_lower for topic in top_topics)

    scored_chunks = list(zip(chunks, similarities))

    if query_has_topic:
        boosted = []
        for chunk, sim in scored_chunks:
            chunk_lower = chunk.lower()
            topic_count = sum(1 for t in top_topics[:5] if t in chunk_lower)
            boost = min(topic_count * 0.05, 0.15)  # Max 0.15 boost
            boosted.append((chunk, sim + boost))
        boosted.sort(key=lambda x: x[1], reverse=True)
        chunks = [b[0] for b in boosted]
    else:
        chunks = [c[0] for c in scored_chunks]

    # 5. Build augmented prompt
    context = "\n\n---\n\n".join(chunks)

    # Build messages for Ollama
    messages = [{"role": "system", "content": LEXO_SYSTEM_PROMPT}]
    print(f"[CHAT-STREAM] System Prompt (first 100 chars): {LEXO_SYSTEM_PROMPT[:100]}")

    # Add recent conversation history (last 10 messages = 5 turns)
    history = session_store[session_id]["messages"][-10:]
    messages.extend(history)

    # Add current question with retrieved context
    user_message = f"Here are relevant sections from the document:\n\n{context}\n\n---\n\nUser's question: {request.question}"
    messages.append({"role": "user", "content": user_message})

    # 6. Stream from Ollama
    async def generate():
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "stop": ["User:", "Question:"]
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if data.get("done"):
                                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Ollama is not running. Start it with: ollama serve'})}\n\n"
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Response timed out. The model may be overloaded.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Store in conversation history (only the raw question, not the context)
        session_store[session_id]["messages"].append(
            {"role": "user", "content": request.question}
        )
        if full_response:
            session_store[session_id]["messages"].append(
                {"role": "assistant", "content": full_response}
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_non_stream(request: ChatRequest):
    """Non-streaming chat fallback (used by AI Summary)"""

    if not RAG_SUPPORT:
        raise HTTPException(status_code=503, detail="RAG components not available")

    session_id = request.session_id or str(uuid.uuid4())
    doc_id = request.doc_id

    # For summary requests, use the question directly with Ollama (no retrieval)
    messages = [
        {"role": "system", "content": LEXO_SYSTEM_PROMPT},
        {"role": "user", "content": request.question}
    ]
    print(f"[CHAT] System Prompt (first 100 chars): {LEXO_SYSTEM_PROMPT[:100]}")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "stop": ["User:", "Question:"]
                    }
                }
            )
            data = response.json()
            answer = data.get("message", {}).get("content", "No response received")

        return ChatResponse(answer=answer, session_id=session_id)

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Response timed out")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
