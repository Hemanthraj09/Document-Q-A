"""Document structure and concept-importance analysis.

Builds a page-aware view of a document (cleaned text, sections, sentences) and ranks
its concepts with a four-signal composite score: TF-IDF frequency, section spread,
heading weight, and PageRank centrality on a co-occurrence graph.
"""

import bisect
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

log = logging.getLogger(__name__)

WEIGHTS = {"frequency": 0.30, "spread": 0.25, "heading": 0.20, "centrality": 0.25}
HEADING_MULTIPLIERS = {1: 3.0, 2: 2.0, 3: 1.5}
EXAM_BOOSTS = {"definition": 1.3, "figure": 1.2, "example": 1.15}
LLM_ENDORSEMENT_BOOST = 2.0
CORE_THRESHOLD = 0.55
SUPPORTING_THRESHOLD = 0.30
MAX_TOPICS = 20
MAX_MERGED_TOPICS = 25
GRAPH_EDGE_MIN = 0.05
DISPLAY_EDGE_MIN = 0.10
MIN_SECTION_CHARS = 30
FALLBACK_SECTION_CHARS = 1500
FRAGMENT_SHARE = 0.7

STATIC_STOPS = {
    'page', 'document', 'also', 'may', 'etc', 'using', 'used',
    'based', 'one', 'two', 'three', 'four', 'five', 'new', 'like', 'use', 'make',
    'said', 'would', 'could', 'get', 'got', 'let', 'see', 'way',
    'know', 'need', 'take', 'come', 'go', 'made', 'well', 'just',
    'even', 'back', 'good', 'give', 'much', 'many', 'mr', 'ms', 'dr',
    'fig', 'figure', 'table', 'section', 'chapter', 'module', 'ref', 'al',
    'exercise', 'problem', 'example', 'note', 'hint', 'appendix',
    'bibliography', 'reference', 'references', 'index', 'abstract',
    'introduction', 'conclusion', 'acknowledgement', 'acknowledgements',
    'vol', 'volume', 'journal', 'conference', 'proceedings', 'isbn',
    'copyright', 'published', 'publisher', 'press', 'edition',
    'university', 'department', 'faculty', 'institute',
    # Generic adjectives/verbs that TF-IDF happily nominates but are never concepts on their own
    'common', 'large', 'small', 'high', 'low', 'different', 'important', 'general', 'simple', 'main',
    'major', 'various', 'several', 'certain', 'possible', 'able', 'better', 'best', 'worse', 'far',
    'reduce', 'reduces', 'increase', 'increases', 'tries', 'try', 'makes', 'shows', 'show', 'gives',
    'turn', 'turns', 'moves', 'called', 'known', 'explains', 'explained', 'detail', 'details',
    'ways', 'thing', 'things', 'lot', 'lots', 'part', 'parts', 'point', 'points', 'case', 'cases',
    'starting', 'start', 'end', 'next', 'previous', 'following', 'instead', 'usually', 'always',
}
_PAGE_REF_END = re.compile(r'(?:\.{3,}|…+)\s*\d{1,4}\s*$|\(?\b(?:page|pg\.?|p\.)\s*\d{1,4}\)?\s*$', re.IGNORECASE)
_TRAILING_NUMBER = re.compile(r'^\D.{2,100}?\s\d{1,4}$')

_METADATA_PATTERNS = re.compile(
    r'^\s*\d+\s*$'
    r'|^\s*-\s*\d+\s*-\s*$'
    r'|^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$'
    r'|^\s*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*$'
    r'|^\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d'
    r'|^\s*\S+@\S+\.\S+\s*$'
    r'|^\s*https?://\S+\s*$'
    r'|^\s*(?:doi|DOI)[:\s]'
    r'|^\s*(?:figure|fig|table|chapter|module|exercise)\s+\d',
    re.IGNORECASE
)
_LIST_ITEM = re.compile(r"^(?:[•\-–*▪●◦]|\d+[.)])\s")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_FIGURE_SIGNAL = re.compile(r"\b(?:figure|fig\.|table|diagram)\b")
_EXAMPLE_SIGNAL = re.compile(r"\b(?:for example|for instance|example:|case study)\b")


# ==========================
# === Term matching
# ==========================

def singular(word: str) -> str:
    """Light plural folding so 'networks' and 'network' compare equal."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def term_key(term: str) -> str:
    """Canonical form of a term for de-duplication."""
    return " ".join(singular(w) for w in re.findall(r"[a-z0-9]+", term.lower()))


def term_body(term: str) -> str:
    """Regex fragment matching `term` as whole words, tolerant of plurals, hyphens and wrapped whitespace."""
    words = term.lower().split()
    if not words:
        return r"(?!x)x"
    parts = [re.escape(w) for w in words[:-1]]
    last = singular(words[-1])
    if last.endswith("y") and len(last) > 3 and not last.endswith(("ay", "ey", "oy", "uy")):
        parts.append(re.escape(last[:-1]) + "(?:y|ies)")
    else:
        parts.append(re.escape(last) + "(?:s|es)?")
    # Words may be separated by spaces or hyphens, but never by a newline (a heading and the next line).
    return r"(?<![A-Za-z0-9])" + r"(?:[^\S\n]|-)+".join(parts) + r"(?![A-Za-z0-9])"


@lru_cache(maxsize=8192)
def term_regex(term: str) -> re.Pattern:
    return re.compile(term_body(term), re.IGNORECASE)


# "X is a ...", "X is when ...", "X refers to ..." read as definitions; "X is shown in ..." does not.
_DEF_VERBS = (r"(?:(?:is|are|was|were)\s+(?:an?|the|one of|any|when|how|what|called|known as|defined as)"
              r"|refers? to|means?|denotes?|describes?|represents?|can be (?:defined|described) as)")


@lru_cache(maxsize=4096)
def _definition_regexes(term: str) -> tuple[re.Pattern, ...]:
    body = term_body(term)
    return (
        # "Gradient descent is ...", "In ML, a neural network (NN) is ..."
        re.compile(rf"^\W*(?:[^,.;]{{0,40}},\s*)?(?:an?\s+|the\s+)?{body}\s*(?:\([^)]*\)\s*)?,?\s+{_DEF_VERBS}\b",
                   re.IGNORECASE),
        # "... is known as overfitting", "... is called a kernel"
        re.compile(rf"(?:defined as|known as|called|termed|referred to as)\s+(?:an?\s+|the\s+)?[\"'“]?{body}",
                   re.IGNORECASE),
        # "Overfitting: when a model ..."
        re.compile(rf"^\W*{body}\s*[:–—-]\s+\S", re.IGNORECASE),
    )


def is_definitional(sentence: str, term: str) -> bool:
    """True if the sentence reads like a definition of `term`."""
    return any(rx.search(sentence) for rx in _definition_regexes(term))


# ==========================
# === Document context
# ==========================

@dataclass
class Sentence:
    text: str
    start: int
    page: int
    section: int
    words: int


@dataclass
class Section:
    start: int
    end: int
    title: str | None
    page: int
    text: str
    flags: frozenset


@dataclass
class DocContext:
    kind: str
    pages: list[str]
    text: str
    page_starts: list[int]
    sections: list[Section]
    sentences: list[Sentence]
    headings: list[dict]
    _sentence_starts: list[int] = field(default_factory=list, repr=False)
    _term_cache: dict = field(default_factory=dict, repr=False)

    def page_at(self, offset: int) -> int:
        return max(1, bisect.bisect_right(self.page_starts, offset))

    def label(self, page: int | None) -> str | None:
        return f"p. {page}" if self.kind == "pdf" and page else None

    def ref(self, sentence_index: int, max_chars: int = 320) -> dict:
        s = self.sentences[sentence_index]
        text = s.text if len(s.text) <= max_chars else s.text[:max_chars].rsplit(" ", 1)[0] + "…"
        return {"text": text, "page": s.page if self.kind == "pdf" else None, "label": self.label(s.page)}

    def _term_matches(self, term: str) -> tuple[list[tuple[int, int, str]], list[int]]:
        cached = self._term_cache.get(term)
        if cached is None:
            matches = [(m.start(), m.end(), m.group(0)) for m in term_regex(term).finditer(self.text)]
            sentence_ids = sorted({bisect.bisect_right(self._sentence_starts, s) - 1 for s, _, _ in matches} - {-1})
            cached = (matches, sentence_ids)
            self._term_cache[term] = cached
        return cached

    def term_sentences(self, term: str) -> list[int]:
        """Indices of sentences mentioning `term`, in document order."""
        return self._term_matches(term)[1]

    def term_mention_count(self, term: str) -> int:
        return len(self._term_matches(term)[0])

    def term_offsets(self, term: str) -> list[int]:
        return [s for s, _, _ in self._term_matches(term)[0]]

    def term_spans(self, term: str) -> list[tuple[int, int]]:
        return [(s, e) for s, e, _ in self._term_matches(term)[0]]

    def surface_form(self, term: str) -> str:
        """How the document itself most often writes `term` (lowercased unless it is an acronym or CamelCase)."""
        forms = Counter(re.sub(r"\s+", " ", text) for _, _, text in self._term_matches(term)[0])
        if not forms:
            return term
        form = forms.most_common(1)[0][0]
        return form if any(c.isupper() for c in form[1:]) else form.lower()


def classify_headings(lines: list[tuple[str, float, int]]) -> list[dict]:
    """Tier (text, font_size, page) lines into H1/H2/H3 by how much larger they are than the body (modal) size."""
    if not lines:
        return []
    body = Counter(round(size, 1) for _, size, _ in lines).most_common(1)[0][0]
    headings = []
    for text, size, page in lines:
        if size <= body * 1.15 or not 3 < len(text) < 150 or not re.search(r"[A-Za-z]", text):
            continue
        level = 1 if size >= body * 1.6 else 2 if size >= body * 1.3 else 3
        headings.append({"text": text, "level": level, "page": page})
    return headings


def _normalize_line(line: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', line.lower()).strip()


def strip_repeated_lines(pages: list[str], threshold: float = 0.3) -> list[str]:
    """Layer 1: drop lines that repeat verbatim across >30% of pages (running headers and footers)."""
    if len(pages) < 3:
        return list(pages)
    line_page_count = Counter()
    for page in pages:
        line_page_count.update({n for n in map(_normalize_line, page.split('\n')) if len(n) > 2})
    min_pages = max(2, int(len(pages) * threshold))
    blacklist = {line for line, count in line_page_count.items() if count >= min_pages}
    return ['\n'.join(l for l in page.split('\n') if _normalize_line(l) not in blacklist) for page in pages]


def strip_metadata_lines(text: str) -> str:
    """Layer 2: drop page numbers, dates, emails, URLs, DOIs, captions, short numeric labels,
    and table-of-contents / index entries (lines that end in a page reference)."""
    lines = [l for l in text.split('\n') if l.strip()]
    # A page where many lines end in a bare number is a contents or index page.
    toc_page = sum(1 for l in lines if _TRAILING_NUMBER.match(l.strip())) >= max(3, 0.3 * len(lines))
    kept = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped and (_METADATA_PATTERNS.match(stripped)
                         or (len(stripped.split()) <= 3 and any(c.isdigit() for c in stripped))
                         or _PAGE_REF_END.search(stripped)
                         or (toc_page and _TRAILING_NUMBER.match(stripped))):
            continue
        kept.append(line)
    return '\n'.join(kept)


def _ends_segment(prev: str, nxt: str) -> bool:
    if prev[-1] in ".!?:;":
        return not nxt[:1].islower()
    if _LIST_ITEM.match(nxt):
        return True
    # Short line without terminal punctuation followed by a capital: a heading or label.
    return len(prev) < 60 and len(prev.split()) <= 8 and prev[-1] not in ",(" and nxt[:1].isupper()


def reflow(page: str) -> str:
    """Join hard-wrapped PDF lines into sentences while keeping real boundaries as newlines."""
    segments, buf = [], ""
    for raw in page.split("\n"):
        line = raw.strip()
        if not line:
            if buf:
                segments.append(buf)
                buf = ""
        elif not buf:
            buf = line
        elif buf.endswith("-") and line[:1].islower():
            buf = buf[:-1] + line
        elif _ends_segment(buf, line):
            segments.append(buf)
            buf = line
        else:
            buf += " " + line
    if buf:
        segments.append(buf)
    return "\n".join(segments)


def _heading_marks(text: str, pages: list[str], page_starts: list[int], headings: list[dict]) -> dict[int, str]:
    """Locate each heading in the text. Headings with a known page are searched only on that page,
    so a table of contents listing the same titles can't hijack the section boundaries."""
    lower = text.lower()
    marks, cursor = {}, 0
    for h in headings:
        needle = re.sub(r"\s+", " ", h["text"]).strip().lower()
        if len(needle) < 3:
            continue
        page = h.get("page")
        if page and 1 <= page <= len(pages):
            lo = page_starts[page - 1]
            idx = lower.find(needle, lo, lo + len(pages[page - 1]))
        else:
            idx = lower.find(needle, cursor)
        if idx >= 0:
            marks.setdefault(idx, h["text"])
            cursor = idx + len(needle)
    return marks


def _fixed_windows(text: str, size: int) -> list[int]:
    marks, pos = [0], 0
    while pos + size < len(text):
        target = pos + size
        snap = max(text.find(". ", target, target + 300), text.find("\n", target, target + 300))
        pos = snap + 1 if snap > 0 else target
        marks.append(pos)
    return marks


def _section_flags(text: str) -> frozenset:
    lower = text.lower()
    flags = set()
    if _FIGURE_SIGNAL.search(lower):
        flags.add("figure")
    if _EXAMPLE_SIGNAL.search(lower):
        flags.add("example")
    return frozenset(flags)


def clean_pages(pages: list[str]) -> list[str]:
    """Cleaning layers 1-2 plus reflow; keeps one entry per page so page numbers stay valid."""
    return [reflow(strip_metadata_lines(p)) for p in strip_repeated_lines(pages)]


def build_context(pages: list[str], headings: list[dict] | None = None, kind: str = "pdf") -> DocContext:
    headings = headings or []
    cleaned = clean_pages(pages)
    page_starts, pos = [], 0
    for p in cleaned:
        page_starts.append(pos)
        pos += len(p) + 2
    text = "\n\n".join(cleaned)

    marks = _heading_marks(text, cleaned, page_starts, headings)
    starts = [0]
    for m in sorted(marks):
        if m - starts[-1] >= MIN_SECTION_CHARS:
            starts.append(m)
    titles = marks
    if len(starts) < 3:
        starts, titles = _fixed_windows(text, FALLBACK_SECTION_CHARS), {}

    ctx = DocContext(kind=kind, pages=cleaned, text=text, page_starts=page_starts,
                     sections=[], sentences=[], headings=headings)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[s:e]
        ctx.sections.append(Section(start=s, end=e, title=titles.get(s), page=ctx.page_at(s),
                                    text=seg, flags=_section_flags(seg)))

    for seg in re.finditer(r"[^\n]+", text):
        last = 0
        bounds = [m.start() for m in _SENTENCE_BREAK.finditer(seg.group())] + [len(seg.group())]
        for b in bounds:
            raw = seg.group()[last:b]
            lead = len(raw) - len(raw.lstrip())
            sentence = raw.strip()
            if sentence:
                start = seg.start() + last + lead
                ctx.sentences.append(Sentence(text=sentence, start=start, page=ctx.page_at(start),
                                              section=bisect.bisect_right(starts, start) - 1,
                                              words=len(sentence.split())))
            last = b
    ctx._sentence_starts = [s.start for s in ctx.sentences]
    return ctx


# ==========================
# === Concept importance
# ==========================

def _dynamic_stops(ctx: DocContext) -> set:
    """Layer 3: branding (e.g. an institution name) that appears on both the cover and the last page
    but nowhere in the body. Requiring absence from the body keeps real topics that happen to be
    listed in a table of contents or covered in the final section."""
    if len(ctx.pages) < 5:
        return set()
    base = ENGLISH_STOP_WORDS | STATIC_STOPS

    def words(text: str) -> Counter:
        return Counter(w for w in re.findall(r'\b[a-z]{4,}\b', text.lower()) if w not in base)

    first, last = words(ctx.pages[0]), words(ctx.pages[-1])
    body = words(" ".join(ctx.pages[1:-1]))
    return {w for w in first.keys() & last.keys() if first[w] + last[w] >= 3 and w not in body}


def nominate_candidates(ctx: DocContext, max_features: int = 50) -> list[str]:
    """Use TF-IDF over sections to nominate candidate terms; the composite score ranks them afterwards."""
    texts = [s.text for s in ctx.sections]
    n = len(texts)
    stops = list(ENGLISH_STOP_WORDS | STATIC_STOPS | _dynamic_stops(ctx))
    strict = (1, 1.0) if n < 3 else (1, 0.95) if n < 8 else (2, 0.7)
    for min_df, max_df in (strict, (1, 1.0)):
        vectorizer = TfidfVectorizer(stop_words=stops, max_features=max_features, ngram_range=(1, 2),
                                     min_df=min_df, max_df=max_df,
                                     token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b')
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        scores = matrix.mean(axis=0).A1
        ranked = sorted(zip(vectorizer.get_feature_names_out(), scores), key=lambda x: -x[1])
        return [term for term, score in ranked if score > 0]
    return []


def _heading_weight(term: str, headings: list[dict]) -> float:
    words = [w for w in term.lower().split() if len(w) > 2 and w not in ENGLISH_STOP_WORDS]
    patterns = [term_regex(term)] + [term_regex(w) for w in words]
    best = 0.0
    for h in headings:
        if any(p.search(h["text"]) for p in patterns):
            best = max(best, HEADING_MULTIPLIERS.get(h.get("level"), HEADING_MULTIPLIERS[3]))
    return best


def measure_terms(ctx: DocContext, terms: list[str]) -> tuple[dict[str, dict], list[dict]]:
    """Compute the four importance signals for an arbitrary list of terms.

    Works on any term (including ones an LLM proposed), because counts come from
    word-boundary regex matches rather than a fitted vectorizer vocabulary.
    """
    n = len(ctx.sections)
    section_starts = [s.start for s in ctx.sections]
    section_words = np.array([max(1, len(s.text.split())) for s in ctx.sections], dtype=float)
    counts = np.zeros((n, len(terms)))
    for j, term in enumerate(terms):
        for offset in ctx.term_offsets(term):
            counts[bisect.bisect_right(section_starts, offset) - 1, j] += 1

    present = counts > 0
    df = present.sum(axis=0)
    tfidf = (counts / section_words[:, None]) * (np.log((1 + n) / (1 + df)) + 1)
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    frequency = (tfidf / row_norms).mean(axis=0)
    frequency = frequency / frequency.max() if frequency.max() > 0 else frequency
    spread = df / n
    heading = np.array([_heading_weight(t, ctx.headings) for t in terms]) / max(HEADING_MULTIPLIERS.values())

    # Co-occurrence: entry [i, j] = number of sections containing both terms.
    binary = present.astype(float)
    co = binary.T @ binary
    graph = nx.Graph()
    graph.add_nodes_from(range(len(terms)))
    edges = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            weight = co[i, j] / n
            if weight >= GRAPH_EDGE_MIN:
                graph.add_edge(i, j, weight=weight)
                if weight >= DISPLAY_EDGE_MIN:
                    edges.append({"source": terms[i], "target": terms[j], "weight": round(float(weight), 3)})
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pagerank = {i: 1.0 / len(terms) for i in range(len(terms))}
    max_pr = max(pagerank.values()) if pagerank else 1.0
    centrality = np.array([pagerank.get(i, 0.0) / max_pr for i in range(len(terms))])

    metrics = {}
    for j, term in enumerate(terms):
        sections_with_term = np.nonzero(present[:, j])[0]
        boost, boosts = 1.0, []
        sentence_ids = ctx.term_sentences(term)[:300]
        if any(is_definitional(ctx.sentences[i].text, term) for i in sentence_ids):
            boost *= EXAM_BOOSTS["definition"]
            boosts.append("definition")
        for flag in ("figure", "example"):
            if any(flag in ctx.sections[s].flags for s in sections_with_term):
                boost *= EXAM_BOOSTS[flag]
                boosts.append(flag)
        base = (WEIGHTS["frequency"] * frequency[j] + WEIGHTS["spread"] * spread[j]
                + WEIGHTS["heading"] * heading[j] + WEIGHTS["centrality"] * centrality[j])
        metrics[term] = {
            "frequency": round(float(frequency[j]), 3),
            "spread": round(float(spread[j]), 3),
            "heading": round(float(heading[j]), 3),
            "centrality": round(float(centrality[j]), 3),
            "exam_boost": round(boost, 3),
            "exam_signals": boosts,
            "composite": float(base * boost) if df[j] else 0.0,
        }
    return metrics, edges


def drop_fragments(ctx: DocContext, terms: list[str]) -> list[str]:
    """Drop terms that mostly occur inside a longer candidate ('descent' inside 'gradient descent')."""
    kept = []
    for term in terms:
        offsets = ctx.term_offsets(term)
        if not offsets:
            continue
        rx, n_words = term_regex(term), len(term.split())
        spans = sorted(span for longer in terms
                       if len(longer.split()) > n_words and rx.search(longer)
                       for span in ctx.term_spans(longer))
        if spans:
            starts = [s for s, _ in spans]
            inside = 0
            for o in offsets:
                i = bisect.bisect_right(starts, o) - 1
                inside += i >= 0 and o < spans[i][1]
            if inside / len(offsets) >= FRAGMENT_SHARE:
                continue
        kept.append(term)
    return kept


def tier_for(score: float) -> str:
    if score > CORE_THRESHOLD:
        return "core"
    if score > SUPPORTING_THRESHOLD:
        return "supporting"
    return "minor"


def _rank(ctx: DocContext, terms: list[str], endorsed: set[str], limit: int) -> dict:
    terms = drop_fragments(ctx, terms)
    if not terms:
        return {"topics": [], "edges": [], "study_priority": ""}
    metrics, edges = measure_terms(ctx, terms)
    scored = []
    for term in terms:
        endorsed_term = term_key(term) in endorsed
        score = metrics[term]["composite"] * (LLM_ENDORSEMENT_BOOST if endorsed_term else 1.0)
        if score > 0:
            scored.append((term, score, endorsed_term))
    scored.sort(key=lambda x: -x[1])

    final, seen = [], set()
    for term, score, endorsed_term in scored:
        key = term_key(term)
        if key not in seen:
            seen.add(key)
            final.append((term, score, endorsed_term))
        if len(final) == limit:
            break
    if not final:
        return {"topics": [], "edges": [], "study_priority": ""}

    heading_words = {w for h in ctx.headings for w in re.findall(r"[a-z0-9]+", h["text"].lower()) if len(w) > 2}
    top = final[0][1]
    topics = []
    for term, score, endorsed_term in final:
        m = metrics[term]
        norm = round(score / top, 3)
        topics.append({
            "term": term,
            "score": norm,
            "tier": tier_for(norm),
            "is_heading": any(w in heading_words for w in term.lower().split()),
            "llm_endorsed": endorsed_term,
            **{k: m[k] for k in ("frequency", "spread", "heading", "centrality", "exam_boost", "exam_signals")},
        })
    names = {t["term"] for t in topics}
    total = sum(t["score"] for t in topics)
    coverage = round(sum(t["score"] for t in topics[:5]) / total * 100) if total else 0
    return {
        "topics": topics,
        "edges": [e for e in edges if e["source"] in names and e["target"] in names],
        "study_priority": f"Top 5 topics cover ~{coverage}% of this document",
    }


def analyze_concepts(ctx: DocContext) -> dict:
    candidates = nominate_candidates(ctx)
    if not candidates:
        return {"topics": [], "edges": [], "study_priority": ""}
    result = _rank(ctx, candidates, endorsed=set(), limit=MAX_TOPICS)
    log.info("Concept analysis: %d candidates -> %d topics, %d edges",
             len(candidates), len(result["topics"]), len(result["edges"]))
    return result


# ==========================
# === LLM topic refinement
# ==========================

def build_llm_digest(ctx: DocContext, budget: int = 8000) -> str:
    """Condense the whole document into `budget` chars: the outline plus an excerpt from every section,
    so the LLM sees the full scope of the document instead of only its first pages."""
    parts = []
    outline = [("  " if h.get("level") == 2 else "") + h["text"] for h in ctx.headings if h.get("level", 3) <= 2]
    if outline:
        parts.append("Outline:\n" + "\n".join(dict.fromkeys(outline))[:1500])
    header, separator = "Excerpts from across the document:\n", "\n...\n"
    remaining = budget - sum(len(p) + 2 for p in parts) - len(header)
    sections = [s for s in ctx.sections if s.text.strip()]
    if sections and remaining > 0:
        per = max(250, remaining // len(sections) - len(separator))
        count = max(1, min(len(sections), remaining // (per + len(separator))))
        step = len(sections) / count
        chosen = [sections[int(i * step)] for i in range(count)]
        if chosen[-1] is not sections[-1]:
            chosen[-1] = sections[-1]
        excerpts = []
        for s in chosen:
            excerpt = s.text.strip()[:per]
            cut = excerpt.rfind(". ")
            excerpts.append(excerpt[:cut + 1] if cut > per // 2 else excerpt)
        parts.append(header + separator.join(excerpts))
    return "\n\n".join(parts)[:budget]


def parse_llm_topics(raw: str) -> list[str]:
    """Extract a list of topic strings from an LLM reply (JSON object, bare array, or array inside prose)."""
    def strings(values):
        return [str(v).strip() for v in values if isinstance(v, str) and len(v.strip()) > 1]

    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return strings(parsed)
        if isinstance(parsed, dict):
            for key in ("topics", "concepts"):
                if isinstance(parsed.get(key), list):
                    return strings(parsed[key])
            for value in parsed.values():
                if isinstance(value, list):
                    return strings(value)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]', raw, re.DOTALL)
    if match:
        try:
            return strings(json.loads(match.group()))
        except json.JSONDecodeError:
            pass
    log.warning("Could not parse LLM topics from: %r", raw[:200])
    return []


def _clean_llm_term(raw: str) -> str | None:
    term = re.sub(r"\s+", " ", raw.strip().strip("\"'“”‘’.,;:•-")).strip()
    if not 2 <= len(term) <= 60 or len(term.split()) > 6 or not re.search(r"[A-Za-z]", term):
        return None
    if term.lower() in ENGLISH_STOP_WORDS:
        return None
    return term


def merge_llm_topics(ctx: DocContext, topics: list[dict], llm_terms: list[str]) -> dict:
    """Merge LLM-proposed concepts into the measured ranking.

    The LLM only nominates: a proposal that never appears in the document is rejected,
    and every accepted term is scored by the same four-signal engine. Concepts the LLM
    names get a 2x endorsement before the scores are renormalised.
    """
    existing = [t["term"] for t in topics]
    existing_keys = {term_key(t) for t in existing}
    endorsed = {term_key(t["term"]) for t in topics if t.get("llm_endorsed")}
    added, rejected = [], []
    for raw in llm_terms:
        term = _clean_llm_term(raw)
        if not term:
            continue
        key = term_key(term)
        if key in existing_keys:
            endorsed.add(key)
            continue
        container = next((e for e in existing
                          if len(e.split()) > len(term.split()) and term_regex(term).search(e)), None)
        if container:
            endorsed.add(term_key(container))
            continue
        if key in {term_key(a) for a in added}:
            continue
        if ctx.term_mention_count(term) == 0:
            rejected.append(term)
            continue
        added.append(ctx.surface_form(term))
        endorsed.add(key)

    result = _rank(ctx, existing + added, endorsed, MAX_MERGED_TOPICS)
    kept = {t["term"] for t in result["topics"]}
    result["llm"] = {
        "endorsed": sum(1 for t in result["topics"] if t["llm_endorsed"]),
        "added": [t for t in added if t in kept],
        "rejected": rejected,
    }
    log.info("LLM merge: %d endorsed, %d added, %d rejected as not in document",
             result["llm"]["endorsed"], len(result["llm"]["added"]), len(rejected))
    return result
