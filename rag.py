"""Retrieval pieces: page-aware chunking, BM25 keyword search, rank fusion, and chat-context assembly."""

import bisect
import math
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from analysis import singular

_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|thanks|thank you|thank u|thx|ty|ok|okay|cool|great|nice|bye|goodbye"
    r"|good (?:morning|afternoon|evening|night))(?: (?:lexo|there|so much|a lot|again))*$")


def _chunk_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    spans, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            break_point = max(text.rfind('. ', start + chunk_size // 2, end),
                              text.rfind('\n', start + chunk_size // 2, end))
            if break_point > start:
                end = break_point + 1
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s > 20:
            spans.append((s, e))
        if end >= n:
            break
        # Start the overlap at a sentence boundary if there is one, else at least on a word boundary.
        window = end - overlap
        boundaries = [i for i in (text.find('. ', window, end), text.find('\n', window, end)) if i != -1]
        if boundaries:
            start = min(boundaries) + 1
        else:
            space = text.find(' ', window, end)
            start = space + 1 if space != -1 else window
    return spans


def chunk_pages(pages: list[str], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split the document into overlapping chunks, each tagged with the (1-based) pages it spans."""
    starts, pos = [], 0
    for page in pages:
        starts.append(pos)
        pos += len(page) + 2
    text = "\n\n".join(pages)
    return [{
        "text": text[s:e],
        "page_start": bisect.bisect_right(starts, s),
        "page_end": bisect.bisect_right(starts, e - 1),
    } for s, e in _chunk_spans(text, chunk_size, overlap)]


def tokenize(text: str) -> list[str]:
    return [singular(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in ENGLISH_STOP_WORDS]


class BM25:
    """Okapi BM25 over pre-tokenized chunks."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_length = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        terms = list(dict.fromkeys(q for q in query if q in self.idf))
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = 1 - self.b + self.b * length / self.avg_length
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(score)
        return out

    def top(self, query: list[str], k: int) -> list[int]:
        scores = self.scores(query)
        return [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists: each list contributes 1 / (k + rank) for every item it ranks."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def source_label(chunk: dict, kind: str, index: int) -> str:
    if kind != "pdf":
        return f"Excerpt {index + 1}"
    a, b = chunk["page_start"], chunk["page_end"]
    return f"p. {a}" if a == b else f"pp. {a}–{b}"


def build_context_block(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['label']}]\n{s['text']}" for s in sources)


def fit_history(history: list[dict], budget_chars: int) -> list[dict]:
    """Keep the most recent messages that fit the budget, starting on a user turn."""
    kept, used = [], 0
    for message in reversed(history):
        used += len(message["content"])
        if used > budget_chars:
            break
        kept.append(message)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def is_small_talk(message: str) -> bool:
    """Only whole-message greetings/thanks count; a short question like 'Define overfitting' does not."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", message.lower())).strip()
    return bool(_SMALL_TALK.match(cleaned))


def small_talk_reply(message: str) -> str:
    lower = message.lower()
    if any(w in lower for w in ("thank", "thx")) or re.search(r"\bty\b", lower):
        return "You're welcome. Ask me anything else about the document whenever you're ready."
    if any(w in lower for w in ("bye", "night")):
        return "Good luck with your studying. Your progress on this document is saved."
    if re.match(r"\s*(?:ok|okay|cool|great|nice)\b", lower):
        return "Sounds good. What would you like to explore next?"
    return "Hey! I'm Lexo, your document analyst. Ask me anything about the document you've uploaded."
