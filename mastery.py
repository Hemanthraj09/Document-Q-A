"""Per-concept mastery tracking, spaced-repetition scheduling, quiz targeting and grading helpers."""

import difflib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis import singular, term_regex

ALPHA = 0.5               # how far one quiz result moves the mastery estimate
MASTERED = 0.8
PREREQUISITE_READY = 0.5  # a prerequisite counts as "ready" once mastery reaches this
CORRECT = 0.6

_DONT_KNOW = re.compile(r"^\s*(?:i\s+)?(?:don'?t|do not)\s+know\b|^\s*(?:no idea|idk|pass|skip|\?+)\s*[.!]*\s*$",
                        re.IGNORECASE)


def concept_key(term: str) -> str:
    return " ".join(term.lower().split())


def new_state() -> dict:
    return {"mastery": 0.0, "attempts": 0, "correct": 0, "exposures": 0, "ease": 2.5,
            "interval": 0, "reps": 0, "due": None, "last_score": None, "last_seen": None}


def status(state: dict) -> str:
    if state["attempts"] and state["mastery"] >= MASTERED:
        return "mastered"
    if state["attempts"]:
        return "learning"
    if state["exposures"]:
        return "explored"
    return "unseen"


def public_state(state: dict) -> dict:
    return {**state, "status": status(state)}


def record_exposure(state: dict, now: datetime) -> None:
    state["exposures"] += 1
    state["last_seen"] = now.isoformat()


def apply_score(state: dict, score: float, now: datetime) -> None:
    """Exponential moving average for mastery, SM-2 for when to review next."""
    state["attempts"] += 1
    if score >= CORRECT:
        state["correct"] += 1
    state["mastery"] = round(state["mastery"] + ALPHA * (score - state["mastery"]), 4)

    quality = round(score * 5)
    if quality < 3:
        state["reps"], state["interval"] = 0, 1
    else:
        state["reps"] += 1
        state["interval"] = (1 if state["reps"] == 1 else 6 if state["reps"] == 2
                             else round(state["interval"] * state["ease"]))
    state["ease"] = round(max(1.3, state["ease"] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)), 3)
    state["due"] = (now + timedelta(days=state["interval"])).isoformat()
    state["last_score"] = round(score, 3)
    state["last_seen"] = now.isoformat()


def is_due(state: dict, now: datetime) -> bool:
    return state["due"] is None or datetime.fromisoformat(state["due"]) <= now


class MasteryStore:
    """Mastery state per document, persisted as one JSON file per document."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _path(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def _load(self, doc_id: str) -> dict:
        if doc_id not in self._cache:
            path = self._path(doc_id)
            self._cache[doc_id] = json.loads(path.read_text(encoding="utf-8"))["concepts"] if path.exists() else {}
        return self._cache[doc_id]

    def _save(self, doc_id: str) -> None:
        path = self._path(doc_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"concepts": self._cache[doc_id]}, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def states(self, doc_id: str, terms: list[str]) -> dict[str, dict]:
        with self._lock:
            concepts = self._load(doc_id)
            return {t: dict(concepts.get(concept_key(t)) or new_state()) for t in terms}

    def record_exposure(self, doc_id: str, terms: list[str], now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            concepts = self._load(doc_id)
            for term in terms:
                record_exposure(concepts.setdefault(concept_key(term), new_state()), now)
            self._save(doc_id)

    def apply_score(self, doc_id: str, term: str, score: float, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            concepts = self._load(doc_id)
            state = concepts.setdefault(concept_key(term), new_state())
            apply_score(state, score, now)
            self._save(doc_id)
            return dict(state)


def summarize(topics: list[dict], states: dict[str, dict]) -> dict:
    counts = {"unseen": 0, "explored": 0, "learning": 0, "mastered": 0}
    for t in topics:
        counts[status(states[t["term"]])] += 1
    core = [t for t in topics if t["tier"] == "core"]
    return {
        **counts,
        "total": len(topics),
        "core_total": len(core),
        "core_mastered": sum(1 for t in core if status(states[t["term"]]) == "mastered"),
    }


def pick_quiz_target(topics: list[dict], states: dict[str, dict], path: dict,
                     now: datetime) -> tuple[str, str] | None:
    """Pick the concept to quiz next: important, not yet mastered, due for review, and with its
    prerequisites in place. Returns (term, human-readable reason)."""
    prerequisites = {}
    for e in path.get("edges", []):
        prerequisites.setdefault(e["target"], []).append(e["source"])
    order = {s["term"]: i for i, s in enumerate(path.get("steps", []))}

    best, best_key = None, None
    for t in topics:
        term, state = t["term"], states[t["term"]]
        due = is_due(state, now)
        blocked_by = [p for p in prerequisites.get(term, []) if states[p]["mastery"] < PREREQUISITE_READY]
        priority = t["score"] * (1 - state["mastery"]) * (1.0 if due else 0.25) * (0.35 if blocked_by else 1.0)
        key = (priority, -order.get(term, len(order)))
        if best_key is None or key > best_key:
            best, best_key = (t, state, due, blocked_by), key
    if best is None:
        return None

    topic, state, due, blocked_by = best
    term = topic["term"]
    parts = [f"{topic['tier'].capitalize()} concept (importance {topic['score']:.0%})"]
    if state["attempts"] == 0:
        parts.append("not quizzed yet")
    else:
        parts.append(f"current mastery {state['mastery']:.0%}")
        if due:
            parts.append("due for review")
    if prerequisites.get(term) and not blocked_by:
        parts.append("its prerequisites are in place")
    return term, ", ".join(parts)


def prerequisite_gap(term: str, states: dict[str, dict], path: dict) -> list[str]:
    return [e["source"] for e in path.get("edges", [])
            if e["target"] == term and states.get(e["source"], new_state())["mastery"] < PREREQUISITE_READY]


def make_cloze(sentence: str, term: str) -> str | None:
    """Fill-in-the-blank question from a sentence; used when no LLM is available."""
    rx = term_regex(term)
    if not rx.search(sentence):
        return None
    blanked = rx.sub("_____", sentence)
    return blanked if len(blanked.replace("_____", "").split()) >= 5 else None


def parse_quiz(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    question, answer = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
    if not 10 <= len(question) <= 400 or not answer:
        return None
    return {"question": question, "answer": answer}


def parse_grade(raw: str) -> tuple[float | None, str | None]:
    try:
        data = json.loads(raw)
        score = float(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None
    if score > 1:
        score /= 100
    feedback = str(data.get("feedback", "")).strip() or None
    return min(1.0, max(0.0, score)), feedback


def is_dont_know(answer: str) -> bool:
    """A short 'I don't know' style reply; longer answers that start that way still get graded."""
    return len(answer.split()) <= 5 and bool(_DONT_KNOW.search(answer))


def _normalize_answer(text: str) -> str:
    return " ".join(singular(w) for w in re.findall(r"[a-z0-9]+", text.lower()))


def grade_exact(answer: str, expected: str) -> float:
    """Grade a fill-in-the-blank answer: exact (plural-insensitive) match, or a near-miss typo."""
    a, e = _normalize_answer(answer), _normalize_answer(expected)
    if not a:
        return 0.0
    if a == e or term_regex(expected).search(answer):
        return 1.0
    if difflib.SequenceMatcher(None, a, e).ratio() >= 0.8:
        return 0.75
    return 0.0


def similarity_score(cosine: float) -> float:
    """Map embedding cosine similarity onto 0..1: <=0.25 is unrelated, >=0.75 is the same idea."""
    return min(1.0, max(0.0, (cosine - 0.25) / 0.5))


def blend(llm_score: float | None, similarity: float) -> float:
    return similarity if llm_score is None else 0.6 * llm_score + 0.4 * similarity


def verdict(score: float) -> str:
    return "correct" if score >= MASTERED else "partial" if score >= 0.5 else "incorrect"
