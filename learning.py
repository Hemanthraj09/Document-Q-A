"""Evidence lookups and the prerequisite learning path, built on the concept analysis.

Everything here is measured from the document text: every node, edge and ordering
decision can be traced back to the sentences that produced it.
"""

import re
from functools import lru_cache

import networkx as nx

from analysis import WEIGHTS, DocContext, is_definitional, term_body, term_key, term_regex

SUBSTANTIVE_WORDS = 6
INTRO_WINDOW = 3
MAX_SHARED_SENTENCES = 150

_ISA = (r"\s+(?:is|are)\s+(?:an?|the)\s+(?:(?:type|kind|form|class|variant|subset|subclass|extension"
        r"|special case|instance)\s+of\s+(?:an?\s+|the\s+)?)?(?:[\w-]+\s+){0,2}?")
_USES = (r"\s+(?:[\w-]+\s+){0,2}?(?:uses?|relies on|rely on|depends on|depend on|builds? on|(?:is|are) built on"
         r"|(?:is|are) based on|extends?|requires?|leverages?|employs?|(?:is|are) derived from|combines?"
         r"|applies|apply)\s+(?:[\w-]+\s+){0,4}?")
_USED_FOR = r"\s+(?:is|are)\s+(?:used|required|needed|essential)\s+(?:in|for|by|to)\s+(?:[\w-]+\s+){0,4}?"


@lru_cache(maxsize=8192)
def _dependency_regexes(dependent: str, prerequisite: str) -> tuple[re.Pattern, ...]:
    b, a = term_body(dependent), term_body(prerequisite)
    return (
        re.compile(b + _ISA + a, re.IGNORECASE),        # "a transformer is a type of neural network"
        re.compile(b + _USES + a, re.IGNORECASE),       # "backpropagation relies on the chain rule"
        re.compile(a + _USED_FOR + b, re.IGNORECASE),   # "gradients are used by gradient descent"
    )


def depends_on(sentence: str, dependent: str, prerequisite: str) -> bool:
    """True if the sentence states that `dependent` builds on `prerequisite`."""
    return any(rx.search(sentence) for rx in _dependency_regexes(dependent, prerequisite))


def mentions_outside(text: str, term: str, other: str) -> bool:
    """True if `term` occurs in `text` somewhere other than inside an occurrence of `other`
    (so 'network' isn't counted as mentioned just because 'neural network' is)."""
    other_spans = [m.span() for m in term_regex(other).finditer(text)]
    return any(not any(s <= m.start() and m.end() <= e for s, e in other_spans)
               for m in term_regex(term).finditer(text))


def _is_specialization(general: str, specific: str) -> bool:
    """'neural network' specializes 'network' (same head noun); 'learning rate' does not specialize 'learning'."""
    g, s = term_key(general).split(), term_key(specific).split()
    return len(s) > len(g) and s[-len(g):] == g


def _introduction(ctx: DocContext, term: str) -> dict | None:
    """Where a concept is introduced: its first definition if there is one, else its first real mention."""
    occurrences = [i for i in ctx.term_sentences(term) if ctx.sentences[i].words >= SUBSTANTIVE_WORDS]
    if not occurrences:
        return None
    definitions = [i for i in occurrences if is_definitional(ctx.sentences[i].text, term)]
    intro = definitions[0] if definitions else occurrences[0]
    section = ctx.sentences[intro].section
    window = [i for i in range(intro, min(intro + INTRO_WINDOW, len(ctx.sentences)))
              if ctx.sentences[i].section == section]
    return {
        "occurrences": occurrences,
        "definitions": definitions,
        "intro": intro,
        "context": sorted(set(window) | set(definitions[:3])),
    }


def build_learning_path(ctx: DocContext, topics: list[dict]) -> dict:
    """Order concepts so prerequisites come first.

    An edge A -> B ("learn A before B") is added when:
      * B's introduction (its definition and the sentences right after it) uses A, and A was introduced earlier;
      * a sentence states the dependency directly ("B is a type of A", "B relies on A"); or
      * B is a more specific version of A by name ("neural network" -> "network").
    Minor-tier concepts never act as prerequisites. Cycles are broken by dropping the weakest edge,
    redundant edges are removed with a
    transitive reduction, and the order is a topological sort that puts concepts which
    unlock the most others first, falling back to document order.
    """
    terms = [t["term"] for t in topics]
    if not terms:
        return {"steps": [], "edges": [], "levels": 0}
    tiers = {t["term"]: t.get("tier") for t in topics}
    info = {t: _introduction(ctx, t) for t in terms}
    occurrence_sets = {t: set(i["occurrences"]) for t, i in info.items() if i}
    intro_index = {t: info[t]["intro"] if info[t] else float("inf") for t in terms}

    graph = nx.DiGraph()
    graph.add_nodes_from(terms)
    for b in terms:
        if not info[b]:
            continue
        for a in terms:
            # Minor concepts are generic vocabulary ("model", "algorithm") that most definitions use;
            # they stay in the path but never gate another concept.
            if a == b or not info[a] or tiers[a] == "minor":
                continue
            weight, reason, evidence = 0.0, None, None
            if intro_index[a] < intro_index[b]:
                hits = [i for i in info[b]["context"] if mentions_outside(ctx.sentences[i].text, a, b)]
                if hits:
                    weight, reason, evidence = float(len(hits)), "introduced_using", hits[0]
            shared = sorted(occurrence_sets[a] & occurrence_sets[b])[:MAX_SHARED_SENTENCES]
            stated = next((i for i in shared if depends_on(ctx.sentences[i].text, b, a)), None)
            if stated is not None:
                weight, reason, evidence = weight + 2.0, "stated_dependency", stated
            if not weight and _is_specialization(a, b):
                weight, reason, evidence = 1.0, "specialization", info[b]["intro"]
            if weight:
                graph.add_edge(a, b, weight=weight, reason=reason, evidence=evidence)

    while True:
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            break
        # Drop the weakest edge; among equals, prefer dropping one that runs against document order.
        u, v = min(((e[0], e[1]) for e in cycle),
                   key=lambda e: (graph.edges[e]["weight"], intro_index[e[0]] < intro_index[e[1]]))
        graph.remove_edge(u, v)

    reduced = nx.transitive_reduction(graph)
    reduced.add_nodes_from(graph.nodes)
    for u, v in reduced.edges:
        reduced.edges[u, v].update(graph.edges[u, v])

    unlocks_total = {n: len(nx.descendants(reduced, n)) for n in reduced}
    order = list(nx.lexicographical_topological_sort(
        reduced, key=lambda n: (-unlocks_total[n], intro_index[n], n)))
    level = {}
    for n in order:
        preds = list(reduced.predecessors(n))
        level[n] = 1 + max(level[p] for p in preds) if preds else 0

    by_term = {t["term"]: t for t in topics}
    steps = [{
        "term": n,
        "level": level[n],
        "prerequisites": sorted(reduced.predecessors(n), key=order.index),
        "unlocks": sorted(reduced.successors(n), key=order.index),
        "unlocks_total": unlocks_total[n],
        "introduced": ctx.ref(info[n]["intro"]) if info[n] else None,
        "tier": by_term[n]["tier"],
        "score": by_term[n]["score"],
    } for n in order]
    edges = [{
        "source": u,
        "target": v,
        "weight": d["weight"],
        "reason": d["reason"],
        "evidence": ctx.ref(d["evidence"]) if d["evidence"] is not None else None,
    } for u, v, d in reduced.edges(data=True)]
    return {"steps": steps, "edges": edges, "levels": (max(level.values()) + 1) if level else 0}


def find_topic(topics: list[dict], term: str) -> dict | None:
    key = term_key(term)
    return next((t for t in topics if term_key(t["term"]) == key), None)


def concept_evidence(ctx: DocContext, topic: dict, path: dict, limit: int = 4) -> dict:
    """Everything the UI needs to justify a node: score breakdown, definitions, mentions, prerequisites."""
    term = topic["term"]
    occurrences = ctx.term_sentences(term)
    definitions = [i for i in occurrences
                   if ctx.sentences[i].words >= 4 and is_definitional(ctx.sentences[i].text, term)][:2]
    mentions, seen_sections = [], set()
    for i in occurrences:
        s = ctx.sentences[i]
        if i in definitions or s.words < SUBSTANTIVE_WORDS or s.section in seen_sections:
            continue
        seen_sections.add(s.section)
        mentions.append(i)
        if len(mentions) == limit:
            break
    pages = sorted({ctx.sentences[i].page for i in occurrences}) if ctx.kind == "pdf" else []
    step_index = next((i for i, s in enumerate(path.get("steps", [])) if s["term"] == term), None)
    return {
        "term": term,
        "breakdown": {k: topic.get(k) for k in ("score", "tier", "frequency", "spread", "heading", "centrality",
                                                 "exam_boost", "exam_signals", "llm_endorsed", "is_heading")},
        "weights": WEIGHTS,
        "definitions": [ctx.ref(i) for i in definitions],
        "mentions": [ctx.ref(i) for i in mentions],
        "stats": {
            "mentions": ctx.term_mention_count(term),
            "sentences": len(occurrences),
            "sections": len({ctx.sentences[i].section for i in occurrences}),
            "total_sections": len(ctx.sections),
            "pages": pages,
        },
        "prerequisites": [e for e in path.get("edges", []) if e["target"] == term],
        "unlocks": [e for e in path.get("edges", []) if e["source"] == term],
        "path_position": step_index + 1 if step_index is not None else None,
    }


def relation_evidence(ctx: DocContext, a: str, b: str, path: dict, limit: int = 5) -> dict:
    """Why two concepts are linked in the graph: the sentences and sections where they meet."""
    sa, sb = set(ctx.term_sentences(a)), set(ctx.term_sentences(b))
    together = sorted(sa & sb)
    sections_a = {ctx.sentences[i].section for i in sa}
    sections_b = {ctx.sentences[i].section for i in sb}
    shared_sections = sorted(sections_a & sections_b)

    section_pairs = []
    if not together:
        for sec in shared_sections[:3]:
            ia = next(i for i in sorted(sa) if ctx.sentences[i].section == sec)
            ib = next(i for i in sorted(sb) if ctx.sentences[i].section == sec)
            section_pairs.append({"section": ctx.sections[sec].title, "a": ctx.ref(ia), "b": ctx.ref(ib)})

    prerequisite = next((e for e in path.get("edges", [])
                         if {e["source"], e["target"]} == {a, b}), None)
    n = len(ctx.sections)
    return {
        "a": a,
        "b": b,
        "weight": round(len(shared_sections) / n, 3) if n else 0.0,
        "shared_sentences": len(together),
        "shared_sections": len(shared_sections),
        "total_sections": n,
        "sentences": [ctx.ref(i) for i in together[:limit]],
        "section_pairs": section_pairs,
        "prerequisite": prerequisite,
    }
