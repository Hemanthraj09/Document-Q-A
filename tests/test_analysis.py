import pytest

import sample_doc
from analysis import (_dynamic_stops, analyze_concepts, build_context, build_llm_digest, classify_headings,
                      drop_fragments,
                      is_definitional, measure_terms, merge_llm_topics, parse_llm_topics, reflow,
                      strip_metadata_lines, strip_repeated_lines, term_key, term_regex, tier_for)


@pytest.fixture(scope="module")
def ctx():
    return build_context(sample_doc.pages(), sample_doc.headings())


def test_term_regex_matches_whole_words_plurals_and_hyphens():
    assert term_regex("neural network").search("Neural networks learn")
    assert term_regex("neural network").search("a neural-network model")
    assert term_regex("study").search("two studies")
    assert not term_regex("art").search("a partial result")
    assert not term_regex("descent gradient").search("Gradient Descent\nGradient descent is")


def test_term_key_folds_plurals():
    assert term_key("Neural Networks") == term_key("neural network")
    assert term_key("processes") == term_key("process")


def test_is_definitional():
    assert is_definitional("Gradient descent is an optimization algorithm.", "gradient descent")
    assert is_definitional("This effect is known as overfitting.", "overfitting")
    assert not is_definitional("We then apply gradient descent to the loss.", "gradient descent")


def test_classify_headings_tiers_by_font_size():
    lines = [("Body text line", 10.0, 1)] * 20 + [
        ("Chapter One", 20.0, 1), ("Section A", 14.0, 2), ("Sub point", 12.0, 2), ("42", 20.0, 3)]
    assert {h["text"]: h["level"] for h in classify_headings(lines)} == {"Chapter One": 1, "Section A": 2, "Sub point": 3}


def test_strip_repeated_lines_removes_running_headers():
    pages = [f"Course Notes Header\nContent about topic {i} here." for i in range(5)]
    cleaned = strip_repeated_lines(pages)
    assert all("Course Notes Header" not in p for p in cleaned)
    assert "topic 3" in cleaned[3]


def test_reflow_joins_wrapped_lines_and_hyphenation():
    page = "Gradient descent is an\noptimiza-\ntion algorithm.\nOverview\nIt updates weights."
    assert reflow(page).split("\n") == ["Gradient descent is an optimization algorithm.", "Overview",
                                        "It updates weights."]


def test_table_of_contents_does_not_hijack_sections(ctx):
    titled = {s.title: s.page for s in ctx.sections if s.title}
    assert titled["Gradient Descent"] == 3
    assert titled["Regularization"] == 7


def test_heading_search_is_scoped_to_the_heading_page():
    pages = ["Overview\nThis unit covers gradient descent and regularization in some depth today.",
             "Gradient Descent\nGradient descent is an optimization algorithm that minimizes a loss.",
             "Regularization\nRegularization is a technique that adds a penalty to the loss function."]
    headings = [{"text": "Gradient Descent", "level": 2, "page": 2}, {"text": "Regularization", "level": 2, "page": 3}]
    titled = {s.title: s.page for s in build_context(pages, headings).sections if s.title}
    assert titled == {"Gradient Descent": 2, "Regularization": 3}


def test_toc_and_index_lines_are_stripped():
    toc = "Contents\nIntroduction to models 1\nGradient Descent 15\nNeural Networks 27\nRegularization 40"
    assert strip_metadata_lines(toc).split("\n") == ["Contents"]
    assert strip_metadata_lines("Loss functions and scoring (page 2)") == ""
    assert strip_metadata_lines("Gradient descent ........ 15") == ""
    assert "released in 2015" in strip_metadata_lines("The library was released in 2015 and is popular.")


def test_sentences_know_their_page_and_section(ctx):
    s = next(s for s in ctx.sentences if s.text.startswith("A neural network is"))
    assert s.page == 4
    assert ctx.sections[s.section].title == "Neural Networks"


def test_running_header_and_page_numbers_are_stripped(ctx):
    assert sample_doc.HEADER not in ctx.text


def test_cooccurrence_edges_match_brute_force(ctx):
    terms = ["gradient descent", "neural network", "loss function", "overfitting"]
    _, edges = measure_terms(ctx, terms)
    n = len(ctx.sections)
    assert edges
    for e in edges:
        both = sum(1 for s in ctx.sections
                   if term_regex(e["source"]).search(s.text) and term_regex(e["target"]).search(s.text))
        assert e["weight"] == round(both / n, 3)


def test_exam_boost_applies_once_per_signal():
    pages = [f"Topic {chr(65 + i)} overview\nThe kernel trick is shown in figure {i} and matters for topic {i}."
             for i in range(8)]
    headings = [{"text": f"Topic {chr(65 + i)} overview", "level": 2, "page": i + 1} for i in range(8)]
    metrics, _ = measure_terms(build_context(pages, headings), ["kernel trick"])
    assert metrics["kernel trick"]["exam_boost"] == pytest.approx(1.2)


def test_analyze_concepts_scores_are_normalized_and_tiered(ctx):
    topics = analyze_concepts(ctx)["topics"]
    assert topics
    assert topics[0]["score"] == 1.0
    assert all(0 < t["score"] <= 1 for t in topics)
    assert all(t["tier"] == tier_for(t["score"]) for t in topics)
    assert len({term_key(t["term"]) for t in topics}) == len(topics)


def test_branding_stopwords_spare_topics_on_edge_pages(ctx):
    terms = [t["term"] for t in analyze_concepts(ctx)["topics"]]
    assert "regularization" in terms
    assert "loss function" in terms
    pages = ["Acme University course pack", *[f"Body page {i} on gradient descent." for i in range(5)],
             "Printed by Acme University. Acme University press."]
    assert "acme" in _dynamic_stops(build_context(pages, []))


def test_fragments_of_longer_terms_are_dropped(ctx):
    kept = drop_fragments(ctx, ["gradient descent", "descent", "neural network", "neural", "overfitting"])
    assert kept == ["gradient descent", "neural network", "overfitting"]
    terms = [t["term"] for t in analyze_concepts(ctx)["topics"]]
    assert "descent" not in terms and "neural" not in terms


def test_parse_llm_topics_handles_objects_arrays_and_prose():
    assert parse_llm_topics('{"topics": ["A b", "Cd", "x"]}') == ["A b", "Cd"]
    assert parse_llm_topics('["x1", "y2"]') == ["x1", "y2"]
    assert parse_llm_topics('Sure! Here: ["alpha", "beta"] hope it helps') == ["alpha", "beta"]
    assert parse_llm_topics("no json here") == []


def test_merge_rejects_hallucinations_and_keeps_specific_variants(ctx):
    topics = [{"term": "data"}, {"term": "gradient descent"}]
    result = merge_llm_topics(ctx, topics, ["Gradient Descent", "quantum entanglement", "data pipeline"])
    by_term = {t["term"]: t for t in result["topics"]}
    assert "quantum entanglement" in result["llm"]["rejected"]
    assert "data pipeline" in result["llm"]["added"]
    assert by_term["gradient descent"]["llm_endorsed"]
    assert not by_term.get("data", {}).get("llm_endorsed", False)


def test_llm_digest_covers_the_whole_document():
    pages = [" ".join(f"Page {p} sentence {i} talks about subject{p}." for i in range(25)) for p in range(40)]
    ctx = build_context(pages, [], kind="pdf")
    digest = build_llm_digest(ctx, budget=8000)
    assert len(digest) <= 8000
    late = [s for s in ctx.sections if s.start > len(ctx.text) * 0.75]
    assert any(s.text.strip()[:40] in digest for s in late)
