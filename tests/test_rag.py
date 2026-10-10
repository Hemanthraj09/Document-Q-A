from rag import BM25, chunk_pages, fit_history, is_small_talk, reciprocal_rank_fusion, source_label, tokenize


def _page(word, n=60):
    return " ".join(f"This sentence is about {word} number {i}." for i in range(n))


def test_chunks_record_the_pages_they_span():
    chunks = chunk_pages([_page("alpha"), _page("beta"), _page("gamma")])
    assert chunks[0]["page_start"] == 1
    assert chunks[-1]["page_end"] == 3
    assert any(c["page_start"] == 1 and c["page_end"] == 2 for c in chunks)
    for c in chunks:
        if "alpha" in c["text"]:
            assert c["page_start"] == 1
        if "gamma" in c["text"]:
            assert c["page_end"] == 3
        if c["page_start"] > 1:
            assert "alpha" not in c["text"]


def test_chunks_respect_size_and_do_not_start_mid_word():
    text = _page("delta", 200)
    chunks = chunk_pages([text])
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert all(text[text.index(c["text"]) - 1] in " \n" for c in chunks[1:])


def test_bm25_ranks_the_chunk_with_the_query_term_first():
    docs = [tokenize(t) for t in ["the loss function measures error", "networks have layers",
                                  "gradient descent minimizes the loss function"]]
    assert BM25(docs).top(tokenize("gradient descent"), 3) == [2]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = [idx for idx, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]])]
    assert fused[:2] == [1, 3]


def test_source_labels():
    assert source_label({"page_start": 4, "page_end": 4}, "pdf", 0) == "p. 4"
    assert source_label({"page_start": 4, "page_end": 5}, "pdf", 0) == "pp. 4–5"
    assert source_label({"page_start": 1, "page_end": 1}, "text", 2) == "Excerpt 3"


def test_small_talk_only_matches_whole_greetings():
    for msg in ["hi", "Thanks a lot!", "good morning lexo", "ok"]:
        assert is_small_talk(msg), msg
    for msg in ["Define overfitting", "show me the loss function", "what is gradient descent", "hi, what is a loss?"]:
        assert not is_small_talk(msg), msg


def test_fit_history_keeps_recent_messages_starting_with_user():
    history = [{"role": "user", "content": "q1" * 50}, {"role": "assistant", "content": "a1" * 50},
               {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    assert fit_history(history, 10) == history[2:]
    assert fit_history(history, 150) == history[2:]
    assert fit_history(history, 10_000) == history
