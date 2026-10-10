import networkx as nx
import pytest

import sample_doc
from analysis import build_context, term_regex
from learning import build_learning_path, concept_evidence, depends_on, relation_evidence

TERMS = ["loss function", "gradient descent", "neural network", "backpropagation", "overfitting", "regularization"]


def topics(terms):
    return [{"term": t, "tier": "core", "score": 1.0} for t in terms]


@pytest.fixture(scope="module")
def ctx():
    return build_context(sample_doc.pages(), sample_doc.headings())


@pytest.fixture(scope="module")
def path(ctx):
    return build_learning_path(ctx, topics(TERMS))


def test_depends_on_patterns():
    assert depends_on("A transformer is a type of neural network.", "transformer", "neural network")
    assert depends_on("Backpropagation relies on the chain rule.", "backpropagation", "chain rule")
    assert depends_on("Gradients are used by gradient descent.", "gradient descent", "gradient")
    assert not depends_on("Gradient descent and backpropagation are both covered.", "backpropagation",
                          "gradient descent")


def test_learning_path_puts_prerequisites_first(path):
    order = [s["term"] for s in path["steps"]]
    assert order.index("loss function") < order.index("gradient descent") < order.index("neural network")
    assert order.index("neural network") < order.index("backpropagation")
    assert order.index("overfitting") < order.index("regularization")


def test_learning_path_is_a_dag_consistent_with_its_order(path):
    graph = nx.DiGraph([(e["source"], e["target"]) for e in path["edges"]])
    assert nx.is_directed_acyclic_graph(graph)
    order = [s["term"] for s in path["steps"]]
    assert all(order.index(e["source"]) < order.index(e["target"]) for e in path["edges"])
    assert all(e["evidence"]["label"].startswith("p. ") for e in path["edges"])


def test_cycles_are_broken_by_dropping_the_weakest_edge():
    pages = ["Beta is a type of alpha used in many systems for testing purposes.",
             "Alpha is a broad family of methods that includes beta and gamma variants."]
    ctx = build_context(pages, [])
    order = [s["term"] for s in build_learning_path(ctx, topics(["alpha", "beta"]))["steps"]]
    assert order == ["alpha", "beta"]


def test_concept_evidence_cites_the_definition_page(ctx, path):
    ev = concept_evidence(ctx, {"term": "gradient descent", "tier": "core", "score": 1.0}, path)
    assert ev["definitions"][0]["label"] == "p. 3"
    assert ev["definitions"][0]["text"].startswith("Gradient descent is")
    assert ev["stats"]["sections"] >= 3
    assert any(e["source"] == "loss function" for e in ev["prerequisites"])


def test_relation_evidence_returns_sentences_with_both_terms(ctx, path):
    r = relation_evidence(ctx, "gradient descent", "neural network", path)
    assert r["shared_sentences"] >= 1
    for s in r["sentences"]:
        assert term_regex("gradient descent").search(s["text"]) and term_regex("neural network").search(s["text"])
    assert 0 < r["weight"] <= 1
