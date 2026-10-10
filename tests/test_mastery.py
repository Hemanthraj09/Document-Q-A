from datetime import datetime, timedelta, timezone

import pytest

import mastery
from mastery import (MasteryStore, apply_score, grade_exact, is_dont_know, make_cloze, new_state, parse_grade,
                     pick_quiz_target, record_exposure, status)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_mastery_moves_toward_score_and_reaches_mastered():
    state = new_state()
    for expected in (0.5, 0.75, 0.875):
        apply_score(state, 1.0, NOW)
        assert state["mastery"] == pytest.approx(expected)
    assert status(state) == "mastered"


def test_sm2_intervals_grow_and_reset_on_failure():
    state = new_state()
    intervals = []
    for _ in range(3):
        apply_score(state, 1.0, NOW)
        intervals.append(state["interval"])
    assert intervals[:2] == [1, 6] and intervals[2] > 6
    apply_score(state, 0.0, NOW)
    assert state["reps"] == 0 and state["interval"] == 1
    assert datetime.fromisoformat(state["due"]) == NOW + timedelta(days=1)


def test_status_progression():
    state = new_state()
    assert status(state) == "unseen"
    record_exposure(state, NOW)
    assert status(state) == "explored"
    apply_score(state, 0.4, NOW)
    assert status(state) == "learning"


def test_quiz_target_waits_for_prerequisites():
    topics = [{"term": "a", "score": 0.6, "tier": "supporting"}, {"term": "b", "score": 1.0, "tier": "core"}]
    path = {"edges": [{"source": "a", "target": "b"}], "steps": [{"term": "a"}, {"term": "b"}]}
    states = {"a": new_state(), "b": new_state()}
    assert pick_quiz_target(topics, states, path, NOW)[0] == "a"

    for _ in range(3):
        apply_score(states["a"], 1.0, NOW)
    term, reason = pick_quiz_target(topics, states, path, NOW)
    assert term == "b"
    assert "prerequisites are in place" in reason


def test_cloze_and_exact_grading():
    question = make_cloze("Gradient descent is an optimization algorithm that minimizes loss.", "gradient descent")
    assert question.startswith("_____ is an optimization")
    assert grade_exact("Gradient descents", "gradient descent") == 1.0
    assert grade_exact("gradiant descent", "gradient descent") == 0.75
    assert grade_exact("backprop", "gradient descent") == 0.0
    assert grade_exact("", "gradient descent") == 0.0


def test_parse_grade_accepts_percent_and_rejects_garbage():
    assert parse_grade('{"score": 85, "feedback": "Good"}') == (0.85, "Good")
    assert parse_grade('{"score": 0.4}') == (0.4, None)
    assert parse_grade("not json") == (None, None)


def test_dont_know_only_matches_short_replies():
    assert is_dont_know("I don't know")
    assert is_dont_know("idk")
    assert not is_dont_know("I don't know exactly, but it minimizes the loss function")


def test_blend_and_similarity_bounds():
    assert mastery.similarity_score(0.1) == 0.0
    assert mastery.similarity_score(0.9) == 1.0
    assert mastery.blend(None, 0.5) == 0.5
    assert mastery.blend(1.0, 0.0) == pytest.approx(0.6)


def test_store_persists_across_instances(tmp_path):
    store = MasteryStore(tmp_path)
    store.record_exposure("doc1", ["Gradient Descent"], NOW)
    store.apply_score("doc1", "gradient descent", 1.0, NOW)
    reloaded = MasteryStore(tmp_path).states("doc1", ["gradient descent", "overfitting"])
    assert reloaded["gradient descent"]["exposures"] == 1
    assert reloaded["gradient descent"]["mastery"] == 0.5
    assert reloaded["overfitting"] == new_state()
