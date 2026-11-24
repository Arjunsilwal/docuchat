import pytest

from ragbot.evaluate import evaluate, load_questions
from ragbot.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragbot.retriever import HybridRetriever

RETRIEVED = ["x1", "r1", "r2", "x2", "r3"]  # relevant hits at ranks 2, 3 and 5
RELEVANT = {"r1", "r2", "r3", "r4"}


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == 3 / 4
    assert recall_at_k(RETRIEVED, RELEVANT, k=1) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=5) == 3 / 5


def test_hit_and_reciprocal_rank():
    assert hit_at_k(RETRIEVED, RELEVANT, k=2) == 1.0
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 1 / 2
    assert reciprocal_rank(["x"], RELEVANT) == 0.0


def test_every_eval_question_has_a_matching_chunk(store):
    """Guards the labeled set: a typo in answer_contains would silently break evaluation."""
    retriever = HybridRetriever(store)
    results = evaluate(retriever, load_questions("eval/questions.jsonl"), k=5)
    assert set(results) == {"vector", "keyword", "hybrid"}
    for metrics in results.values():
        assert 0.0 <= metrics["Recall@5"] <= 1.0


def test_unmatched_label_raises(store):
    bad = [{"question": "q", "source": "product-faq.md", "answer_contains": "no such text"}]
    with pytest.raises(ValueError):
        evaluate(HybridRetriever(store), bad, k=5)