"""Tests for the lightweight retrieval layer (the RAG half of the agent)."""

from __future__ import annotations

import pytest

from app.knowledge_base import get_knowledge_base


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("I forgot my password and cannot sign in", "kb_password_reset"),
        ("how much does the Pro plan cost", "kb_plans"),
        ("I want to upgrade to a bigger plan", "kb_upgrade"),
        ("what is your cancellation policy", "kb_cancellation"),
        ("can I get my money back", "kb_refunds"),
        ("my card was declined, when do you retry", "kb_billing_faq"),
    ],
)
def test_search_returns_the_right_article(query: str, expected_id: str) -> None:
    results = get_knowledge_base().search(query)
    assert results, f"no results for {query!r}"
    assert results[0].document.id == expected_id


def test_search_respects_top_k() -> None:
    assert len(get_knowledge_base().search("billing plan refund upgrade", top_k=2)) <= 2


def test_search_ignores_stopword_only_queries() -> None:
    assert get_knowledge_base().search("what is it") == []


def test_search_returns_nothing_for_unrelated_topics() -> None:
    assert get_knowledge_base().search("kangaroo tectonics submarine") == []


def test_results_are_ordered_by_descending_score() -> None:
    scores = [r.score for r in get_knowledge_base().search("cancel my subscription")]
    assert scores == sorted(scores, reverse=True)


def test_weak_tail_matches_are_dropped() -> None:
    """Only genuinely relevant articles are returned, so `sources` stays honest."""
    results = get_knowledge_base().search("What is your cancellation policy?")
    ids = [result.document.id for result in results]
    assert ids[0] == "kb_cancellation"
    assert "kb_data_export" not in ids
