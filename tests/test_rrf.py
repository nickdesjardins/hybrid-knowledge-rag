"""Tests for Reciprocal Rank Fusion math, candidate merging, and the SQL CTE shape."""

from __future__ import annotations

import math

import pytest

from app.database import vector_literal
from app.retrieval.hybrid_search import HYBRID_SEARCH_SQL, reciprocal_rank_fusion, rrf_score


def test_single_retriever_scores_follow_formula() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_document_in_both_lists_outranks_single_list_winner() -> None:
    dense = ["x", "both", "y"]
    sparse = ["z", "both", "w"]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))
    assert fused["both"] == pytest.approx(2 / 62)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["both"] > fused["x"]
    ordered = [doc for doc, _ in reciprocal_rank_fusion([dense, sparse], k=60)]
    assert ordered[0] == "both"


def test_rrf_score_helper_handles_missing_ranks() -> None:
    assert rrf_score([1, 1], k=60) == pytest.approx(2 / 61)
    assert rrf_score([3, None], k=60) == pytest.approx(1 / 63)
    assert rrf_score([None, None], k=60) == 0.0
    assert rrf_score([1], k=1) == pytest.approx(0.5)


def test_k_controls_how_much_top_ranks_dominate() -> None:
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert small_k["a"] / small_k["b"] > large_k["a"] / large_k["b"]
    assert math.isclose(large_k["a"] / large_k["b"], 1.0, rel_tol=0.01)


def test_ties_follow_the_sql_order_by() -> None:
    """Equal scores: better dense rank first, then sparse rank, then id (as in the CTE)."""
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    assert [doc for doc, _ in fused] == ["b", "a"]  # b has a dense rank, a only a sparse one
    assert fused[0][1] == fused[1][1]
    # Only sparse hits: identical scores fall back to the id.
    fused = reciprocal_rank_fusion([[], ["q", "p"]], k=60)
    assert [doc for doc, _ in fused] == ["q", "p"]  # rank 1 beats rank 2 regardless of id
    fused = reciprocal_rank_fusion([["m", "n"], ["n", "m"]], k=60)
    assert [doc for doc, _ in fused] == ["m", "n"]  # same score, m has the better dense rank


def test_limit_truncates_after_sorting() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60, limit=2)
    assert len(fused) == 2
    assert fused[0][0] == "c"  # rank 3 + rank 1 beats rank 1 alone


def test_duplicate_ids_within_one_retriever_count_once() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_python_fusion_matches_sql_formula_component_wise() -> None:
    """The SQL CTE computes COALESCE(1/(k+d.rank),0)+COALESCE(1/(k+s.rank),0) per chunk."""
    dense = ["c1", "c2", "c3", "c4"]
    sparse = ["c3", "c5", "c1"]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))
    for chunk_id, score in fused.items():
        dense_rank = dense.index(chunk_id) + 1 if chunk_id in dense else None
        sparse_rank = sparse.index(chunk_id) + 1 if chunk_id in sparse else None
        assert score == pytest.approx(rrf_score([dense_rank, sparse_rank], k=60))


def test_sql_cte_has_expected_structure() -> None:
    sql = str(HYBRID_SEARCH_SQL)
    assert "WITH dense_raw AS MATERIALIZED" in sql
    assert "dense_search AS" in sql
    assert "ORDER BY distance + 0 ASC, id ASC" in sql  # forced re-sort for iterative scans
    assert "sparse_search AS" in sql
    assert "rrf_fusion AS" in sql
    assert "<=>" in sql  # pgvector cosine distance operator
    assert "websearch_to_tsquery('english'" in sql
    assert "' & ', ' | '" in sql  # OR-relaxed match for natural-language questions
    assert "ts_rank_cd" in sql
    assert "FULL OUTER JOIN sparse_search" in sql
    assert "COALESCE(1.0 / (CAST(:rrf_k AS double precision) + d.rank), 0.0)" in sql
    assert "COALESCE(1.0 / (CAST(:rrf_k AS double precision) + s.rank), 0.0)" in sql
    assert "ORDER BY f.rrf_score DESC" in sql
    for name in (":query_vector", ":query_text", ":document_id", ":limit_n", ":rrf_k"):
        assert name in sql


def test_vector_literal_round_trips_floats() -> None:
    assert vector_literal([0.5, -1.0, 2]) == "[0.5,-1.0,2.0]"
    assert vector_literal([]) == "[]"
    assert vector_literal([1e-5]) == "[1e-05]"
