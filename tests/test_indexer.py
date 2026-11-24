import numpy as np
import pytest

from semsearch.indexer import BruteForceIndex


def test_search_exact_matches_full_sort(mini_index):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=mini_index.dim).astype(np.float32)
        q /= np.linalg.norm(q)
        ids, scores = mini_index.search(q, k=3)
        full = mini_index._matrix @ q
        expected = set(np.argsort(-full)[:3])
        assert set(ids) == {mini_index.doc_ids[i] for i in expected}
        assert list(scores) == sorted(scores, reverse=True)


def test_self_retrieval_ranks_first(mini_index):
    """Feeding a doc's own vector back must rank that doc first at ~1.0 cosine."""
    target = 0  # doc-0001 by construction of the mini corpus
    ids, scores = mini_index.search(mini_index._matrix[target], k=5)
    assert ids[0] == "doc-0001"
    assert scores[0] > 0.999


def test_k_larger_than_corpus_is_clamped(mini_index):
    q = mini_index._matrix[0]
    ids, scores = mini_index.search(q, k=99)
    assert len(ids) == mini_index.size == 5


def test_scores_are_cosine_bounded(mini_index):
    q = mini_index._matrix[2]
    _, scores = mini_index.search(q, k=5)
    assert np.all(scores <= 1.0 + 1e-6)
    assert np.all(scores >= -1.0 - 1e-6)


def test_misshapen_matrix_rejected():
    with pytest.raises(ValueError, match="2-D"):
        BruteForceIndex(np.zeros(10, dtype=np.float32), ["a"])


def test_unnormalized_rows_rejected():
    with pytest.raises(ValueError, match="normalized"):
        BruteForceIndex(np.ones((2, 4), dtype=np.float32), ["a", "b"])


def test_row_count_mismatch_rejected():
    bad = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="match"):
        BruteForceIndex(bad, ["a", "b"])
