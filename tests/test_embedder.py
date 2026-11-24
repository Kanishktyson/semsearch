import numpy as np
import pytest

from semsearch.embedder import HashingEmbedder, SentenceTransformerEmbedder, get_embedder


def test_output_is_l2_normalized():
    emb = HashingEmbedder(dim=256)
    vectors = emb.embed(["hello world", "the quick brown fox jumps"])
    assert vectors.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_deterministic_same_input_same_output():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["identical text"])
    b = emb.embed(["identical text"])
    assert np.array_equal(a, b)


def test_similar_texts_closer_than_unrelated():
    emb = HashingEmbedder(dim=512)
    texts = [
        "oil prices surge on supply fears",
        "oil prices climb as supply drops",
        "tennis player wins grand slam match",
    ]
    v = emb.embed(texts)
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_bigrams_carry_word_order():
    emb = HashingEmbedder(dim=512)
    v = emb.embed(["oil price rises", "price oil rises"])
    assert float(v[0] @ v[1]) < 0.999  # not identical despite same unigram bag


def test_empty_text_is_zero_vector_by_contract():
    """No tokens -> no signal -> exactly zero vector. The index treats zero
    rows as 'never ranks' (see indexer) instead of pretending they are meaningful."""
    emb = HashingEmbedder(dim=64)
    v = emb.embed([""])
    assert np.allclose(v[0], 0.0)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown embedder"):
        get_embedder("gpt4")


def test_factory_variants_have_distinct_names():
    assert get_embedder("hashing", hashing_dim=512).name == "hashing-512"
    assert get_embedder("hashing", hashing_dim=256).name == "hashing-256"


def test_sbert_wrapper_defers_torch_import():
    """The class must be constructible without torch installed (lazy loading)."""
    st = SentenceTransformerEmbedder()
    assert st.name == "sbert:all-MiniLM-L6-v2"
    assert st.dim == 384
