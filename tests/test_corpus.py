from pathlib import Path

import pytest

from semsearch.corpus import Corpus, Document, load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_document_combined_text_repeats_title():
    doc = Document(id="x", title="Oil!", text="prices", label="Business")
    assert doc.combined_text == "Oil!. Oil!. prices"


def test_real_corpus_loads_and_is_clean():
    corpus = load_corpus(PROJECT_ROOT / "data/corpus.csv")
    assert len(corpus) == 400
    ids = [d.id for d in corpus.docs]
    assert len(set(ids)) == 400
    assert all(d.title and d.text and d.label for d in corpus.docs)
    assert {"World", "Sports", "Business", "Sci/Tech"} == {d.label for d in corpus.docs}


def test_duplicate_ids_rejected():
    a = Document(id="same", title="t", text="x", label="World")
    b = Document(id="same", title="t", text="y", label="World")
    with pytest.raises(ValueError, match="duplicate"):
        Corpus(docs=(a, b))


def test_missing_corpus_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_corpus"):
        load_corpus(tmp_path / "nope.csv")
