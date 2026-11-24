"""Streamlit demo — semantic search in a browser.

    pip install -e '.[ui]'
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from semsearch.config import SearchConfig  # noqa: E402
from semsearch.search import SearchService  # noqa: E402

st.set_page_config(page_title="semsearch", page_icon="🔎", layout="wide")

st.title("🔎 semsearch — semantic news search")
st.caption("400 AG News articles · pluggable embedders · exact cosine index · hand-labeled eval")


@st.cache_resource(show_spinner="building index…")
def load_service() -> SearchService:
    return SearchService.from_config(SearchConfig.from_env(ROOT))


service = load_service()
st.badge(f"embedder: {service.embedder.name}" if hasattr(st, "badge") else service.embedder.name)

query = st.text_input("Search by meaning — try *president confirmed by election* or *phone virus*:", "")
k = st.slider("results", 1, 20, 5)

if query.strip():
    hits = service.search(query, k=k)
    st.write(f"`{len(hits)}` results for **{query}**")
    for r in hits:
        score_color = "🟢" if r.score > 0.5 else ("🟡" if r.score > 0.2 else "⚪")
        with st.container(border=True):
            st.markdown(f"**{r.title}**  ·  `{r.doc_id}` · {r.label} · {score_color} cosine **{r.score:.3f}**")
            st.markdown(r.text[:300] + ("…" if len(r.text) > 300 else ""))
else:
    st.info("Enter a query above. Queries with different wording than the articles are where "
            "semantic search shines — e.g. *'headphones battery life'* finds audio-hardware pieces.")
