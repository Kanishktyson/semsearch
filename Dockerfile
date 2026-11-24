# Torch-free image: default SEMSEARCH_EMBEDDER=hashing, so the container is
# small and starts instantly. For the MiniLM embedder run locally instead:
#   pip install -e '.[sbert]' && uvicorn semsearch.api:app
FROM python:3.12-slim

# non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY pyproject.toml README.md ./
COPY semsearch/ semsearch/
COPY data/ data/
COPY eval/ eval/

RUN pip install --no-cache-dir . && chown -R appuser:appuser /app
USER appuser

ENV SEMSEARCH_EMBEDDER=hashing \
    SEMSEARCH_CACHE_DIR=/tmp/.semsearch_cache \
    SEMSEARCH_CORPUS=/app/data/corpus.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "semsearch.api:app", "--host", "0.0.0.0", "--port", "8000"]
