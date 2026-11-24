.PHONY: install dev-install lint test coverage eval bench run ui docker-build clean

install:            ## core install (hashing embedder, no torch)
	pip install -e .

dev-install: install ## + test/lint tooling
	pip install -r requirements-dev.txt

lint:               ## ruff check
	ruff check semsearch tests scripts ui

test:               ## fast, torch-free test suite
	python -m pytest tests -v

coverage:
	coverage run --source=semsearch -m pytest tests
	coverage report --fail-under=88

eval:               ## eval set -> eval/metrics.json (add EMBEDDER=sbert for MiniLM)
	python scripts/run_eval.py --embedder $(or $(EMBEDDER),hashing) --out eval/metrics.json

bench:
	python scripts/bench.py --n 100

run:                ## local API
	uvicorn semsearch.api:app --host 0.0.0.0 --port 8000 --reload

ui:                 ## streamlit demo (needs: pip install -e '.[ui]')
	streamlit run ui/app.py

docker-build:
	docker build -t semsearch:latest .

clean:
	rm -rf .coverage htmlcov .semsearch_cache *.egg-info build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
