.PHONY: install format lint test clean run-api run-baseline run-agent

install:
	poetry install
	poetry run python -m spacy download en_core_web_sm || true

format:
	poetry run black .

lint:
	poetry run ruff check . --fix

type:
	poetry run mypy src/agentic_rag -q

test:
	poetry run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

run-api:
	poetry run uvicorn agentic_rag.api.app:create_app --factory --host $(API_HOST) --port $(API_PORT) --reload

ingest:
	python -m agentic_rag.ingest.ingest --input-dir src/agentic_rag/data/corpus --out-dir artifacts/faiss

run-baseline:
	python -m agentic_rag.eval.runner

run-agent:
	python -m agentic_rag.eval.runner --gate-on

smoke-online:
	python -m agentic_rag.ingest.ingest --input data/corpus --out artifacts/faiss --backend openai
	python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system baseline --gate-off --n 3
	python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system agent --gate-on --n 3

smoke-mock:
	python -m agentic_rag.ingest.ingest --input data/corpus --out artifacts/faiss --backend mock
	python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system baseline --gate-off --n 3
	python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system agent --gate-on --n 3
