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

crag-prepare:
	python scripts/prepare_crag.py --out-dir data/crag_corpus --qs-file data/crag_questions.jsonl --split test --static-only --n 200

crag-ingest-openai:
	python -m agentic_rag.ingest.ingest --input data/crag_corpus --out artifacts/crag_faiss --backend openai

crag-ingest-mock:
	python -m agentic_rag.ingest.ingest --input data/crag_corpus --out artifacts/crag_faiss --backend mock

crag-run-baseline:
	python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --gate-off --n 200

crag-run-agent:
	python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 200

crag-all-mock: crag-prepare crag-ingest-mock crag-run-baseline crag-run-agent

crag-full-download:
	python scripts/crag_full_download.py

crag-full-prepare:
	python scripts/prepare_crag_from_jsonl.py --src data/crag_task_1_and_2_dev_v4.jsonl.bz2 --out-dir data/crag_corpus_html --qs-file data/crag_questions.jsonl --meta-file data/crag_meta.jsonl --static-only --n 200 --min-chars 500 --max-pages-per-q 20

crag-full-ingest-openai:
	python -m agentic_rag.ingest.ingest --input data/crag_corpus_html --out artifacts/crag_faiss --backend openai

crag-full-ingest-mock:
	python -m agentic_rag.ingest.ingest --input data/crag_corpus_html --out artifacts/crag_faiss --backend mock

crag-full-run-baseline:
	python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --gate-off --n 200

crag-full-run-agent:
	python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 200

crag-full-all-openai: crag-full-download crag-full-prepare crag-full-ingest-openai crag-full-run-baseline crag-full-run-agent
