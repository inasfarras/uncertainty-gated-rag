.PHONY: install format lint test clean run-api run-baseline run-agent

install:
	poetry install --with dev
	poetry run python -m spacy download en_core_web_sm
	poetry run pre-commit install

format:
	poetry run black src tests
	poetry run ruff --fix src tests

lint:
	poetry run black --check src tests
	poetry run ruff src tests
	poetry run mypy src tests

test:
	poetry run pytest -v --cov=src --cov-report=term-missing tests/

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

run-baseline:
	poetry run python -m agentic_rag.baseline.run

run-agent:
	poetry run python -m agentic_rag.agent.run
