# Agentic RAG

An advanced RAG (Retrieval-Augmented Generation) system with agentic capabilities for enhanced information retrieval and generation.

## Features

- FastAPI-based REST API
- Support for multiple LLM providers (OpenAI, Llama, Ollama)
- Vector search using FAISS
- Advanced text processing with spaCy and NLTK
- Evaluation metrics using RAGAS
- Comprehensive test suite

## Requirements

- Python 3.11+
- An OpenAI API Key

## Windows Quickstart

1.  **Create and activate a virtual environment:**
    ```powershell
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2.  **Install dependencies:**
    ```bash
    pip install -U pip wheel setuptools
    pip install fastapi uvicorn pydantic numpy pandas scikit-learn faiss-cpu tiktoken ragas datasets nltk pyyaml rich loguru typer httpx orjson openai tenacity black ruff mypy pytest pytest-cov
    ```

3.  **Configure your environment:**
    Copy `.env.example` to `.env` and set your `OPENAI_API_KEY`.
    ```powershell
    copy .env.example .env
    # Now edit the .env file
    ```

4.  **Create a toy corpus:**
    Create a directory `data/corpus` and add a few `.txt` files with some text content.

5.  **Build the FAISS index:**
    ```bash
    python -m agentic_rag.ingest.ingest --input data/corpus --out artifacts/faiss
    ```

6.  **Run a smoke test:**
    ```bash
    make run-baseline
    ```
    This will ask 3 hardcoded questions and print the results. To run with the agentic gate enabled (once implemented), use `make run-agent`.

## Usage

### Ingest Data
```bash
make ingest
```

The API will be available at http://localhost:8000

### Running the Baseline System

```bash
make run-baseline
```

### Running the Agent System

```bash
make run-agent
```

## Development

### Code Quality

Format code:
```bash
make format
```

Run linters:
```bash
make lint
```

### Testing

Run tests with coverage:
```bash
make test
```

### Clean Build Files

```bash
make clean
```

## Optional Dependencies

The project supports multiple LLM backends:

- OpenAI API: `poetry install --extras openai`
- Llama.cpp: `poetry install --extras llama`
- Ollama: No additional installation needed (uses HTTP API)

## License

MIT
