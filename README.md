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
- Poetry for dependency management
- spaCy English language model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
```

2. Install dependencies:
```bash
make install
```

This will:
- Install all dependencies using Poetry
- Download the spaCy English language model
- Set up pre-commit hooks

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
- Set your LLM provider (openai/llama/ollama)
- Configure API keys if using OpenAI
- Adjust API host/port if needed

## Usage

### Running the API

```bash
make run-api
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
