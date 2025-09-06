import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from agentic_rag.ingest.ingest import main as ingest_main
from agentic_rag.retriever.vector import VectorRetriever


@pytest.fixture(scope="module")
def test_data_dir():
    dir_path = Path("./test_data")
    dir_path.mkdir(exist_ok=True)

    (dir_path / "doc1.txt").write_text("This is a document about cats.")
    (dir_path / "doc2.txt").write_text("This is a document about dogs.")
    (dir_path / "doc3.txt").write_text("This is a document about birds.")

    yield str(dir_path)

    shutil.rmtree(dir_path)


@pytest.fixture(scope="module")
def test_index_dir():
    dir_path = Path("./test_index")
    yield str(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)


@pytest.mark.e2e
@patch("agentic_rag.embed.encoder.get_openai")
def test_pipeline(mock_get_openai, test_data_dir, test_index_dir):
    # Mock the OpenAI embedder
    mock_embedder = mock_get_openai.return_value

    def mock_embed(texts, embed_model):
        # Return small, deterministic vectors
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 8)).tolist()

    mock_embedder.embed.side_effect = mock_embed

    # 1. Ingestion
    ingest_main(input=test_data_dir, out=test_index_dir)

    # 2. Retriever
    retriever = VectorRetriever(test_index_dir)
    results = retriever.retrieve("What are cats?", k=1)

    assert len(results) == 1
    assert "cats" in results[0]["text"]
