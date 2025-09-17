import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
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
def test_pipeline(mock_get_openai, test_data_dir, test_index_dir, monkeypatch):
    from agentic_rag.ingest.ingest import Ingestor

    # Mock the OpenAI embedder
    mock_embedder = mock_get_openai.return_value
    monkeypatch.setattr("agentic_rag.config.settings.EMBED_BACKEND", "openai")

    def mock_embed(texts, embed_model):
        # Return small, deterministic vectors
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 8)).tolist()

    mock_embedder.embed.side_effect = mock_embed

    # 1. Ingestion
    ingestor = Ingestor(test_data_dir)
    ingestor.ingest(output_dir=test_index_dir)

    # 2. Retriever
    retriever = VectorRetriever(test_index_dir)
    results = retriever.retrieve("What are cats?", k=1)

    assert len(results) == 1
    assert "cats" in results[0]["text"]


def test_loop_short_circuit_no_new_hits():
    from agentic_rag.agent.loop import Agent

    with patch("agentic_rag.agent.loop.VectorRetriever") as mock_retriever, patch(
        "agentic_rag.agent.loop.OpenAIAdapter"
    ) as mock_llm:

        # Setup mock retriever to return the same doc twice
        mock_retriever.return_value.retrieve_pack.side_effect = [
            (
                [{"id": "doc1", "text": "text1"}],
                {"retrieved_ids": ["doc1"], "n_ctx_blocks": 1, "context_tokens": 10},
            ),
            (
                [{"id": "doc1", "text": "text1"}],
                {"retrieved_ids": ["doc1"], "n_ctx_blocks": 1, "context_tokens": 10},
            ),
        ]

        # Setup mock LLM to return some text
        mock_llm.return_value.chat.return_value = ("some answer", {"total_tokens": 5})

        agent = Agent(gate_on=True)
        agent.answer("some question")

        # Should only run for one round then stop due to no new hits
        assert mock_retriever.return_value.retrieve_pack.call_count == 2
        assert mock_llm.return_value.chat.call_count == 1
