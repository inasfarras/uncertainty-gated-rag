import re

import numpy as np
import numpy.typing as npt

from agentic_rag.embed.encoder import embed_texts


def sentence_split(text: str) -> list[str]:
    """Splits text into sentences using regex."""
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def embed_norm(texts: list[str]) -> npt.NDArray[np.float32]:
    """Embeds and L2-normalizes texts."""
    return embed_texts(texts)


def cosine_matrix(
    A: npt.NDArray[np.float32], B: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Computes cosine similarity matrix between two sets of normalized vectors."""
    return np.dot(A, B.T)


def overlap_ratio(
    answer: str, contexts: list[str], sim_threshold: float = 0.7
) -> float:
    """Computes the ratio of answer sentences supported by context sentences."""
    if not answer or not contexts:
        return 0.0

    s_answer = sentence_split(answer)
    s_contexts = [s for c in contexts for s in sentence_split(c)]

    # Limit to a reasonable number of sentences to avoid excessive computation
    s_contexts = s_contexts[:200]

    if not s_answer or not s_contexts:
        return 0.0

    emb_answer = embed_norm(s_answer)
    emb_contexts = embed_norm(s_contexts)

    sim_matrix = cosine_matrix(emb_answer, emb_contexts)

    supported_count = sum(1 for row in sim_matrix if np.max(row) >= sim_threshold)

    return supported_count / max(1, len(s_answer))


def faithfulness_score(question: str, contexts: list[str], answer: str) -> float | None:
    """Computes faithfulness score using ragas if available."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        dataset = Dataset.from_dict(
            {"question": [question], "contexts": [contexts], "answer": [answer]}
        )

        # NOTE: ragas expects OPENAI_API_KEY to be set in env for its internal LLM calls
        result = evaluate(dataset, metrics=[faithfulness])
        scores = result["faithfulness"]
        return scores[0] if scores else None
    except ImportError:
        # ragas not installed
        return None
    except Exception:
        # Could fail for other reasons e.g. API keys not set
        return None
