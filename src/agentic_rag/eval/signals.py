import re
from typing import List, Set

import numpy as np
import numpy.typing as npt

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts


def sentence_split(text: str) -> list[str]:
    """Splits text into sentences using regex."""
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_citations(text: str) -> List[str]:
    """Extract citation IDs from text using regex pattern [CIT:<doc_id>]."""
    pattern = r"\[CIT:([^\]]+)\]"
    return re.findall(pattern, text)


def extract_sentence_citations(text: str) -> List[Set[str]]:
    """Extract citations for each sentence in the text."""
    sentences = sentence_split(text)
    sentence_citations = []

    for sentence in sentences:
        citations = set(extract_citations(sentence))
        sentence_citations.append(citations)

    return sentence_citations


def embed_norm(texts: list[str]) -> npt.NDArray[np.float32]:
    """Embeds and L2-normalizes texts."""
    return embed_texts(texts)


def cosine_matrix(
    A: npt.NDArray[np.float32], B: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Computes cosine similarity matrix between two sets of normalized vectors."""
    return np.dot(A, B.T)


def overlap_ratio(
    answer: str,
    contexts: list[str],
    sim_threshold: float | None = None,
    context_ids: list[str] | None = None,
) -> float:
    """Computes the ratio of answer sentences supported by citations and context similarity."""
    if not answer or not contexts:
        return 0.0

    # Use configured similarity threshold
    if sim_threshold is None:
        sim_threshold = settings.OVERLAP_SIM_TAU

    sentences = sentence_split(answer)
    if not sentences:
        return 0.0

    # Extract citations for each sentence
    sentence_citations = extract_sentence_citations(answer)

    # Get available context IDs (if provided, use them; otherwise extract from contexts)
    available_ids = set(context_ids) if context_ids else set()

    supported_count = 0

    for sentence, citations in zip(sentences, sentence_citations):
        # Check if sentence has valid citations
        if citations and (not available_ids or citations.intersection(available_ids)):
            supported_count += 1
        else:
            # Fallback to semantic similarity if no valid citations
            s_contexts = [s for c in contexts for s in sentence_split(c)]
            if s_contexts:
                s_contexts = s_contexts[:200]  # Limit computation

                emb_sentence = embed_norm([sentence])
                emb_contexts = embed_norm(s_contexts)

                sim_matrix = cosine_matrix(emb_sentence, emb_contexts)
                if np.max(sim_matrix) >= sim_threshold:
                    supported_count += 1

    return supported_count / len(sentences)


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
