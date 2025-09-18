import re
import string
from typing import Any, Dict, List, Set

import numpy as np
import numpy.typing as npt

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts

# Strict citation regex
CIT_RE = re.compile(r"\[CIT:(?P<id>[A-Za-z0-9_\-]+)\]")
# Expose alternate alias with non-named group for external consumers/tests
CITATION_RE = re.compile(r"\[CIT:([A-Za-z0-9_\-]+)\]")


def split_sentences(text: str) -> list[str]:
    # Don't split on sentence-ending punctuation if it's followed by a citation
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?!\[CIT:)", text.strip())
        if s.strip()
    ]


def is_idk(s: str) -> bool:
    t = s.strip().lower().rstrip(".! ")
    return t in {"i don't know", "i dont know"}


def extract_citations(text: str) -> List[str]:
    return [m.group("id") for m in CIT_RE.finditer(text)]


def extract_sentence_citations(text: str) -> List[Set[str]]:
    sentences = split_sentences(text)
    out: List[Set[str]] = []
    for s in sentences:
        out.append(set(extract_citations(s)))
    return out


def embed_norm(texts: list[str]) -> npt.NDArray[np.float32]:
    return embed_texts(texts)


def cosine_matrix(
    A: npt.NDArray[np.float32], B: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    return np.dot(A, B.T)


def sentence_support(
    answer: str, ctx_map: Dict[str, str], tau_sim: float | None = None
) -> Dict[str, float | int]:
    if tau_sim is None:
        tau_sim = settings.OVERLAP_SIM_TAU

    if not answer:
        return {
            "overlap": 0.0,
            "sentences": 0,
            "supported": 0,
            "idk_with_citation_count": 0,
        }

    sents = split_sentences(answer)
    if not sents:
        return {
            "overlap": 0.0,
            "sentences": 0,
            "supported": 0,
            "idk_with_citation_count": 0,
        }

    supported = 0
    idk_with_cit = 0
    for s in sents:
        # Check for IDK *after* stripping citations
        s_no_cit = CIT_RE.sub("", s).strip()
        if is_idk(s_no_cit):
            # IDK sentences cannot be supported and must not have a citation
            if extract_citations(s):
                idk_with_cit += 1
            continue

        # Must contain exactly one valid [CIT:<doc_id>]
        cits = extract_citations(s)
        if len(cits) != 1:
            continue
        doc_id = cits[0]
        if doc_id not in ctx_map:
            continue

        # Compute similarity between sentence and the matched chunk
        # Deterministic TF-IDF-like proxy using embeddings
        emb_s = embed_norm([s_no_cit])
        emb_c = embed_norm([ctx_map[doc_id]])
        sim = float(cosine_matrix(emb_s, emb_c)[0, 0])
        if sim >= (tau_sim or 0.60):
            supported += 1

    overlap = supported / len(sents)
    return {
        "overlap": overlap,
        "sentences": len(sents),
        "supported": supported,
        "idk_with_citation_count": idk_with_cit,
    }


def faithfulness_fallback(answer: str, gold: str | None, overlap: float) -> float:
    if is_idk(answer) and gold and gold.strip():
        return 0.0
    return min(1.0, 0.6 + 0.4 * overlap)


def _normalize_text(t: Any) -> str:
    t = str(t).lower()
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def em_f1(pred: str, gold: str | None) -> Dict[str, float]:
    if gold is None:
        return {"em": 0.0, "f1": 0.0}
    p = _normalize_text(pred or "")
    g = _normalize_text(gold or "")
    em = 1.0 if p == g and g != "" else 0.0

    # Token-level F1
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        f1 = 1.0
    elif not p_tokens or not g_tokens:
        f1 = 0.0
    else:
        # Count overlap
        from collections import Counter

        pc = Counter(p_tokens)
        gc = Counter(g_tokens)
        common = sum((pc & gc).values())
        if common == 0:
            f1 = 0.0
        else:
            precision = common / max(1, len(p_tokens))
            recall = common / max(1, len(g_tokens))
            f1 = 2 * precision * recall / (precision + recall)
    return {"em": em, "f1": f1}


# Aliases to match alternate API names requested
def normalize_text(s: Any) -> str:
    return _normalize_text(s)


def compute_em_f1(gold: str | list[str] | None, pred: str) -> tuple[float, float]:
    # If gold list provided, choose max over candidates
    if isinstance(gold, list):
        if not gold:
            return (0.0, 0.0)
        pairs = [em_f1(pred, g) for g in gold]
        em = max(p["em"] for p in pairs)
        f1 = max(p["f1"] for p in pairs)
        return (float(em), float(f1))
    m = em_f1(pred, gold)
    return (float(m["em"]), float(m["f1"]))


# Backwards-compatible helpers used in older code/tests
def sentence_split(text: str) -> list[str]:
    return split_sentences(text)


def overlap_ratio(
    answer: str,
    contexts: list[str],
    sim_threshold: float | None = None,
    context_ids: list[str] | None = None,
) -> float:
    # Legacy overlap: any sentence with any valid citation to available IDs, else semantic fallback
    if not answer or not contexts:
        return 0.0
    if sim_threshold is None:
        sim_threshold = settings.OVERLAP_SIM_TAU
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    sent_cits = extract_sentence_citations(answer)
    available_ids = set(context_ids) if context_ids else set()
    supported = 0
    for s, cits in zip(sentences, sent_cits):
        if cits and (not available_ids or (cits & available_ids)):
            supported += 1
            continue
        # fallback similarity vs all contexts
        emb_s = embed_norm([s])
        emb_c = embed_norm(contexts)
        sim = float(np.max(cosine_matrix(emb_s, emb_c)))
        if sim >= sim_threshold:
            supported += 1
    return supported / len(sentences)


def faithfulness_score(question: str, contexts: list[str], answer: str) -> float | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        dataset = Dataset.from_dict(
            {"question": [question], "contexts": [contexts], "answer": [answer]}
        )
        result = evaluate(dataset, metrics=[faithfulness])
        scores = result["faithfulness"]
        return scores[0] if scores else None
    except Exception:
        return None
