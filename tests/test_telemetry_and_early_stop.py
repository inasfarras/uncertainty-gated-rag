from types import SimpleNamespace

from agentic_rag.supervisor.orchestrator import AnchorSystem


class DummyLLM:
    def tokenize_count(self, text, model=None):
        return max(1, len(text) // 4)

    def chat(self, messages, model=None, max_tokens=128, temperature=0.0):
        # Always produce IDK to keep things deterministic
        return ("I don't know", {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})


def test_anchor_orchestrator_runs_monkeypatch(tmp_path, monkeypatch):
    # Monkeypatch LLM and retrieval to avoid FAISS
    sys = AnchorSystem(debug_mode=False)
    monkeypatch.setattr(sys, "llm", DummyLLM())

    # Fake retrieval: always return same doc id -> new_hits_ratio = 0 beyond first
    def fake_explore(anchor, question, hop_budget=1, seen_doc_ids=None):
        return [
            {
                "anchor": anchor,
                "hops": 1,
                "doc_ids": ["doc1"],
                "passages": [{"id": "doc1", "text": "some text 2007.", "score": 1.0}],
                "novelty_ratio": 0.0,
                "rough_scores": [1.0],
                "fine_scores": [0.5],
                "pruned_count": 0,
                "terminated_by": "BUDGET",
            }
        ]

    monkeypatch.setattr(sys.retriever, "explore", fake_explore)

    res = sys.answer("When was iPhone released?", qid="test_qid")
    assert isinstance(res, dict)
    assert "final_answer" in res
    # JSONL log file should be present
    # (path derived from settings.LOG_DIR, which defaults to logs/)

