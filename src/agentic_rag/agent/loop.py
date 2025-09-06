import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import typer

from agentic_rag.config import settings
from agentic_rag.eval.signals import faithfulness_score, overlap_ratio
from agentic_rag.models.adapter import ChatMessage, OpenAIAdapter
from agentic_rag.retriever.vector import ContextChunk, VectorRetriever
from agentic_rag.utils.encoder import NpEncoder
from agentic_rag.utils.timing import timer


def is_global_question(q: str) -> bool:
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(q)
        return len(q.split()) >= 20 or len(doc.ents) >= 2
    except Exception:
        return len(q.split()) >= 20


def build_prompt(contexts: List[ContextChunk], question: str) -> List[ChatMessage]:
    """Builds a prompt for the LLM."""
    context_str = "\n\n".join([f"ID: {c['id']}\nText: {c['text']}" for c in contexts])
    content = (
        "You are a helpful assistant. Answer the user's question based only on the provided context.\n"
        "Your answer must be grounded in the context. Every sentence in your answer must be supported by the context.\n"
        "For each sentence, you must cite the ID of the context chunk that supports it. Append the citation in the format `[CIT:<ID>]` at the end of the sentence.\n"
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return [ChatMessage(role="user", content=content)]


class GateAction(str, Enum):
    STOP = "STOP"
    STOP_LOW_CONF = "STOP_LOW_CONF"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    SWITCH_GRAPH = "SWITCH_GRAPH"  # Not implemented
    REFLECT = "REFLECT"  # Not implemented


class BaseAgent:
    def __init__(self, system: str = "base"):
        self.system = system
        self.retriever = VectorRetriever(settings.FAISS_INDEX_PATH)
        self.llm = OpenAIAdapter()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def _log_jsonl(self, data: Dict[str, Any], log_path: Path):
        with open(log_path, "a") as f:
            f.write(json.dumps(data, cls=NpEncoder) + "\n")


class Baseline(BaseAgent):
    def __init__(self):
        super().__init__(system="baseline")

    def answer(self, question: str, qid: str | None = None) -> Dict[str, Any]:
        qid = qid or str(uuid.uuid4())
        log_path = self.log_dir / f"{self.system}_{qid}.jsonl"

        tokens_left = settings.MAX_TOKENS_TOTAL
        k = settings.RETRIEVAL_K

        contexts = self.retriever.retrieve(question, k=k)
        prompt = build_prompt(contexts, question)

        with timer() as t:
            draft, usage = self.llm.chat(messages=prompt)
        latency_ms = t()

        tokens_left -= usage["total_tokens"]

        context_texts = [c["text"] for c in contexts]
        o = overlap_ratio(draft, context_texts)
        f = faithfulness_score(question, context_texts, draft)
        if f is None:
            f = min(1.0, 0.6 + 0.4 * o)

        step_log = {
            "qid": qid,
            "round": 1,
            "action": "STOP",
            "k": k,
            "mode": "vector",
            "f": f,
            "o": o,
            "tokens_left": tokens_left,
            "usage": usage,
            "latency_ms": latency_ms,
            "prompt": [m.content for m in prompt],
            "draft": draft,
        }
        self._log_jsonl(step_log, log_path)

        summary_log = {
            "qid": qid,
            "final_answer": draft,
            "final_f": f,
            "final_o": o,
            "rounds": 1,
            "total_tokens": usage["total_tokens"],
            "p50_latency_ms": latency_ms,
            "latencies": [latency_ms],
        }
        self._log_jsonl(summary_log, log_path)

        return summary_log


class Agent(BaseAgent):
    def __init__(self, gate_on: bool = True):
        super().__init__(system="agent")
        self.gate_on = gate_on

    def _gate(self, f: float, o: float) -> GateAction:
        if f >= settings.FAITHFULNESS_TAU and o >= settings.OVERLAP_TAU:
            return GateAction.STOP
        return GateAction.RETRIEVE_MORE

    def answer(self, question: str, qid: str | None = None) -> Dict[str, Any]:
        qid = qid or str(uuid.uuid4())
        log_path = self.log_dir / f"{self.system}_{qid}.jsonl"

        # Initialize state
        r = 0
        k = settings.RETRIEVAL_K
        mode = "vector"
        tokens_left = settings.MAX_TOKENS_TOTAL
        total_tokens = 0
        latencies = []

        while r < settings.MAX_ROUNDS and tokens_left > 0:
            r += 1

            contexts = self.retriever.retrieve(question, k=k)
            prompt = build_prompt(contexts, question)

            with timer() as t:
                draft, usage = self.llm.chat(messages=prompt)
            latency_ms = t()
            latencies.append(latency_ms)

            total_tokens += usage["total_tokens"]
            tokens_left -= usage["total_tokens"]

            context_texts = [c["text"] for c in contexts]
            o = overlap_ratio(draft, context_texts)
            f = faithfulness_score(question, context_texts, draft)
            if f is None:
                f = min(1.0, 0.6 + 0.4 * o)

            action = self._gate(f, o) if self.gate_on else GateAction.STOP

            step_log = {
                "qid": qid,
                "round": r,
                "action": action.value,
                "k": k,
                "mode": mode,
                "f": f,
                "o": o,
                "tokens_left": tokens_left,
                "usage": usage,
                "latency_ms": latency_ms,
                "prompt": [m.content for m in prompt],
                "draft": draft,
            }
            self._log_jsonl(step_log, log_path)

            if action == GateAction.STOP:
                break

            if action == GateAction.RETRIEVE_MORE:
                k = min(32, k + 4)

        p50_latency_ms = np.percentile(latencies, 50) if latencies else 0

        summary_log = {
            "qid": qid,
            "final_answer": draft,
            "final_f": f,
            "final_o": o,
            "rounds": r,
            "total_tokens": total_tokens,
            "p50_latency_ms": p50_latency_ms,
            "latencies": latencies,
        }
        self._log_jsonl(summary_log, log_path)

        return summary_log


def main():
    """A small CLI to run 3 hardcoded queries."""
    agent = Agent()
    questions = [
        "What are cats?",
        "What are dogs?",
        "What are birds?",
    ]
    for q in questions:
        print(f"--- Answering question: {q} ---")
        agent.answer(q)
        print("-" * (len(q) + 26))


if __name__ == "__main__":
    typer.run(main)
