import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import typer

from agentic_rag.config import settings
from agentic_rag.eval.signals import (
    CIT_RE,
    faithfulness_score,
    is_idk,
    overlap_ratio,
)
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


def build_prompt(contexts: List[ContextChunk], question: str) -> tuple[List[ChatMessage], str]:
    """Builds a prompt for the LLM and returns messages and a debug string."""
    # Render context blocks
    context_blocks = []
    for c in contexts:
        context_blocks.append(f"CTX[{c['id']}]:\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    system_content = (
        "You answer ONLY using the provided CONTEXT.\n"
        "If information is missing, answer EXACTLY: I don't know.\n"
        "When you DO provide information, EVERY sentence MUST end with [CIT:<doc_id>], where <doc_id> is one of the CTX ids below.\n"
        "Do NOT add any citation to 'I don't know.'\n"
        "Keep answers concise and factual."
    )

    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_str}"
    )

    debug_prompt = f"SYSTEM:\n{system_content}\n\n{user_content}"
    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ], debug_prompt


def _normalize_answer(text: str, allowed_ids: List[str]) -> str:
    if is_idk(text):
        return "I don't know"
    # Remove any non-conforming citations; keep only [CIT:<allowed_id>]
    def _repl(m: Any) -> str:  # type: ignore[name-defined]
        cid = m.group("id")
        return m.group(0) if cid in set(allowed_ids) else ""

    # CIT_RE is compiled with named group 'id'
    try:
        return CIT_RE.sub(_repl, text)
    except Exception:
        # Fallback: naive strip of [CIT:...]
        return text


class GateAction(str, Enum):
    STOP = "STOP"
    STOP_LOW_CONF = "STOP_LOW_CONF"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    SWITCH_GRAPH = "SWITCH_GRAPH"  # Not implemented
    REFLECT = "REFLECT"  # Not implemented


class BaseAgent:
    def __init__(self, system: str = "base", debug_mode: bool = False):
        self.system = system
        self.debug_mode = debug_mode
        self.retriever = VectorRetriever(settings.FAISS_INDEX_PATH)
        self.llm = OpenAIAdapter()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def _log_jsonl(self, data: Dict[str, Any], log_path: Path):
        with open(log_path, "a") as f:
            f.write(json.dumps(data, cls=NpEncoder) + "\n")


class Baseline(BaseAgent):
    def __init__(self, debug_mode: bool = False):
        super().__init__(system="baseline", debug_mode=debug_mode)

    def answer(self, question: str, qid: str | None = None) -> Dict[str, Any]:
        qid = qid or str(uuid.uuid4())
        log_path = self.log_dir / f"{self.system}_{qid}.jsonl"

        tokens_left = settings.MAX_TOKENS_TOTAL
        k = settings.RETRIEVAL_K

        contexts, stats = self.retriever.retrieve(question, k=k)
        prompt, debug_prompt = build_prompt(contexts, question)

        with timer() as t:
            draft, usage = self.llm.chat(messages=prompt, max_tokens=256, temperature=0.0)
        latency_ms = t()

        tokens_left -= usage["total_tokens"]

        context_texts = [c["text"] for c in contexts]
        context_ids = [c["id"] for c in contexts]
        draft = _normalize_answer(draft, context_ids)
        o = overlap_ratio(draft, context_texts, context_ids=context_ids)
        f = faithfulness_score(question, context_texts, draft)
        if f is None:
            f = min(1.0, 0.6 + 0.4 * o)

        # Debug logging
        if self.debug_mode:
            from agentic_rag.eval.debug import log_debug_info

            prompt_messages = [{"role": m.role, "content": m.content} for m in prompt]
            log_debug_info(qid, question, prompt_messages, draft, context_ids)

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
            "debug_prompt": debug_prompt,
            "draft": draft,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
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
            "contexts": contexts,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
            "action": "STOP",
            "debug_prompt": debug_prompt,
        }
        self._log_jsonl(summary_log, log_path)

        return summary_log


class Agent(BaseAgent):
    def __init__(self, gate_on: bool = True, debug_mode: bool = False):
        super().__init__(system="agent", debug_mode=debug_mode)
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

            contexts, stats = self.retriever.retrieve(question, k=k)
            prompt, debug_prompt = build_prompt(contexts, question)

            with timer() as t:
                draft, usage = self.llm.chat(messages=prompt, max_tokens=256, temperature=0.0)
            latency_ms = t()
            latencies.append(latency_ms)

            total_tokens += usage["total_tokens"]
            tokens_left -= usage["total_tokens"]

            context_texts = [c["text"] for c in contexts]
            context_ids = [c["id"] for c in contexts]
            draft = _normalize_answer(draft, context_ids)
            o = overlap_ratio(draft, context_texts, context_ids=context_ids)
            f = faithfulness_score(question, context_texts, draft)
            if f is None:
                f = min(1.0, 0.6 + 0.4 * o)

            # Debug logging for first round only
            if self.debug_mode and r == 1:
                from agentic_rag.eval.debug import log_debug_info

                prompt_messages = [
                    {"role": m.role, "content": m.content} for m in prompt
                ]
                log_debug_info(qid, question, prompt_messages, draft, context_ids)

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
                "debug_prompt": debug_prompt,
                "draft": draft,
                "retrieved_ids": stats.get("retrieved_ids"),
                "n_ctx_blocks": stats.get("n_ctx_blocks"),
                "context_tokens": stats.get("context_tokens"),
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
            "contexts": contexts,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
            "action": action.value,
            "debug_prompt": debug_prompt,
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
