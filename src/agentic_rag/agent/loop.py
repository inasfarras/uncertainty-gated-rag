import json
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer

from agentic_rag.agent import finalize as finalize_utils
from agentic_rag.agent.gate import GateAction, GateSignals, make_gate
from agentic_rag.agent.judge import (
    anchors_present_in_texts,
    create_judge,
    create_query_transformer,
    extract_required_anchors,
    validate_factoid_anchors,
)
from agentic_rag.agent.qanchors import (
    anchors_present_in_texts as q_anchors_present_in_texts,
)
from agentic_rag.agent.qanchors import (
    extract_required_anchors as q_extract_required_anchors,
)
from agentic_rag.agent.qanchors import (
    is_factoid as q_is_factoid,
)
from agentic_rag.config import settings
from agentic_rag.eval.signals import (
    CIT_RE,
    faithfulness_score,
    is_idk,
    sentence_support,
)
from agentic_rag.models.adapter import ChatMessage, OpenAIAdapter
from agentic_rag.prompting import build_system_instructions
from agentic_rag.prompting_reflect import build_reflect_prompt, should_reflect
from agentic_rag.retriever.vector import VectorRetriever
from agentic_rag.utils.encoder import NpEncoder
from agentic_rag.utils.timing import timer


def _assess_lexical_uncertainty(response: str, use_cache: bool = True) -> float:
    """Enhanced lexical uncertainty assessment with caching and semantic analysis."""
    if not response or len(response.strip()) < 3:
        return 1.0

    # Cache key for performance
    cache_key = hash(response) if use_cache else None
    if cache_key and hasattr(_assess_lexical_uncertainty, "_cache"):
        if cache_key in _assess_lexical_uncertainty._cache:
            return _assess_lexical_uncertainty._cache[cache_key]

    response_lower = response.lower()
    words = response_lower.split()
    total_words = len(words)

    if total_words == 0:
        return 1.0

    # Enhanced uncertainty keywords with weights
    uncertainty_indicators = {
        # High uncertainty
        "might": 0.8,
        "maybe": 0.8,
        "perhaps": 0.7,
        "possibly": 0.7,
        "unclear": 0.9,
        "uncertain": 0.9,
        "unsure": 0.8,
        "don't know": 1.0,
        "can't say": 0.9,
        "not sure": 0.8,
        # Medium uncertainty
        "likely": 0.5,
        "probably": 0.5,
        "seems": 0.6,
        "appears": 0.6,
        "suggests": 0.4,
        "indicates": 0.4,
        "could be": 0.6,
        # Hedging
        "somewhat": 0.3,
        "rather": 0.3,
        "quite": 0.2,
        "fairly": 0.3,
    }

    confidence_indicators = {
        "definitely": -0.8,
        "certainly": -0.7,
        "clearly": -0.6,
        "obviously": -0.6,
        "undoubtedly": -0.8,
        "absolutely": -0.9,
        "precisely": -0.7,
        "exactly": -0.7,
        "specifically": -0.5,
    }

    uncertainty_score = 0.0

    # Check for uncertainty phrases
    for phrase, weight in uncertainty_indicators.items():
        if phrase in response_lower:
            uncertainty_score += weight

    # Check for confidence phrases (reduce uncertainty)
    for phrase, weight in confidence_indicators.items():
        if phrase in response_lower:
            uncertainty_score += weight  # weight is negative

    # Normalize by response length (longer responses might have more indicators)
    uncertainty_score = uncertainty_score / max(1, total_words / 20)

    # Question marks might indicate uncertainty
    question_marks = response.count("?")
    if question_marks > 0:
        uncertainty_score += min(0.3, question_marks * 0.1)

    # Cache result
    if cache_key:
        if not hasattr(_assess_lexical_uncertainty, "_cache"):
            _assess_lexical_uncertainty._cache = {}  # type: ignore
        _assess_lexical_uncertainty._cache[cache_key] = min(  # type: ignore
            1.0, max(0.0, uncertainty_score)
        )

    return min(1.0, max(0.0, uncertainty_score))


def _assess_response_completeness(response: str) -> float:
    """Enhanced response completeness assessment."""
    if not response or len(response.strip()) < 3:
        return 0.0

    response = response.strip()
    length = len(response)

    # Length-based scoring (more nuanced)
    if length < 10:
        length_score = 0.1
    elif length < 30:
        length_score = 0.4
    elif length < 100:
        length_score = 0.8
    else:
        length_score = 1.0

    # Sentence structure scoring
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if not sentences:
        structure_score = 0.2
    else:
        # Check for complete sentences
        complete_sentences = sum(1 for s in sentences if len(s) > 5 and " " in s)
        structure_score = min(1.0, complete_sentences / max(1, len(sentences)))

    # Punctuation completeness
    punct_score = 1.0 if response.endswith((".", "!", "?", "]")) else 0.6

    # Check for abrupt endings or incomplete thoughts
    incomplete_indicators = ["...", "etc.", "and so on", "among others"]
    if any(indicator in response.lower() for indicator in incomplete_indicators):
        punct_score *= 0.8

    # Combine scores
    completeness = length_score * 0.4 + structure_score * 0.4 + punct_score * 0.2
    return min(1.0, completeness)


def _assess_semantic_coherence(response: str, question: str = "") -> float:
    """Assess semantic coherence of the response."""
    if not response or len(response.strip()) < 10:
        return 0.0

    # Simple coherence indicators
    sentences = [s.strip() for s in response.split(".") if s.strip()]
    if len(sentences) < 2:
        return 0.8  # Single sentence, assume coherent

    # Check for contradictory statements
    contradictory_patterns = [
        ("yes", "no"),
        ("true", "false"),
        ("correct", "incorrect"),
        ("is", "is not"),
        ("can", "cannot"),
        ("will", "will not"),
    ]

    response_lower = response.lower()
    contradiction_penalty = 0.0

    for pos, neg in contradictory_patterns:
        if pos in response_lower and neg in response_lower:
            contradiction_penalty += 0.2

    # Check for logical flow indicators
    flow_indicators = [
        "however",
        "therefore",
        "consequently",
        "furthermore",
        "moreover",
        "additionally",
        "in contrast",
        "similarly",
    ]

    flow_bonus = min(
        0.2, sum(0.05 for indicator in flow_indicators if indicator in response_lower)
    )

    # Base coherence score
    base_score = 0.7
    coherence = base_score + flow_bonus - contradiction_penalty

    return min(1.0, max(0.0, coherence))


def _assess_question_complexity(question: str) -> float:
    """Assess question complexity to adapt gate behavior."""
    if not question:
        return 0.5

    question_lower = question.lower()
    words = question_lower.split()

    # Length-based complexity
    length_complexity = min(1.0, len(words) / 30)

    # Complex question indicators
    complex_indicators = [
        "compare",
        "contrast",
        "analyze",
        "evaluate",
        "synthesize",
        "explain why",
        "how does",
        "what are the implications",
        "multiple",
        "various",
        "different",
        "relationship between",
    ]

    complexity_score = sum(
        0.1 for indicator in complex_indicators if indicator in question_lower
    )

    # Question type complexity
    if any(word in question_lower for word in ["why", "how", "explain"]):
        complexity_score += 0.2
    elif any(word in question_lower for word in ["what", "when", "where", "who"]):
        complexity_score += 0.1

    return min(1.0, length_complexity * 0.3 + complexity_score * 0.7)


def is_global_question(q: str) -> bool:
    """Determine if question requires global/comprehensive knowledge."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(q)
        return len(q.split()) >= 20 or len(doc.ents) >= 2
    except Exception:
        return len(q.split()) >= 20


def build_prompt(
    contexts: list[dict[str, Any]], question: str
) -> tuple[list[ChatMessage], str]:
    """Builds a prompt for the LLM and returns messages and a debug string."""
    # Render context blocks
    context_blocks = []
    for c in contexts:
        context_blocks.append(f"CTX[{c['id']}]:\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    system_content = build_system_instructions()

    user_content = f"QUESTION:\n{question}\n\nCONTEXT:\n{context_str}"

    debug_prompt = f"SYSTEM:\n{system_content}\n\n{user_content}"
    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ], debug_prompt


def _normalize_answer(text: str, allowed_ids: list[str]) -> str:
    # Enforce that IDK carries no citations or extra text
    tstrip = (text or "").strip()
    if is_idk(tstrip) or tstrip.lower().startswith("i don't know"):
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


def _should_force_idk(question: str, context_texts: list[str]) -> bool:
    """Heuristic: prefer abstain for underspecified superlatives/temporal questions.

    Returns True if we should force "I don't know" given the question and provided contexts.
    """
    q = (question or "").lower()
    ctx = (" \n".join(context_texts or [])).lower()

    def any_in(s: str, terms: list[str]) -> bool:
        return any(t in s for t in terms)

    superlative_terms = [
        "top ",
        " top-",
        "most ",
        "most-",
        "of all time",
        "best ",
        "average",
        "largest",
        "highest",
        "lowest",
        "number one",
        "#1",
    ]
    temporal_terms = [
        "past month",
        "first week",
        "1st qtr",
        "q1",
        "quarter",
        "in 20",
        " 2021",
        " 2022",
        " 2023",
        " 2024",
    ]
    required_anchors = [
        "rank",
        "no. 1",
        "number one",
        "winners",
        "winner",
        "academy",
        "oscar",
        "best visual effects",
        "list",
        "chart",
        "ex-dividend",
        "dividend",
        "1 month",
        "1m",
        "return",
        "%",
        "percent",
        "50-40-90",
        "top",
        "most",
        "week",
        "month",
    ]

    # If asking for songs count, but contexts have only albums, abstain
    if "songs" in q and (
        "song" not in ctx and "single" not in ctx and "singles" not in ctx
    ):
        return True

    # For 50-40-90 specific questions, require that token is present in ctx
    if "50-40-90" in q and "50-40-90" not in ctx:
        return True

    # Superlatives require anchor evidence in context
    if any_in(q, superlative_terms) and not any_in(ctx, required_anchors):
        return True

    # Temporal windows require matching temporal anchors in ctx
    if any_in(q, temporal_terms) and not any_in(
        ctx,
        [
            "2021",
            "2022",
            "2023",
            "2024",
            "week",
            "month",
            "day",
            "q1",
            "quarter",
            "jan",
            "feb",
            "mar",
            "april",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    ):
        return True

    # Finance deltas require price/series hints
    if ("rise" in q or "change" in q or "past month" in q) and not any_in(
        ctx,
        [
            "%",
            "percent",
            "change",
            "return",
            "price",
            "close",
            "open",
            "high",
            "low",
            "1 month",
            "1m",
        ],
    ):
        return True

    return False


class BaseAgent:
    def __init__(self, system: str = "base", debug_mode: bool = False):
        self.system = system
        self.debug_mode = debug_mode
        self.retriever = VectorRetriever(settings.FAISS_INDEX_PATH)
        self.llm = OpenAIAdapter()
        # Use configurable log directory
        # Prefer new lowercase setting; fallback to legacy uppercase; then default
        self.log_dir = Path(
            getattr(settings, "log_dir", None)
            or getattr(settings, "LOG_DIR", None)
            or "logs"
        )
        self.log_dir.mkdir(exist_ok=True)

    def _log_jsonl(self, data: dict[str, Any], log_path: Path):
        with open(log_path, "a") as f:
            f.write(json.dumps(data, cls=NpEncoder) + "\n")


class Baseline(BaseAgent):
    def __init__(self, debug_mode: bool = False):
        # Enforce a "vanilla" baseline configuration:
        # - FAISS-only (no hybrid BM25), no rerank, no MMR diversification
        # - Small probe factor so retrieval pool ~= top_k
        # - Keep k=8 and ~1k context cap
        try:
            settings.USE_RERANK = False
            settings.MMR_LAMBDA = 0.0
            settings.USE_HYBRID_SEARCH = False
            settings.PROBE_FACTOR = 1
            settings.RETRIEVAL_K = 8
            settings.ANCHOR_BONUS = 0.0
            settings.GATE_SEED_MISSING_ANCHORS = False
            settings.ANCHOR_GATE_ON = False
            settings.FACTOID_ONE_SHOT_RETRIEVAL = False
            # Clamp to a ~1k cap to ensure comparable packing
            settings.MAX_CONTEXT_TOKENS = 1000
        except Exception:
            # If settings is immutable for any reason, proceed with defaults
            pass

        super().__init__(system="baseline", debug_mode=debug_mode)

    def answer(self, question: str, qid: str | None = None) -> dict[str, Any]:
        qid = qid or str(uuid.uuid4())
        log_path = self.log_dir / f"{self.system}_{qid}.jsonl"

        tokens_left = settings.MAX_TOKENS_TOTAL
        k = settings.RETRIEVAL_K

        contexts, stats = self.retriever.retrieve_pack(
            question, k=k, probe_factor=settings.PROBE_FACTOR
        )
        prompt, debug_prompt = build_prompt(contexts, question)

        with timer() as t:
            draft, usage = self.llm.chat(
                messages=prompt,
                max_tokens=settings.MAX_OUTPUT_TOKENS,
                temperature=settings.TEMPERATURE,
            )
        latency_ms = t()

        tokens_left -= usage["total_tokens"]

        context_texts = [c["text"] for c in contexts]
        context_ids = [c["id"] for c in contexts]
        if _should_force_idk(question, context_texts):
            draft = "I don't know"
        draft = _normalize_answer(draft, context_ids)
        sup = sentence_support(
            draft,
            {i: t for i, t in zip(context_ids, context_texts)},
            tau_sim=settings.OVERLAP_SIM_TAU,
        )
        o = float(sup.get("overlap", 0.0))
        faith_ragas = faithfulness_score(question, context_texts, draft)
        faith_fallback = min(1.0, 0.6 + 0.4 * o)
        f = faith_ragas if faith_ragas is not None else faith_fallback

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
            "overlap_est": o,
            "o": o,
            "faith_fallback": faith_fallback,
            "faith_ragas": faith_ragas,
            "tokens_left": tokens_left,
            "usage": usage,
            "latency_ms": latency_ms,
            "prompt": [m.content for m in prompt],
            "debug_prompt": debug_prompt,
            "draft": draft,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
            "gen_tokens": usage.get("completion_tokens", 0),
        }
        self._log_jsonl(step_log, log_path)

        summary_log = {
            "qid": qid,
            "question": question,
            "final_answer": draft,
            "final_f": f,
            "final_o": o,
            "final_faith_fallback": faith_fallback,
            "final_faith_ragas": faith_ragas,
            "rounds": 1,
            "n_rounds": 1,
            "total_tokens": usage["total_tokens"],
            "p50_latency_ms": latency_ms,
            "latencies": [latency_ms],
            "contexts": contexts,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
            "action": "STOP",
            "used_judge": False,
            "judge_calls": 0,
            "final_action": "STOP",
            "debug_prompt": debug_prompt,
        }
        self._log_jsonl(summary_log, log_path)

        return summary_log


class Agent(BaseAgent):
    def __init__(self, gate_on: bool = True, debug_mode: bool = False):
        super().__init__(system="agent", debug_mode=debug_mode)
        self.gate_on = gate_on
        self.gate = make_gate(settings)
        self.judge = create_judge(self.llm, settings)
        self.query_transformer = create_query_transformer(self.llm)

    def answer(self, question: str, qid: str | None = None) -> dict[str, Any]:
        qid = qid or str(uuid.uuid4())
        log_path = self.log_dir / f"{self.system}_{qid}.jsonl"

        print(f"🔍 Starting Agent for qid: {qid}")
        print(f"📝 Question: {question[:100]}{'...' if len(question) > 100 else ''}")

        # Initialize state
        r = 0
        k = settings.RETRIEVAL_K
        mode = "vector"
        tokens_left = settings.MAX_TOKENS_TOTAL
        total_tokens = 0
        latencies = []
        seen_doc_ids: set[str] = set()
        draft = ""
        prev_overlap: float = 0.0
        prev_top_ids: set[str] = set()
        EPS_OVERLAP = settings.EPSILON_OVERLAP
        # Min ratio of new docs in retrieved set to continue
        MIN_NEW_HITS_RATIO = 0.2
        # Max allowed similarity between consecutive retrieved sets
        MAX_JACCARD_SIM = 0.8
        has_reflect_left = True
        pending_anchor_terms: list[str] = []
        force_retrieval_boost = False
        # Defaults for metrics in case no generation occurs
        f: float = 0.0
        o: float = 0.0
        faith_ragas = None  # type: ignore[assignment]
        faith_fallback: float = 0.6
        action = "INIT"
        policy = settings.JUDGE_POLICY
        max_judge_calls = getattr(settings, "JUDGE_MAX_CALLS_PER_Q", 1)
        judge_calls = 0
        any_judge_used = False

        print(f"⚙️  Config: MAX_ROUNDS={settings.MAX_ROUNDS}, GATE=UncertaintyGate")
        print(
            f"🎯 Thresholds: FAITH_TAU={settings.FAITHFULNESS_TAU}, OVERLAP_TAU={settings.OVERLAP_TAU}, UNCERTAINTY_TAU={settings.UNCERTAINTY_TAU}"
        )
        print(f"💰 Budget: {tokens_left} tokens available")

        while r < settings.MAX_ROUNDS and tokens_left > 0:
            r += 1
            print(f"\n🔄 Round {r}/{settings.MAX_ROUNDS} - Retrieving k={k} docs...")

            query_for_round = question
            seed_terms_used = False
            if self.gate_on and pending_anchor_terms and getattr(settings, "GATE_SEED_MISSING_ANCHORS", True):
                seed_terms = sorted(set(pending_anchor_terms))[:4]
                if seed_terms:
                    query_for_round = question + " " + " ".join(seed_terms)
                    print(f"dY� Seeding retrieval with missing anchors: {seed_terms}")
                    seed_terms_used = True
            if self.gate_on and force_retrieval_boost:
                bonus = getattr(settings, "GATE_RETRIEVAL_K_BONUS", 0)
                if bonus:
                    k = min(32, max(k, settings.RETRIEVAL_K + bonus))
                print(f"dY� Retrieval boost active (k={k})")
                force_retrieval_boost = False
            contexts, stats = self.retriever.retrieve_pack(
                query_for_round,
                k=k,
                exclude_doc_ids=seen_doc_ids,
                probe_factor=settings.PROBE_FACTOR,
                round_idx=r - 1,
                llm_client=self.llm,
            )
            if seed_terms_used:
                pending_anchor_terms = []
            prompt, debug_prompt = build_prompt(contexts, question)

            print(
                f"📚 Retrieved {len(contexts)} contexts, {stats.get('context_tokens', 0)} tokens"
            )
            if stats.get("used_hyde"):
                print("🔮 HyDE query rewriting applied")
            if stats.get("used_hybrid"):
                print("🔍 Hybrid search (Vector + BM25) applied")
            if stats.get("used_rerank"):
                print("🏆 BGE cross-encoder reranking applied")
            if stats.get("used_mmr"):
                print("🎯 MMR diversification applied")

            # Pre-generation novelty check to avoid extra LLM calls
            retrieved_ids = cast(list[str], stats.get("retrieved_ids", []))
            curr_top_ids = set(retrieved_ids)
            new_hits = [d for d in retrieved_ids if d not in seen_doc_ids]
            has_new_hits = len(new_hits) > 0
            new_hits_ratio = (
                (len(new_hits) / max(1, len(retrieved_ids))) if retrieved_ids else 0.0
            )
            inter = len(prev_top_ids & curr_top_ids)
            union = len(prev_top_ids | curr_top_ids) or 1
            jaccard_sim = inter / union

            # Strengthen early stops: check before LLM call
            print(
                f"🔍 Novelty check: {len(new_hits)} new hits ({new_hits_ratio:.2f} ratio), Jaccard={jaccard_sim:.2f}"
            )
            if r > 1 and (
                (new_hits_ratio < MIN_NEW_HITS_RATIO) or (jaccard_sim > MAX_JACCARD_SIM)
            ):
                print(
                    f"⏹️  EARLY STOP: Low novelty (ratio={new_hits_ratio:.2f} < {MIN_NEW_HITS_RATIO} or jaccard={jaccard_sim:.2f} > {MAX_JACCARD_SIM})"
                )
                action = "STOP_PRE_LOW_NOVELTY"
                step_log = {
                    "qid": qid,
                    "round": r,
                    "action": action,
                    "k": k,
                    "mode": mode,
                    "tokens_left": tokens_left,
                    "usage": None,
                    "latency_ms": 0,
                    "prompt": [m.content for m in prompt],
                    "debug_prompt": debug_prompt,
                    "draft": draft if "draft" in locals() else None,
                    "retrieved_ids": retrieved_ids,
                    "new_hits": new_hits,
                    "has_new_hits": has_new_hits,
                    "new_hits_ratio": new_hits_ratio,
                    "jaccard": jaccard_sim,
                    "n_ctx_blocks": stats.get("n_ctx_blocks"),
                    "context_tokens": stats.get("context_tokens"),
                    "reason": "PRE_LOW_NOVELTY",
                    "used_gate": "uncertainty",
                    "uncertainty_score": None,
                }
                self._log_jsonl(step_log, log_path)
                prev_top_ids = curr_top_ids
                break

            # Optional: Pre-generation Judge to assess context sufficiency
            judge_assessment = None
            allow_pregen = (
                getattr(settings, "JUDGE_PREGEN", True)
                and policy == "always"
                and judge_calls < max_judge_calls
            )
            if allow_pregen:
                print("🧠 Pre-generation Judge: assessing context sufficiency...")
                judge_assessment = self.judge.assess_context_sufficiency(
                    question, contexts, round_idx=r - 1
                )
                judge_calls += 1
                any_judge_used = True
                print(
                    f"🧠 Judge (pre-gen): sufficient={judge_assessment.is_sufficient}, "
                    f"conf={judge_assessment.confidence:.2f}, action={judge_assessment.suggested_action}"
                )

                # If clearly insufficient, optionally try transformation or skip generation for this round
                if (not judge_assessment.is_sufficient) and (
                    judge_assessment.confidence > 0.7
                ):
                    did_transform = False
                    if (
                        judge_assessment.suggested_action == "TRANSFORM_QUERY"
                        and r == 1
                    ):
                        print(
                            "🔄 Judge suggests transformation (pre-gen). Trying remedial retrieval..."
                        )
                        transformations = self.query_transformer.transform_query(
                            question, judge_assessment, contexts
                        )
                        if transformations:
                            for i, transformed_query in enumerate(transformations[:2]):
                                print(
                                    f"🔄 [Pre-gen] Transformation {i + 1}: {transformed_query[:80]}..."
                                )
                                transform_contexts, transform_stats = (
                                    self.retriever.retrieve_pack(
                                        transformed_query,
                                        k=k,
                                        exclude_doc_ids=seen_doc_ids,
                                        probe_factor=settings.PROBE_FACTOR,
                                        round_idx=r,
                                        llm_client=self.llm,
                                    )
                                )
                                transform_ids = cast(
                                    list[str], transform_stats.get("retrieved_ids", [])
                                )
                                new_transform_hits = [
                                    d for d in transform_ids if d not in seen_doc_ids
                                ]
                                if new_transform_hits:
                                    print(
                                        f"🔄 [Pre-gen] Transformation {i + 1} added {len(new_transform_hits)} new contexts"
                                    )
                                    contexts.extend(transform_contexts[:3])
                                    seen_doc_ids.update(new_transform_hits)
                                    enhanced_judge = (
                                        self.judge.assess_context_sufficiency(
                                            question, contexts, round_idx=r - 1
                                        )
                                    )
                                    if enhanced_judge.is_sufficient:
                                        print(
                                            "✅ [Pre-gen] Contexts now sufficient after transformation"
                                        )
                                        judge_assessment = enhanced_judge
                                        did_transform = True
                                        break
                    if not did_transform:
                        if r < settings.MAX_ROUNDS:
                            # Skip generation and go retrieve more next round
                            action = GateAction.RETRIEVE_MORE
                            step_log = {
                                "qid": qid,
                                "round": r,
                                "action": "SKIP_GEN_RETRIEVE",
                                "k": k,
                                "mode": mode,
                                "tokens_left": tokens_left,
                                "usage": None,
                                "latency_ms": 0,
                                "retrieved_ids": retrieved_ids,
                                "new_hits": new_hits,
                                "has_new_hits": has_new_hits,
                                "new_hits_ratio": new_hits_ratio,
                                "jaccard": jaccard_sim,
                                "n_ctx_blocks": stats.get("n_ctx_blocks"),
                                "context_tokens": stats.get("context_tokens"),
                                "reason": "PREGEN_JUDGE_INSUFFICIENT",
                                "used_gate": "uncertainty",
                            }
                            self._log_jsonl(step_log, log_path)
                            # Prepare for next round
                            k = min(32, k + 4)
                            prev_top_ids = curr_top_ids
                            print(
                                f"⏭️  Skipping generation this round due to insufficient context; next k={k}"
                            )
                            continue
                        else:
                            # Last round and still insufficient: abstain to produce a stable summary
                            draft = "I don't know"
                            f = 0.0
                            o = 0.0
                            action = GateAction.ABSTAIN
                            step_log = {
                                "qid": qid,
                                "round": r,
                                "action": "STOP_PREGEN_JUDGE_ABSTAIN",
                                "k": k,
                                "mode": mode,
                                "tokens_left": tokens_left,
                                "usage": None,
                                "latency_ms": 0,
                                "retrieved_ids": retrieved_ids,
                                "new_hits": new_hits,
                                "has_new_hits": has_new_hits,
                                "new_hits_ratio": new_hits_ratio,
                                "jaccard": jaccard_sim,
                                "n_ctx_blocks": stats.get("n_ctx_blocks"),
                                "context_tokens": stats.get("context_tokens"),
                                "reason": "PREGEN_JUDGE_INSUFFICIENT_LAST_ROUND",
                                "used_gate": "uncertainty",
                            }
                            self._log_jsonl(step_log, log_path)
                            print(
                                "⛔ Last round with insufficient context — abstaining."
                            )
                            break

            print(f"🤖 Generating answer... (budget: {tokens_left} tokens left)")
            with timer() as t:
                draft, usage = self.llm.chat(
                    messages=prompt,
                    max_tokens=settings.MAX_OUTPUT_TOKENS,
                    temperature=settings.TEMPERATURE,
                )
            latency_ms = t()
            latencies.append(latency_ms)

            print(
                f"✅ Generated answer in {latency_ms:.0f}ms, used {usage['total_tokens']} tokens"
            )

            total_tokens += usage["total_tokens"]
            tokens_left -= usage["total_tokens"]

            context_texts = [c["text"] for c in contexts]
            context_ids = [c["id"] for c in contexts]
            if _should_force_idk(question, context_texts):
                draft = "I don't know"
            draft = _normalize_answer(draft, context_ids)
            print("📊 Evaluating answer quality...")
            sup = sentence_support(
                draft,
                {i: t for i, t in zip(context_ids, context_texts)},
                tau_sim=settings.OVERLAP_SIM_TAU,
            )
            o = float(sup.get("overlap", 0.0))
            print(f"📈 Overlap score: {o:.3f} (threshold: {settings.OVERLAP_TAU})")

            # Enhanced Judge Assessment - Always invoke for first round
            # Note: judge_assessment may have been computed pre-generation
            faith_ragas = None
            faith_fallback = min(1.0, 0.6 + 0.4 * o)

            # Always use judge for first round or when policy demands it
            should_use_judge = False
            if judge_assessment is None and judge_calls < max_judge_calls:
                if policy == "always":
                    should_use_judge = True
                elif policy == "gray_zone" and settings.TAU_LO <= o < settings.TAU_HI:
                    should_use_judge = True

            if should_use_judge:
                print("🧠 Invoking Judge for context sufficiency assessment...")
                judge_assessment = self.judge.assess_context_sufficiency(
                    question, contexts, round_idx=r - 1
                )
                judge_calls += 1
                any_judge_used = True
                print(
                    f"🧠 Judge assessment: sufficient={judge_assessment.is_sufficient}, "
                    f"confidence={judge_assessment.confidence:.3f}"
                )
                print(f"🧠 Judge reasoning: {judge_assessment.reasoning[:100]}...")

                # Also compute faithfulness score for compatibility
                faith_ragas = faithfulness_score(question, context_texts, draft)

            f = faith_ragas if faith_ragas is not None else faith_fallback
            print(
                f"🎯 Faithfulness score: {f:.3f} (threshold: {settings.FAITHFULNESS_TAU})"
            )

            retrieved_ids = cast(list[str], stats.get("retrieved_ids", []))
            new_hits = [d for d in retrieved_ids if d not in seen_doc_ids]
            has_new_hits = len(new_hits) > 0

            # Prepare gate extras early so logging never breaks
            gate_extras: dict[str, Any] = {}
            # Initialize anchor variables for logging regardless of branch
            qtype = finalize_utils.detect_type(question)
            required_anchors = []  # type: list[str]
            anchor_cov = 0.0
            anchor_missing: list[str] = []
            validators: dict[str, Any] = {}
            used_anchor_constrained_search = False

            try:
                required_anchors = list(extract_required_anchors(question))
                present_set, anchor_cov = anchors_present_in_texts(
                    set(required_anchors), [c["text"] for c in contexts]
                )
                anchor_missing = sorted(list(set(required_anchors) - present_set))
                validators = validate_factoid_anchors(
                    question, [c["text"] for c in contexts]
                )
            except Exception:
                # Keep defaults if any helper fails
                pass

            # Short-circuit: no new hits
            short_reason: str | None = None
            if r > 1 and not has_new_hits:
                print("⏹️  SHORT CIRCUIT: No new hits")
                short_reason = "NO_NEW_HITS"
                action = "STOP_SHORT_CIRCUIT"
            # Short-circuit: overlap stagnation
            elif r > 1 and (o - prev_overlap) < EPS_OVERLAP:
                print(
                    f"⏹️  SHORT CIRCUIT: Overlap stagnant ({o:.3f} - {prev_overlap:.3f} = {o - prev_overlap:.3f} < {EPS_OVERLAP})"
                )
                short_reason = "OVERLAP_STAGNANT"
                action = "STOP_STAGNANT"
            else:
                print("🚪 Consulting Enhanced UncertaintyGate...")

                # Enhanced uncertainty assessments with caching
                lexical_uncertainty = _assess_lexical_uncertainty(draft, use_cache=True)
                completeness = _assess_response_completeness(draft)
                semantic_coherence = _assess_semantic_coherence(draft, question)
                question_complexity = _assess_question_complexity(question)

                # Populate gate extras during full gate evaluation
                gate_extras = {}

                # Integrate Judge signals into gate if available
                if judge_assessment:
                    gate_extras["judge_sufficient"] = judge_assessment.is_sufficient
                    gate_extras["judge_confidence"] = judge_assessment.confidence
                    gate_extras["judge_action"] = judge_assessment.suggested_action
                    # New: pass deeper quality signals
                    if getattr(judge_assessment, "anchor_coverage", None) is not None:
                        gate_extras["anchor_coverage"] = (
                            judge_assessment.anchor_coverage
                        )
                    if getattr(judge_assessment, "conflict_risk", None) is not None:
                        gate_extras["conflict_risk"] = judge_assessment.conflict_risk
                    if getattr(judge_assessment, "mismatch_flags", None) is not None:
                        gate_extras["mismatch_flags"] = judge_assessment.mismatch_flags

                    # If judge says insufficient with high confidence, consider query transformation
                    if (
                        not judge_assessment.is_sufficient
                        and judge_assessment.confidence > 0.7
                        and judge_assessment.suggested_action == "TRANSFORM_QUERY"
                        and r == 1
                    ):  # Only on first round to avoid loops
                        print(
                            "🔄 Judge suggests query transformation - attempting remedial retrieval..."
                        )

                        # Get query transformations
                        transformations = self.query_transformer.transform_query(
                            question, judge_assessment, contexts
                        )

                        if transformations:
                            print(
                                f"🔄 Trying {len(transformations)} query transformations..."
                            )
                            for i, transformed_query in enumerate(
                                transformations[:2]
                            ):  # Try up to 2
                                print(
                                    f"🔄 Transformation {i + 1}: {transformed_query[:80]}..."
                                )

                                # Retrieve with transformed query
                                transform_contexts, transform_stats = (
                                    self.retriever.retrieve_pack(
                                        transformed_query,
                                        k=k,
                                        exclude_doc_ids=seen_doc_ids,
                                        probe_factor=settings.PROBE_FACTOR,
                                        round_idx=r,
                                        llm_client=self.llm,
                                    )
                                )

                                # Check if we got new/better contexts
                                transform_ids = cast(
                                    list[str], transform_stats.get("retrieved_ids", [])
                                )
                                new_transform_hits = [
                                    d for d in transform_ids if d not in seen_doc_ids
                                ]

                                if new_transform_hits:
                                    print(
                                        f"🔄 Transformation {i + 1} found {len(new_transform_hits)} new contexts"
                                    )
                                    # Merge new contexts with existing ones
                                    contexts.extend(
                                        transform_contexts[:3]
                                    )  # Add top 3 new contexts
                                    seen_doc_ids.update(new_transform_hits)

                                    # Re-assess with enhanced contexts
                                    enhanced_judge = (
                                        self.judge.assess_context_sufficiency(
                                            question, contexts, round_idx=r - 1
                                        )
                                    )
                                    if enhanced_judge.is_sufficient:
                                        print(
                                            "🔄 Query transformation successful - contexts now sufficient!"
                                        )
                                        judge_assessment = enhanced_judge
                                        gate_extras["used_query_transformation"] = True
                                        gate_extras["successful_transformation"] = (
                                            transformed_query
                                        )
                                        break

                # Anchor extraction & validation for factoids
                qtype = finalize_utils.detect_type(question)
                required_anchors = list(extract_required_anchors(question))
                present_set, anchor_cov = anchors_present_in_texts(
                    set(required_anchors), context_texts
                )
                anchor_missing = sorted(list(set(required_anchors) - present_set))
                validators = validate_factoid_anchors(question, context_texts)
                gate_extras["required_anchors"] = required_anchors
                gate_extras["anchor_coverage"] = anchor_cov
                gate_extras["missing_anchors"] = anchor_missing
                gate_extras.update(validators)
                fail_time = bool(validators.get("fail_time"))
                fail_unit = bool(validators.get("fail_unit"))
                fail_event = bool(validators.get("fail_event"))
                missing_anchors = list(gate_extras.get("missing_anchors") or anchor_missing or [])

                signals = GateSignals(
                    faith=f,
                    overlap=o,
                    lexical_uncertainty=lexical_uncertainty,
                    completeness=completeness,
                    semantic_coherence=semantic_coherence,
                    answer_length=len(draft),
                    question_complexity=question_complexity,
                    budget_left_tokens=tokens_left,
                    round_idx=r - 1,
                    has_reflect_left=has_reflect_left,
                    novelty_ratio=new_hits_ratio,
                    extras=gate_extras,
                )

                print(
                    f"📊 Enhanced metrics: lex_unc={lexical_uncertainty:.3f}, "
                    f"completeness={completeness:.3f}, coherence={semantic_coherence:.3f}, "
                    f"q_complexity={question_complexity:.3f}"
                )
                if self.gate_on:
                    action = self.gate.decide(signals)
                    print(f"🚪 Gate decision: {action}")
                else:
                    action = GateAction.RETRIEVE_MORE
                    print("🚫 Gate OFF - continuing to next round")
                if "uncertainty_score" in gate_extras:
                    print(
                        f"🌡️  Enhanced uncertainty score: {gate_extras['uncertainty_score']:.3f}"
                    )
                    if "adaptive_weights" in gate_extras:
                        weights = gate_extras["adaptive_weights"]
                        print(
                            f"⚖️  Adaptive weights: faith={weights.get('faith', 0):.2f}, "
                            f"overlap={weights.get('overlap', 0):.2f}, semantic={weights.get('semantic', 0):.2f}"
                        )
                    if "cache_hit_rate" in gate_extras:
                        print(f"💾 Cache hit rate: {gate_extras['cache_hit_rate']:.2f}")

            action_str = str(action)
            coverage_min = getattr(settings, "BAUG_STOP_COVERAGE_MIN", 0.3)
            stop_blocked = (
                fail_time
                or fail_unit
                or fail_event
                or float(anchor_cov) < coverage_min
                or o < settings.OVERLAP_TAU
            )
            if self.gate_on and stop_blocked and action_str.startswith("STOP"):
                if tokens_left >= getattr(settings, "FACTOID_MIN_TOKENS_LEFT", 300):
                    print("dY", "Low-support STOP blocked - retrieving more")
                    action = GateAction.RETRIEVE_MORE
                    gate_extras["stop_reason"] = "LOW_SUPPORT"
                    force_retrieval_boost = True
                    if getattr(settings, "GATE_SEED_MISSING_ANCHORS", True):
                        seed_pool = missing_anchors or required_anchors
                        if seed_pool:
                            pending_anchor_terms = list(seed_pool[:4])
                else:
                    print("dY", "Low-support STOP but budget exhausted - abstaining")
                    action = GateAction.ABSTAIN
                    gate_extras["stop_reason"] = "LOW_SUPPORT_NO_BUDGET"
            overlap_gain = o - prev_overlap if r > 1 else o
            min_overlap_gain = getattr(settings, "OVERLAP_IMPROVEMENT_MIN", 0.0)
            if (
                self.gate_on
                and action == GateAction.STOP
                and r > 1
                and overlap_gain >= min_overlap_gain
                and o < settings.OVERLAP_TAU
                and tokens_left >= getattr(settings, "FACTOID_MIN_TOKENS_LEFT", 300)
            ):
                print("dY", "Overlap improving but below tau - continuing")
                action = GateAction.RETRIEVE_MORE
                gate_extras["stop_reason"] = "OVERLAP_IMPROVING"
                force_retrieval_boost = True
                if getattr(settings, "GATE_SEED_MISSING_ANCHORS", True):
                    seed_pool = missing_anchors or required_anchors
                    if seed_pool:
                        pending_anchor_terms = list(seed_pool[:4])

            # If about to STOP or ABSTAIN on a factoid with missing anchors, optionally try one anchor-constrained retrieval
            if (
                self.gate_on
                and (action == GateAction.ABSTAIN or str(action).startswith("STOP"))
                and getattr(settings, "FACTOID_ONE_SHOT_RETRIEVAL", True)
                and tokens_left >= getattr(settings, "FACTOID_MIN_TOKENS_LEFT", 300)
                and (q_is_factoid(question) or qtype in ("date", "number", "entity"))
            ):
                cov = float(gate_extras.get("anchor_coverage", 1.0) or 1.0)
                missing_anchors = list(gate_extras.get("missing_anchors") or anchor_missing or [])
                if fail_time or fail_unit or fail_event or cov < 0.5 or missing_anchors:
                    new_query = question
                    # Recompute anchors using lean helpers for robustness
                    try:
                        required_anchors = list(q_extract_required_anchors(question))
                        present_set = q_anchors_present_in_texts(
                            [c["text"] for c in contexts], set(required_anchors)
                        )
                        anchor_missing = sorted(
                            list(set(required_anchors) - present_set)
                        )
                    except Exception:
                        pass
                    if anchor_missing:
                        new_query = (
                            question + " " + " ".join(sorted(anchor_missing)[:4])
                        )
                    print(
                        f"🧲 Anchor-constrained retrieval: missing={anchor_missing[:4]} (cov={cov:.2f})"
                    )
                    used_anchor_constrained_search = True
                    # Retrieve with constrained query
                    transform_contexts, transform_stats = self.retriever.retrieve_pack(
                        new_query,
                        k=k,
                        exclude_doc_ids=seen_doc_ids,
                        probe_factor=settings.PROBE_FACTOR,
                        round_idx=r,  # subsequent round
                        llm_client=self.llm,
                    )
                    transform_ids = cast(
                        list[str], transform_stats.get("retrieved_ids", [])
                    )
                    new_transform_hits = [
                        d for d in transform_ids if d not in seen_doc_ids
                    ]
                    if new_transform_hits:
                        print(
                            f"🧲 Anchor-constrained search found {len(new_transform_hits)} new docs"
                        )
                        # Merge new contexts and regenerate once
                        contexts = transform_contexts[
                            : max(2, min(4, len(transform_contexts)))
                        ]
                        context_texts = [c["text"] for c in contexts]
                        context_ids = [c["id"] for c in contexts]
                        prompt, debug_prompt = build_prompt(contexts, question)
                        with timer() as t2:
                            draft, usage = self.llm.chat(
                                messages=prompt,
                                max_tokens=settings.MAX_OUTPUT_TOKENS,
                                temperature=settings.TEMPERATURE,
                            )
                        latency_ms = t2()
                        latencies.append(latency_ms)
                        total_tokens += usage["total_tokens"]
                        tokens_left -= usage["total_tokens"]
                        if _should_force_idk(question, context_texts):
                            draft = "I don't know"
                        draft = _normalize_answer(draft, context_ids)
                        sup = sentence_support(
                            draft,
                            {i: t for i, t in zip(context_ids, context_texts)},
                            tau_sim=settings.OVERLAP_SIM_TAU,
                        )
                        o = float(sup.get("overlap", 0.0))
                        f = (
                            faith_ragas
                            if faith_ragas is not None
                            else min(1.0, 0.6 + 0.4 * o)
                        )
                        print(
                            f"🧲 After anchor-constrained generation: f={f:.3f}, o={o:.3f}"
                        )
                        action = (
                            GateAction.STOP
                            if (
                                f >= settings.FAITHFULNESS_TAU
                                and o >= settings.OVERLAP_TAU
                            )
                            else GateAction.RETRIEVE_MORE
                        )

            # Final-round safeguard: if gate says RETRIEVE_MORE on last round, attempt one anchor-constrained rescue
            if (
                action == GateAction.RETRIEVE_MORE
                and r == settings.MAX_ROUNDS
                and getattr(settings, "FACTOID_ONE_SHOT_RETRIEVAL", True)
                and tokens_left >= getattr(settings, "FACTOID_MIN_TOKENS_LEFT", 300)
                and (q_is_factoid(question) or qtype in ("date", "number", "entity"))
            ):
                try:
                    rq = list(q_extract_required_anchors(question))
                    ctx_txt = [c["text"] for c in contexts]
                    present = q_anchors_present_in_texts(ctx_txt, set(rq))
                    miss = sorted(list(set(rq) - set(present)))
                except Exception:
                    miss = []
                if miss:
                    print(
                        f"⏳ Final-round rescue: trying anchor-constrained retrieval (missing={miss[:4]})"
                    )
                    new_query = question + " " + " ".join(miss[:4])
                    transform_contexts, transform_stats = self.retriever.retrieve_pack(
                        new_query,
                        k=k,
                        exclude_doc_ids=seen_doc_ids,
                        probe_factor=settings.PROBE_FACTOR,
                        round_idx=r,
                        llm_client=self.llm,
                    )
                    contexts = (
                        transform_contexts[: max(2, min(4, len(transform_contexts)))]
                        or contexts
                    )
                    context_ids = [c["id"] for c in contexts]
                    context_texts = [c["text"] for c in contexts]
                    prompt, debug_prompt = build_prompt(contexts, question)
                    with timer() as t2:
                        draft, usage = self.llm.chat(
                            messages=prompt,
                            max_tokens=settings.MAX_OUTPUT_TOKENS,
                            temperature=settings.TEMPERATURE,
                        )
                    latency_ms = t2()
                    latencies.append(latency_ms)
                    total_tokens += usage["total_tokens"]
                    tokens_left -= usage["total_tokens"]
                    if _should_force_idk(question, context_texts):
                        draft = "I don't know"
                    draft = _normalize_answer(draft, context_ids)
                    sup = sentence_support(
                        draft,
                        {i: t for i, t in zip(context_ids, context_texts)},
                        tau_sim=settings.OVERLAP_SIM_TAU,
                    )
                    o = float(sup.get("overlap", 0.0))
                    f = (
                        faith_ragas
                        if faith_ragas is not None
                        else min(1.0, 0.6 + 0.4 * o)
                    )
                    action = (
                        GateAction.STOP
                        if (
                            f >= settings.FAITHFULNESS_TAU and o >= settings.OVERLAP_TAU
                        )
                        else GateAction.ABSTAIN
                    )
                    used_anchor_constrained_search = True
                    print(
                        f"⏹️  Final-round decision after anchor search: {action} (f={f:.3f}, o={o:.3f})"
                    )

            # Zero-overlap rescue (one extra try before reflection)
            if (
                self.gate_on
                and action == GateAction.ABSTAIN
                and o == 0.0
                and tokens_left >= getattr(settings, "ZERO_RESCUE_MIN_TOKENS", 2200)
                and (q_is_factoid(question) or qtype in ("date", "number", "entity"))
            ):
                rescue_terms = [
                    t
                    for t in (
                        gate_extras.get("missing_anchors")
                        or required_anchors
                        or list(q_extract_required_anchors(question))
                        or []
                    )
                    if t
                ]
                additional = sorted(set(rescue_terms))[:4]
                if not additional:
                    print("dY", "Zero-overlap rescue skipped: no anchor terms")
                    if not gate_extras.get("stop_reason"):
                        gate_extras["stop_reason"] = "NO_SUPPORT"
                else:
                    print("dY", "Triggering zero-overlap rescue...")
                    rescue_query = question + " " + " ".join(additional)
                    if "countries" in question.lower() and "countries" not in rescue_query.lower():
                        rescue_query += " countries list"
                    rescue_k = min(
                        32,
                        max(
                            k + getattr(settings, "ZERO_RESCUE_K_BONUS", 2),
                            settings.RETRIEVAL_K,
                        ),
                    )
                    rescue_contexts, rescue_stats = self.retriever.retrieve_pack(
                        rescue_query,
                        k=rescue_k,
                        exclude_doc_ids=seen_doc_ids,
                        probe_factor=settings.PROBE_FACTOR,
                        round_idx=r,
                        llm_client=self.llm,
                    )
                    stats = rescue_stats
                    k = rescue_k
                    rescue_ids = [c["id"] for c in rescue_contexts]
                    seen_doc_ids.update(rescue_ids)
                    contexts = rescue_contexts[: max(2, min(4, len(rescue_contexts)))] or contexts
                    context_ids = [c["id"] for c in contexts]
                    context_texts = [c["text"] for c in contexts]
                    prompt, debug_prompt = build_prompt(contexts, question)
                    with timer() as t_rescue:
                        draft, usage = self.llm.chat(
                            messages=prompt,
                            max_tokens=settings.MAX_OUTPUT_TOKENS,
                            temperature=settings.TEMPERATURE,
                        )
                    rescue_latency = t_rescue()
                    latencies.append(rescue_latency)
                    total_tokens += usage["total_tokens"]
                    tokens_left -= usage["total_tokens"]
                    if _should_force_idk(question, context_texts):
                        draft = "I don't know"
                    draft = _normalize_answer(draft, context_ids)
                    sup = sentence_support(
                        draft,
                        {i: t for i, t in zip(context_ids, context_texts)},
                        tau_sim=settings.OVERLAP_SIM_TAU,
                    )
                    o = float(sup.get("overlap", 0.0))
                    f = (
                        faith_ragas if faith_ragas is not None else min(1.0, 0.6 + 0.4 * o)
                    )
                    if f >= settings.FAITHFULNESS_TAU and o >= settings.OVERLAP_TAU:
                        action = GateAction.STOP
                        gate_extras["stop_reason"] = "ZERO_RESCUE_SUCCESS"
                        print("dY", f"Zero-overlap rescue succeeded (f={f:.3f}, o={o:.3f})")
                    else:
                        gate_extras["stop_reason"] = "ZERO_RESCUE_NO_SUPPORT"
                        print("dY", f"Zero-overlap rescue still lacking support (f={f:.3f}, o={o:.3f})")

            # Handle REFLECT action
            if should_reflect(action, has_reflect_left):
                print("🤔 Applying REFLECT to improve answer...")
                reflect_messages, reflect_debug = build_reflect_prompt(
                    contexts, draft, required_anchors=required_anchors
                )
                with timer() as t_reflect:
                    reflected_answer, reflect_usage = self.llm.chat(
                        messages=reflect_messages,
                        max_tokens=settings.MAX_OUTPUT_TOKENS,
                        temperature=settings.TEMPERATURE,
                    )
                reflect_latency_ms = t_reflect()
                latencies.append(reflect_latency_ms)

                # Normalize the reflected answer
                context_ids = [c["id"] for c in contexts]
                context_texts = [c["text"] for c in contexts]
                if _should_force_idk(question, context_texts):
                    reflected_answer = "I don't know"
                reflected_answer = _normalize_answer(reflected_answer, context_ids)

                # Update draft with reflected answer
                draft = reflected_answer
                total_tokens += reflect_usage["total_tokens"]
                tokens_left -= reflect_usage["total_tokens"]
                has_reflect_left = False  # Only one reflection per query

                # Re-evaluate after reflection
                sup = sentence_support(
                    draft,
                    {i: t for i, t in zip(context_ids, context_texts)},
                    tau_sim=settings.OVERLAP_SIM_TAU,
                )
                o = float(sup.get("overlap", 0.0))
                f = faith_ragas if faith_ragas is not None else min(1.0, 0.6 + 0.4 * o)

                # Log reflection step
                reflect_log = {
                    "qid": qid,
                    "round": r,
                    "action": "REFLECT_APPLIED",
                    "k": k,
                    "mode": mode,
                    "f": f,
                    "overlap_est": o,
                    "o": o,
                    "faith_fallback": min(1.0, 0.6 + 0.4 * o),
                    "faith_ragas": faith_ragas,
                    "tokens_left": tokens_left,
                    "usage": reflect_usage,
                    "latency_ms": reflect_latency_ms,
                    "draft": draft,
                    "reflected_from": (
                        step_log.get("draft") if "step_log" in locals() else None
                    ),
                    "used_gate": "enhanced_uncertainty",
                    "uncertainty_score": gate_extras.get("uncertainty_score"),
                    "semantic_coherence": semantic_coherence,
                    "question_complexity": question_complexity,
                    "adaptive_weights": gate_extras.get("adaptive_weights"),
                    "required_anchors": required_anchors,
                    "present_anchor_rate": anchor_cov,
                    "missing_anchors": anchor_missing,
                }
                self._log_jsonl(reflect_log, log_path)

                # After reflection, decide whether to stop or continue
                if f >= settings.FAITHFULNESS_TAU and o >= settings.OVERLAP_TAU:
                    action = GateAction.STOP
                    print(
                        f"✅ REFLECT improved quality - STOPPING (f={f:.3f}, o={o:.3f})"
                    )
                else:
                    action = GateAction.RETRIEVE_MORE
                    print(
                        f"🔄 REFLECT done, but still need more - CONTINUING (f={f:.3f}, o={o:.3f})"
                    )

            # Debug logging for first round only
            if self.debug_mode and r == 1:
                from agentic_rag.eval.debug import log_debug_info

                prompt_messages = [
                    {"role": m.role, "content": m.content} for m in prompt
                ]
                log_debug_info(qid, question, prompt_messages, draft, context_ids)

            step_log = {
                "qid": qid,
                "round": r,
                "action": action,
                "k": k,
                "mode": mode,
                "f": f,
                "overlap_est": o,
                "o": o,
                "faith_fallback": faith_fallback,
                "faith_ragas": faith_ragas,
                "tokens_left": tokens_left,
                "usage": usage,
                "latency_ms": latency_ms,
                "prompt": [m.content for m in prompt],
                "debug_prompt": debug_prompt,
                "draft": draft,
                "retrieved_ids": retrieved_ids,
                "new_hits": new_hits,
                "has_new_hits": has_new_hits,
                "n_ctx_blocks": stats.get("n_ctx_blocks"),
                "context_tokens": stats.get("context_tokens"),
                "reason": short_reason or gate_extras.get("stop_reason"),
                "gen_tokens": usage.get("completion_tokens", 0),
                "used_gate": "uncertainty",
                "uncertainty_score": gate_extras.get("uncertainty_score"),
                "new_hits_ratio": new_hits_ratio,
                "cache_hit_rate": gate_extras.get("cache_hit_rate"),
                "used_hyde": stats.get("used_hyde"),
                "used_hybrid": stats.get("used_hybrid"),
                "used_rerank": stats.get("used_rerank"),
                "used_mmr": stats.get("used_mmr"),
                "required_anchors": required_anchors,
                "anchor_coverage": anchor_cov,
                "missing_anchors": anchor_missing,
                "fail_time": (
                    validators.get("fail_time") if "validators" in locals() else None
                ),
                "fail_unit": (
                    validators.get("fail_unit") if "validators" in locals() else None
                ),
                "fail_event": (
                    validators.get("fail_event") if "validators" in locals() else None
                ),
                "used_anchor_constrained_search": used_anchor_constrained_search,
            }
            self._log_jsonl(step_log, log_path)
            seen_doc_ids.update(curr_top_ids)

            if (
                action
                in (
                    "STOP",
                    "STOP_SHORT_CIRCUIT",
                    "STOP_STAGNANT",
                    "STOP_PRE_LOW_NOVELTY",
                )
                or action == GateAction.STOP
                or action == GateAction.STOP_LOW_BUDGET
            ):
                print(f"🏁 STOPPING with action: {action}")
                break

            if action == GateAction.RETRIEVE_MORE:
                k = min(32, k + 4)
                print(f"🔄 CONTINUING to round {r + 1} with k={k}")
            prev_overlap = o
            prev_top_ids = curr_top_ids

        p50_latency_ms = np.percentile(latencies, 50) if latencies else 0

        stop_reason = short_reason or gate_extras.get("stop_reason")
        if action == GateAction.ABSTAIN and o == 0.0 and self.gate_on:
            stop_reason = stop_reason or "NO_SUPPORT"

        stop_reason = short_reason or gate_extras.get("stop_reason")
        if action == GateAction.ABSTAIN and o == 0.0 and self.gate_on:
            stop_reason = stop_reason or "NO_SUPPORT"

        print("\n📋 Final Summary:")
        print(f"   Rounds completed: {r}")
        print(f"   Total tokens used: {total_tokens}")
        print(f"   Final overlap: {o:.3f}")
        print(f"   Final faithfulness: {f:.3f}")
        print(f"   Unique docs seen: {len(seen_doc_ids)}")
        print(f"   Final action: {action}")

        if action == GateAction.ABSTAIN or str(action).upper().startswith("STOP_PREGEN_JUDGE_ABSTAIN"):
            draft = "I don't know"
            final_short = "i don't know"
        else:
            try:
                final_short = finalize_utils.finalize_short_answer(question, draft)
            except Exception:
                final_short = None

        summary_log = {
            "qid": qid,
            "question": question,
            "final_answer": draft,
            "final_short": final_short,
            "reason": stop_reason,
            "final_f": f,
            "final_o": o,
            "final_faith_fallback": faith_fallback,
            "final_faith_ragas": faith_ragas,
            "rounds": r,
            "n_rounds": r,
            "total_tokens": total_tokens,
            "p50_latency_ms": p50_latency_ms,
            "latencies": latencies,
            "contexts": contexts,
            "retrieved_ids": stats.get("retrieved_ids"),
            "n_ctx_blocks": stats.get("n_ctx_blocks"),
            "context_tokens": stats.get("context_tokens"),
            "action": action,
            "reason": stop_reason,
            "debug_prompt": debug_prompt,
            "uniq_docs_seen": len(seen_doc_ids),
            "used_judge": any_judge_used,
            "judge_calls": judge_calls,
            "final_action": action,
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
