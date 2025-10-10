from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

from agentic_rag.agent.finalize import finalize_short_answer
from agentic_rag.agent.judge import create_judge
from agentic_rag.anchors.predictor import propose_anchors
from agentic_rag.anchors.validators import (
    AWARD_TOKENS,
    award_tournament_requirements,
    list_requirements,
    units_time_requirements,
)
from agentic_rag.anchors.validators import (
    conflict_risk as estimate_conflict_risk,
)
from agentic_rag.anchors.validators import (
    coverage as anchor_coverage,
)
from agentic_rag.anchors.validators import (
    mismatch_flags as anchor_mismatch_flags,
)
from agentic_rag.config import settings
from agentic_rag.data.meta import load_meta
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.eval.signals import (
    CIT_RE,
    faithfulness_score,
    is_idk,
    sentence_support,
)
from agentic_rag.gate.adapter import BAUGAdapter
from agentic_rag.intent.interpreter import interpret
from agentic_rag.models.adapter import ChatMessage, OpenAIAdapter, get_openai
from agentic_rag.prompting import ContextBlock, pack_context
from agentic_rag.prompting_reflect import build_reflect_prompt, should_reflect
from agentic_rag.retrieval.agent import Path, RetrievalAgent
from agentic_rag.telemetry.recorder import log_round, log_summary
from agentic_rag.utils.timing import timer


def _build_prompt(
    contexts: list[dict[str, Any]], question: str
) -> tuple[list[ChatMessage], str]:
    blocks = [f"CTX[{c['id']}]:\n{c['text']}" for c in contexts]
    context_str = "\n\n".join(blocks)
    system = (
        "You answer ONLY using the provided CONTEXT.\n"
        "If information is missing, answer EXACTLY: I don't know.\n"
        "Limit your answer to 1–2 sentences.\n"
        "Each non-IDK sentence MUST include exactly one citation in the form [CIT:<doc_id>].\n"
        "If you answer I don't know (or Tidak tahu), do not include any citation.\n"
        "Citation format must be exactly [CIT:<doc_id>] where <doc_id> matches the CTX header and uses only letters, digits, _, or -.\n"
        "When the question requires an average or a numeric derived from a table or per-season rows (e.g., 3-point attempts per game across specific seasons), extract the numbers directly from the CONTEXT and compute the requested value.\n"
        "Return a single concise sentence with exactly one [CIT:<doc_id>] pointing to the chunk that contains the numbers you used.\n"
        "Be precise and concise; avoid extra sentences."
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context_str}"
    dbg = f"SYSTEM:\n{system}\n\n{user}"
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ], dbg


def _normalize_answer(text: str, allowed_ids: list[str]) -> str:
    t = (text or "").strip()
    if is_idk(t) or t.lower().startswith("i don't know"):
        return "I don't know"

    allowed: set[str] = set(allowed_ids)

    def _repl(m: Any) -> str:  # type: ignore[no-redef]
        cid = m.group("id")
        return m.group(0) if cid in allowed else ""

    try:
        return CIT_RE.sub(_repl, t)
    except Exception:
        return t


class AnchorSystem:
    """Thin multi-agent orchestrator with BAUG as final policy.

    Keeps default MAX_ROUNDS small and performs single generation per round
    (temp=0, top_p=0). REFLECT is optional and only used when BAUG requests it.
    """

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.retriever = RetrievalAgent()
        self.llm = get_openai()
        self.baug = BAUGAdapter()
        self.judge = create_judge(cast(OpenAIAdapter, self.llm), settings)
        # Load CRAG page metadata (url/title) for richer logs
        self.meta_map = load_meta()

    def answer(self, question: str, qid: str | None = None) -> dict[str, Any]:
        qid = qid or str(uuid.uuid4())

        tokens_left = settings.MAX_TOKENS_TOTAL
        total_tokens = 0
        latencies: list[int] = []

        seen_doc_ids: set[str] = set()
        intent = interpret(
            question,
            llm_budget_ok=tokens_left >= settings.FACTOID_MIN_TOKENS_LEFT,
        )
        llm_calls = (
            1
            if (
                intent.source_of_intent != "rule_only"
                or any(
                    flag in intent.ambiguity_flags
                    for flag in ("invalid_llm_json", "llm_call_failed")
                )
            )
            else 0
        )
        anchors = propose_anchors(intent, top_m=6)
        # Always include a global retrieval pass (raw question) to emulate baseline coverage
        # then add up to 5 best anchors (total ~6 passes)
        top_anchor_texts = [a["text"] for a in anchors[:5]]
        selected_anchors = [""] + top_anchor_texts

        validators_state: dict[str, Any] = {
            "award": {},
            "numeric": {},
            "passed": True,
        }

        round_idx = 0
        draft = ""
        used_judge = False
        final_action = "INIT"
        has_reflect_left = True
        prev_overlap = 0.0
        prev_anchor_cov = 0.0
        did_constrained_retrieval = False

        # Simple loop bounded by MAX_ROUNDS
        while round_idx < settings.MAX_ROUNDS and tokens_left > 0:
            round_idx += 1
            # Explore per anchor (parallel, one hop)
            paths: list[Path] = []
            max_workers = max(1, min(settings.MAX_WORKERS, len(selected_anchors)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(self.retriever.explore, a, question, 1, seen_doc_ids)
                    for a in selected_anchors
                ]
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                        paths.extend(res)
                    except Exception:
                        continue

            # Merge contexts from all paths (assemble as dicts, convert to ContextBlock before pack)
            all_ctx: list[dict[str, Any]] = []
            retrieved_ids: list[str] = []
            novelty_ratios = []
            fine_scores_all: list[float] = []
            for p in paths:
                for c in p.get("passages", []):
                    doc_id = c.get("id", "")
                    meta = self.meta_map.get(doc_id, {})
                    all_ctx.append(
                        {
                            "id": doc_id,
                            "text": c.get("text", ""),
                            "score": float(c.get("score", 0.0)),
                            "url": meta.get("url"),
                            "title": meta.get("title"),
                            "rank": meta.get("rank"),
                        }
                    )
                retrieved_ids.extend(p.get("doc_ids", []))
                novelty_ratios.append(float(p.get("novelty_ratio", 0.0)))
                fine_scores_all.extend([float(x) for x in p.get("fine_scores", [])])

            # Targeted helper retrieval: if question mentions 50-40-90 (or similar),
            # and contexts list specific seasons, try an extra pass to pull per-game table chunks
            try:
                import re as _re

                ql = (question or "").lower()
                has_504090 = bool(_re.search(r"\b\d{2}[-\/]\d{2}[-\/]\d{2}\b", ql)) or (
                    "50-40-90" in ql or "50/40/90" in ql or "50–40–90" in ql
                )
                seasons = []
                if all_ctx:
                    joined = " \n".join([c.get("text", "") for c in all_ctx])
                    seasons = _re.findall(r"\b20\d{2}[-–]\d{2}\b", joined)
                if has_504090 and seasons:
                    # Add tokens likely to appear in per-game tables
                    helper_terms = ["3PA", "3P", "three-point attempts", "per game"]
                    helper_anchor = " ".join(
                        list(dict.fromkeys(seasons + helper_terms))[:10]
                    )
                    extra_paths = self.retriever.explore(
                        helper_anchor, question, 1, seen_doc_ids
                    )
                    for p in extra_paths:
                        for c in p.get("passages", []):
                            doc_id = c.get("id", "")
                            meta = self.meta_map.get(doc_id, {})
                            all_ctx.append(
                                {
                                    "id": doc_id,
                                    "text": c.get("text", ""),
                                    "score": float(c.get("score", 0.0))
                                    + 0.5,  # small boost for helper hits
                                    "url": meta.get("url"),
                                    "title": meta.get("title"),
                                    "rank": meta.get("rank"),
                                }
                            )
            except Exception:
                pass

            # Fine filter summary
            fine_median = float(np.median(fine_scores_all)) if fine_scores_all else 0.0

            # Pack under context token cap using uniform fine-sim to the question
            # Compute fine similarity for each candidate block to make scores comparable across anchors
            if all_ctx:
                try:
                    q_emb = embed_texts([question])[0]
                    ctx_embs = embed_texts([c.get("text", "") for c in all_ctx])
                    fine_sims = (ctx_embs @ q_emb).tolist()
                except Exception:
                    fine_sims = [float(c.get("score", 0.0)) for c in all_ctx]
                for blk, fs in zip(all_ctx, fine_sims):
                    blk["fine_sim"] = float(fs)

            # Pre-pack domain checks and optional reserves when coverage is low
            reserved_anchor_slots = 0
            if all_ctx:
                try:
                    cov_all, present_all, missing_all = anchor_coverage(
                        question, [c.get("text", "") for c in all_ctx]
                    )
                except Exception:
                    cov_all, present_all, missing_all = 0.0, set(), set()

                # Domain-specific requirements across all candidates
                req_aw = award_tournament_requirements(
                    question, [c.get("text", "") for c in all_ctx]
                )
                req_ut = units_time_requirements(
                    question, [c.get("text", "") for c in all_ctx]
                )

                if (
                    settings.PACK_RESERVE_ON_LOW_COVERAGE
                    and cov_all < settings.ANCHOR_COVERAGE_TAU
                    and settings.RESERVE_ANCHOR_SLOTS > 0
                ):
                    req_tokens: set[str] = set(present_all | set(missing_all))
                    if req_aw.get("missing"):
                        req_tokens |= set(AWARD_TOKENS)
                    if req_ut.get("missing"):
                        req_tokens |= {
                            "per game",
                            "%",
                            "percent",
                            "q1",
                            "q2",
                            "q3",
                            "q4",
                        }

                    scored = []
                    for blk in all_ctx:
                        t = (blk.get("text", "") or "").lower()
                        hit = sum(1 for tok in req_tokens if tok in t)
                        scored.append((hit, blk))
                    scored.sort(
                        key=lambda x: (x[0], x[1].get("score", 0.0)), reverse=True
                    )
                    for hit, blk in scored[: settings.RESERVE_ANCHOR_SLOTS]:
                        if hit > 0:
                            blk["score"] = float(blk.get("score", 0.0)) + 10.0
                            reserved_anchor_slots += 1

            # Meta-aware features (lightweight):
            # - title_match: 1 if title contains anchor tokens
            # - rank_score: small score based on page rank (higher for better ranks)
            # - list_dense hint for enumeration queries
            def _anchors_from_question(q: str) -> set[str]:
                import re as _re

                s = (q or "").lower()
                toks: set[str] = set(_re.findall(r"\b(?:19|20)\d{2}\b", s))
                for t in [
                    "oscar",
                    "academy",
                    "best",
                    "grand slam",
                    "per game",
                    "%",
                    "percent",
                ]:
                    if t in s:
                        toks.add(t)
                # ambil kata kapital (2 kata) sebagai anchor kasar
                caps = _re.findall(
                    r"\b([A-Z][A-Za-z0-9’'\-]*(?:\s+[A-Z][A-Za-z0-9’'\-]*){0,2})\b",
                    question or "",
                )
                if caps:
                    toks.update({c.lower() for c in caps})
                return toks

            anchor_tokens = _anchors_from_question(question)
            list_like = (
                (question or "").lower().startswith("what are")
                or (question or "").lower().startswith("list ")
                or "list the" in (question or "").lower()
            )

            # Compute title_match, rank_score, list_dense and final pack score via weights
            w_fine = float(getattr(settings, "PACK_W_FINE", 0.9))
            w_title = float(getattr(settings, "PACK_W_TITLE", 0.05))
            w_rank = float(getattr(settings, "PACK_W_RANK", 0.05))

            for blk in all_ctx:
                title = (blk.get("title") or "").lower()
                title_match = (
                    1.0
                    if (title and any(tok in title for tok in anchor_tokens))
                    else 0.0
                )
                try:
                    r = blk.get("rank")
                    if isinstance(r, int) and r is not None:
                        # simple rank score: 1 for top-2, 0.6 for top-5, else 0
                        rank_score = 1.0 if r <= 2 else (0.6 if r <= 5 else 0.0)
                    else:
                        rank_score = 0.0
                except Exception:
                    rank_score = 0.0
                if list_like:
                    text = (blk.get("text") or "").strip()
                    # heuristik list-dense: jumlah koma/semicolon/bullet ≥ 8 atau title mengandung 'list'
                    dense = (
                        text.count(",")
                        + text.count(";")
                        + text.count("•")
                        + text.count("–")
                    )
                    list_dense = (
                        1.0 if (dense >= 8 or (title and "list" in title)) else 0.0
                    )
                else:
                    list_dense = 0.0
                fine = float(blk.get("fine_sim", 0.0))
                # Final score fusion
                blk["score"] = (
                    w_fine * fine
                    + w_title * (title_match + list_dense)
                    + w_rank * rank_score
                )
            # Type-aware budgets
            eff_cap = settings.MAX_CONTEXT_TOKENS
            eff_reserve = settings.RESERVE_ANCHOR_SLOTS
            if list_like:
                eff_cap = getattr(settings, "LIST_CAP_TOKENS", eff_cap)
                eff_reserve = getattr(settings, "LIST_RESERVE_SLOTS", eff_reserve)
            else:
                # simple factoid heuristic
                ql = (question or "").lower()
                is_factoid = any(
                    t in ql
                    for t in [
                        "how many",
                        "how much",
                        "what year",
                        "when",
                        "average",
                        "%",
                        "per game",
                    ]
                ) or bool(re.search(r"\b(19|20)\d{2}\b", ql))
                if is_factoid:
                    eff_cap = getattr(settings, "FACTOID_CAP_TOKENS", eff_cap)
                    eff_reserve = getattr(
                        settings, "FACTOID_RESERVE_SLOTS", eff_reserve
                    )

            # Per-domain cap (≤N blok/domain) sebelum pack untuk memperlebar cakupan
            sorted_ctx = sorted(
                all_ctx, key=lambda b: float(b.get("score", 0.0)), reverse=True
            )

            # Deduplicate by exact context id, keeping highest-score occurrence
            seen_ids: set[str] = set()
            dedup_sorted_ctx: list[dict[str, Any]] = []
            for b in sorted_ctx:
                bid = str(b.get("id", ""))
                if not bid:
                    continue
                if bid in seen_ids:
                    continue
                seen_ids.add(bid)
                dedup_sorted_ctx.append(b)
            kept_raw: list[dict[str, Any]] = []
            domain_count: dict[str, int] = {}
            per_domain_limit = max(1, int(getattr(settings, "PACK_MAX_PER_DOMAIN", 2)))
            per_doc_limit = (
                max(1, int(getattr(settings, "PACK_MAX_PER_DOC_LIST", 2)))
                if list_like
                else None
            )
            doc_count: dict[str, int] = {}
            for b in dedup_sorted_ctx:
                url = b.get("url") or ""
                host = urlparse(url).netloc if url else ""
                if host:
                    if domain_count.get(host, 0) >= per_domain_limit:
                        continue
                    domain_count[host] = domain_count.get(host, 0) + 1
                if per_doc_limit is not None:
                    doc_id = b.get("id") or ""
                    stem = doc_id.split("__")[0] if "__" in doc_id else doc_id
                    if stem:
                        if doc_count.get(stem, 0) >= per_doc_limit:
                            continue
                        doc_count[stem] = doc_count.get(stem, 0) + 1
                kept_raw.append(b)

            # Build ContextBlock list from raw dicts (id, text, score only)
            kept_blocks: list[ContextBlock] = [
                {
                    "id": str(b.get("id", "")),
                    "text": str(b.get("text", "")),
                    "score": float(b.get("score", 0.0)),
                }
                for b in kept_raw
            ]
            packed, context_tokens, n_blocks = pack_context(
                kept_blocks, max_tokens_cap=eff_cap
            )
            contexts = [cast(dict[str, Any], dict(b)) for b in packed]
            # Enrich packed contexts with meta if missing (defensive)
            for ctx in contexts:
                mid = ctx.get("id", "")
                if mid and "url" not in ctx:
                    meta = self.meta_map.get(mid, {})
                    if meta:
                        ctx["url"] = meta.get("url")
                        ctx["title"] = meta.get("title")
                        ctx["rank"] = meta.get("rank")
            new_hits = [d for d in retrieved_ids if d not in seen_doc_ids]
            new_hits_ratio = (
                (len(new_hits) / max(1, len(retrieved_ids))) if retrieved_ids else 0.0
            )

            # Build prompt and get a draft answer (single generation per round)
            messages, debug_prompt = _build_prompt(contexts, question)
            with timer() as tmr:
                draft, usage = self.llm.chat(
                    messages=messages,
                    max_tokens=settings.MAX_OUTPUT_TOKENS,
                    temperature=settings.TEMPERATURE,
                )
            llm_calls += 1
            latency_ms = tmr()
            latencies.append(latency_ms)
            total_tokens += usage.get("total_tokens", 0)
            tokens_left -= usage.get("total_tokens", 0)

            # Normalize answer citations
            context_ids = [str(c.get("id", "")) for c in contexts]
            draft = _normalize_answer(draft, context_ids)
            final_short = finalize_short_answer(question, draft)

            # Estimate support and faith
            ctx_map = {str(c.get("id", "")): str(c.get("text", "")) for c in contexts}
            sup = sentence_support(draft, ctx_map, tau_sim=settings.OVERLAP_SIM_TAU)
            overlap_est = float(sup.get("overlap", 0.0))
            faith_ragas = faithfulness_score(
                question, [str(c.get("text", "")) for c in contexts], draft
            )
            faith_est = (
                faith_ragas
                if faith_ragas is not None
                else min(1.0, 0.6 + 0.4 * overlap_est)
            )

            # Anchor validators and judge (gray-zone policy ≤ 1 call)
            try:
                cov, present, missing = anchor_coverage(
                    question, [str(c.get("text", "")) for c in contexts]
                )
            except Exception:
                cov, present, missing = 0.0, set(), set()
            try:
                mismatches = anchor_mismatch_flags(
                    question, [str(c.get("text", "")) for c in contexts]
                )
            except Exception:
                mismatches = {
                    "temporal_mismatch": False,
                    "unit_mismatch": False,
                    "entity_mismatch": False,
                }
            try:
                conf_risk = estimate_conflict_risk(
                    [str(c.get("text", "")) for c in contexts]
                )
            except Exception:
                conf_risk = 0.0

            texts_for_validation = [str(c.get("text", "")) for c in contexts]
            award_req = award_tournament_requirements(question, texts_for_validation)
            numeric_req = units_time_requirements(question, texts_for_validation)
            list_req = list_requirements(question, texts_for_validation)
            validators_passed = not (
                award_req.get("missing")
                or numeric_req.get("missing")
                or list_req.get("missing")
            )
            validators_snapshot = {
                "award": award_req,
                "numeric": numeric_req,
                "list": list_req,
                "passed": validators_passed,
            }
            validators_state.update(validators_snapshot)
            judge_extras: dict[str, Any] = {"validators": validators_snapshot.copy()}
            if (settings.JUDGE_POLICY == "always") or (
                settings.JUDGE_POLICY == "gray_zone"
                and not used_judge
                and 0.4 <= overlap_est <= 0.6
            ):
                try:
                    judge_assessment = self.judge.assess_context_sufficiency(
                        question, contexts, round_idx=round_idx - 1
                    )
                    llm_calls += 1
                    used_judge = True
                    judge_extras.update(
                        {
                            "judge_sufficient": judge_assessment.is_sufficient,
                            "judge_confidence": judge_assessment.confidence,
                            "judge_action": judge_assessment.suggested_action,
                            "anchor_coverage": (
                                float(judge_assessment.anchor_coverage)
                                if judge_assessment.anchor_coverage is not None
                                else cov
                            ),
                            "conflict_risk": (
                                float(judge_assessment.conflict_risk)
                                if judge_assessment.conflict_risk is not None
                                else conf_risk
                            ),
                            "mismatch_flags": judge_assessment.mismatch_flags
                            or mismatches,
                            "required_anchors": judge_assessment.required_anchors
                            or list(present | set(missing)),
                        }
                    )
                    # Prefer judge coverage/risk
                    cov = float(judge_extras.get("anchor_coverage", cov))
                    conf_risk = float(judge_extras.get("conflict_risk", conf_risk))
                    mismatches = judge_extras.get("mismatch_flags", mismatches)
                except Exception:
                    pass

            # Completeness/coherence proxies
            lexical_uncertainty = 0.0  # optional future hook
            completeness = 1.0 if len(draft.split()) > 3 else 0.6
            semantic_coherence = 1.0

            # BAUG decision
            if "validators" in judge_extras:
                judge_extras["validators"]["passed"] = validators_passed
            else:
                judge_extras["validators"] = {"passed": validators_passed}
            judge_extras.setdefault(
                "intent",
                {
                    "task_type": intent.task_type,
                    "core_entities": intent.core_entities,
                    "slots": intent.slots,
                },
            )
            baug_signals = {
                "overlap_est": overlap_est,
                "faith_est": float(faith_est),
                "new_hits_ratio": float(new_hits_ratio),
                "anchor_coverage": float(cov),
                "conflict_risk": float(conf_risk),
                "budget_left": int(tokens_left),
                "round_idx": int(round_idx - 1),
                "has_reflect_left": bool(has_reflect_left),
                "lexical_uncertainty": float(lexical_uncertainty),
                "completeness": float(completeness),
                "semantic_coherence": float(semantic_coherence),
                "answer_length": len(draft),
                "question_complexity": 0.5,
                "intent_confidence": float(intent.intent_confidence),
                "slot_completeness": float(intent.slot_completeness),
                "source_of_intent": intent.source_of_intent,
                "validators_passed": bool(validators_passed),
                "extras": judge_extras,
                "fine_median": fine_median,
            }
            # Gate control: allow disabling BAUG via settings.ANCHOR_GATE_ON
            if getattr(settings, "ANCHOR_GATE_ON", True):
                action = self.baug.decide(baug_signals)
                baug_reasons = self.baug.last_reasons()
            else:
                # When gate is OFF: follow a simple policy
                # - If more rounds allowed, request RETRIEVE_MORE, else STOP
                action = "RETRIEVE_MORE" if round_idx < settings.MAX_ROUNDS else "STOP"
                baug_reasons = ["gate_off"]
            baug_signals["baug_reasons"] = baug_reasons
            final_action = action

            # Update seen ids
            seen_doc_ids.update(retrieved_ids)

            # Round log with structured BAUG decision
            log_round(
                qid,
                round_idx,
                {
                    "system": "anchor",
                    "anchors_proposed": anchors,
                    "anchors_selected": selected_anchors,
                    "path_count": len(paths),
                    "new_hits_ratio": new_hits_ratio,
                    "pack_sort": "fine_sim",
                    "reserved_anchor_slots": reserved_anchor_slots,
                    "context_tokens": context_tokens,
                    "n_ctx_blocks": n_blocks,
                    "retrieved_ids": retrieved_ids,
                    "draft": draft,
                    "final_short": final_short or "",
                    "overlap_est": overlap_est,
                    "faith_est": float(faith_est),
                    "fine_median": fine_median,
                    "tokens_left": tokens_left,
                    "intent_confidence": float(intent.intent_confidence),
                    "slot_completeness": float(intent.slot_completeness),
                    "source_of_intent": intent.source_of_intent,
                    "validators_passed": validators_passed,
                    "validators": validators_snapshot,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "action": action,
                    "baug_reasons": baug_reasons,
                    "gate_kind": self.baug.kind(),
                    "gate_last_decision": self.baug.last_decision() or "",
                    # Structured BAUG decision for analysis
                    "baug_decision": {
                        "action": action,
                        "reasons": baug_reasons,
                        "signals": {
                            "overlap_est": overlap_est,
                            "faith_est": float(faith_est),
                            "anchor_coverage": float(cov),
                            "new_hits_ratio": float(new_hits_ratio),
                            "conflict_risk": float(conf_risk),
                            "budget_left": int(tokens_left),
                            "has_reflect_left": bool(has_reflect_left),
                        },
                        "thresholds": {
                            "overlap_tau": settings.OVERLAP_TAU,
                            "faithfulness_tau": settings.FAITHFULNESS_TAU,
                            "coverage_min": getattr(
                                settings, "BAUG_STOP_COVERAGE_MIN", 0.3
                            ),
                            "new_hits_eps": settings.NEW_HITS_EPS,
                        },
                        "round_idx": round_idx - 1,  # 0-indexed for BAUG
                        "gate_kind": self.baug.kind(),
                    },
                    "debug_prompt": debug_prompt if self.debug_mode else "",
                },
            )

            # Early termination checks within supervisor (CR-like)
            # Only stop early if we have low new hits AND low quality signals
            short_reason: str | None = None
            if not getattr(settings, "ANCHOR_GATE_ON", True):
                if (
                    round_idx > 1 and validators_passed
                ):  # Only check for early stop if validators are OK
                    if new_hits_ratio < settings.NEW_HITS_EPS:
                        short_reason = "NO_NEW_HITS"
                    elif (overlap_est - prev_overlap) < settings.EPSILON_OVERLAP:
                        short_reason = "OVERLAP_STAGNANT"
                    elif (cov - prev_anchor_cov) < settings.ANCHOR_PLATEAU_EPS:
                        short_reason = "ANCHOR_PLATEAU"
                    elif fine_median < settings.FINE_FILTER_TAU:
                        short_reason = "FINE_FILTER_FLOOR"

            # One-shot constrained retrieval if anchors missing and budget allows
            if (
                not short_reason
                and not did_constrained_retrieval
                and tokens_left > settings.FACTOID_MIN_TOKENS_LEFT
                and (cov < settings.ANCHOR_COVERAGE_TAU or not validators_passed)
            ):
                did_constrained_retrieval = True
                # pick strongest anchor
                best_anchor = selected_anchors[0] if selected_anchors else ""
                if not validators_passed:
                    slot_candidates = [v for v in intent.slots.values() if v]
                    if slot_candidates:
                        best_anchor = slot_candidates[0]
                try:
                    extra_paths = self.retriever.explore(
                        best_anchor, question, hop_budget=1, seen_doc_ids=seen_doc_ids
                    )
                    # Merge minimal extra contexts
                    for p in extra_paths:
                        for c in p.get("passages", []):
                            contexts.append(
                                {
                                    "id": c.get("id", ""),
                                    "text": c.get("text", ""),
                                    "score": float(c.get("score", 0.0)),
                                }
                            )
                            retrieved_ids.append(c.get("id", ""))
                except Exception:
                    pass

            # Decide next step
            if action == "REFLECT" and should_reflect(action, has_reflect_left):
                has_reflect_left = False
                msgs, dbg = build_reflect_prompt(contexts, draft, required_anchors=None)
                with timer() as tmr2:
                    draft2, usage2 = self.llm.chat(
                        messages=msgs,
                        max_tokens=settings.MAX_OUTPUT_TOKENS,
                        temperature=settings.TEMPERATURE,
                    )
                llm_calls += 1
                latency_ms2 = tmr2()
                latencies.append(latency_ms2)
                total_tokens += usage2.get("total_tokens", 0)
                tokens_left -= usage2.get("total_tokens", 0)
                draft = _normalize_answer(draft2, context_ids)
                final_action = "REFLECT"
                # After reflect, stop this minimal version
                break
            elif (
                action == "RETRIEVE_MORE"
                and round_idx < settings.MAX_ROUNDS
                and not short_reason
            ):
                # loop continues
                prev_overlap = overlap_est
                prev_anchor_cov = cov
                continue
            else:
                # STOP or ABSTAIN
                break

        # Summary
        summary: dict[str, Any] = {
            "qid": qid,
            "question": question,
            "final_answer": draft,
            "final_short": final_short or "",
            "rounds": round_idx,
            "n_rounds": round_idx,
            "total_tokens": total_tokens,
            "p50_latency_ms": int(np.median(latencies)) if latencies else 0,  # type: ignore[name-defined]
            "latencies": latencies,
            "contexts": contexts if locals().get("contexts") is not None else [],
            "retrieved_ids": list(seen_doc_ids),
            "n_ctx_blocks": locals().get("n_blocks", 0),
            "context_tokens": locals().get("context_tokens", 0),
            "action": final_action,
            "baug_reasons": baug_reasons,
            "used_judge": used_judge,
            "final_action": final_action,
            "debug_prompt": locals().get("debug_prompt", "") if self.debug_mode else "",
            "anchor_coverage": float(locals().get("cov", 0.0)),
            "conflict_risk": float(locals().get("conf_risk", 0.0)),
            "intent_confidence": float(intent.intent_confidence),
            "slot_completeness": float(intent.slot_completeness),
            "source_of_intent": intent.source_of_intent,
            "validators_passed": bool(validators_state.get("passed", True)),
            "validators": dict(validators_state),
            "llm_calls": llm_calls,
            "stop_reason": locals().get("short_reason") or final_action,
        }

        log_summary(qid, summary)
        return summary
