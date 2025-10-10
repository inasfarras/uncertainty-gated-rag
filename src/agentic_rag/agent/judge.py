"""
Judge module for assessing context sufficiency and triggering remedial actions.

This module implements a lightweight LLM-based judge that evaluates whether
retrieved context is sufficient to answer a given question.
"""

import json
from typing import Any, Optional

from agentic_rag.config import Settings
from agentic_rag.models.adapter import ChatMessage, OpenAIAdapter


class ContextSufficiencyResult:
    """Result of context sufficiency assessment."""

    def __init__(
        self,
        is_sufficient: bool,
        confidence: float,
        reasoning: str,
        suggested_action: str,
        query_transformations: Optional[list[str]] = None,
        # New: deeper quality signals
        anchor_coverage: float | None = None,
        conflict_risk: float | None = None,
        mismatch_flags: Optional[dict[str, bool]] = None,
        required_anchors: Optional[list[str]] = None,
    ):
        self.is_sufficient = is_sufficient
        self.confidence = confidence
        self.reasoning = reasoning
        self.suggested_action = suggested_action
        self.query_transformations = query_transformations or []
        self.anchor_coverage = anchor_coverage
        self.conflict_risk = conflict_risk
        self.mismatch_flags = mismatch_flags or {}
        self.required_anchors = required_anchors or []


class Judge:
    """
    Judge module that assesses context sufficiency and suggests remedial actions.

    The Judge evaluates whether retrieved context contains sufficient information
    to answer a question accurately. If insufficient, it suggests query transformations
    or decompositions to improve retrieval.
    """

    def __init__(self, llm_client: OpenAIAdapter, settings: Settings):
        self.llm = llm_client
        self.settings = settings

    def assess_context_sufficiency(
        self, question: str, contexts: list[dict[str, Any]], round_idx: int = 0
    ) -> ContextSufficiencyResult:
        """
        Assess whether the retrieved contexts are sufficient to answer the question.

        Args:
            question: The original user question
            contexts: List of retrieved context blocks with 'id' and 'text'
            round_idx: Current retrieval round (0-based)

        Returns:
            ContextSufficiencyResult with assessment details
        """
        # Build context summary for the judge
        context_summary = self._build_context_summary(contexts)

        # Create judge prompt
        prompt = self._build_judge_prompt(question, context_summary, round_idx)

        try:
            # Get judge assessment
            response, usage = self.llm.chat(
                messages=prompt,
                max_tokens=200,  # Judge needs less output than generation
                temperature=0.0,  # Deterministic assessment
            )

            # Parse judge response
            result = self._parse_judge_response(response, question)

            # Heuristic post-checks to guard against being "faithful to bad context"
            anchors = self._extract_anchors(question)
            coverage, missing, temporal_mismatch = self._compute_anchor_coverage(
                anchors, [c.get("text", "") for c in contexts]
            )
            unit_mismatch = self._detect_unit_mismatch(question, contexts)
            entity_mismatch = self._detect_entity_mismatch(question, contexts)

            # Attach signals
            result.anchor_coverage = coverage
            # If LLM provided conflict risk, keep it; else compute a simple proxy
            if result.conflict_risk is None:
                result.conflict_risk = self._estimate_conflict_risk(contexts)
            result.required_anchors = anchors
            result.mismatch_flags.update(
                {
                    "temporal_mismatch": temporal_mismatch,
                    "unit_mismatch": unit_mismatch,
                    "entity_mismatch": entity_mismatch,
                }
            )

            # Enforce stricter insufficiency when anchors are missing or mismatched
            try:
                tau_cov = float(self.settings.ANCHOR_COVERAGE_TAU)
            except Exception:
                tau_cov = 0.6
            try:
                tau_conf = float(self.settings.CONFLICT_RISK_TAU)
            except Exception:
                tau_conf = 0.25

            if (
                (coverage is not None and coverage < tau_cov)
                or (
                    result.conflict_risk is not None and result.conflict_risk > tau_conf
                )
                or temporal_mismatch
                or unit_mismatch
            ):
                # Downgrade sufficiency
                if result.is_sufficient:
                    result.is_sufficient = False
                    result.confidence = min(0.7, result.confidence)
                    result.suggested_action = (
                        "TRANSFORM_QUERY"
                        if unit_mismatch or entity_mismatch
                        else "RETRIEVE_MORE"
                    )
                    extra_reason = []
                    if coverage is not None and coverage < tau_cov:
                        extra_reason.append(
                            f"low anchor coverage {coverage:.2f} < {tau_cov:.2f}"
                        )
                    if (
                        result.conflict_risk is not None
                        and result.conflict_risk > tau_conf
                    ):
                        extra_reason.append(
                            f"high conflict risk {result.conflict_risk:.2f} > {tau_conf:.2f}"
                        )
                    if temporal_mismatch:
                        extra_reason.append("temporal mismatch")
                    if unit_mismatch:
                        extra_reason.append("unit mismatch")
                    if entity_mismatch:
                        extra_reason.append("entity mismatch")
                    result.reasoning = (
                        result.reasoning
                        + " | Heuristic check: "
                        + ", ".join(extra_reason)
                    )

            return result

        except Exception as e:
            # Fallback: assume insufficient if judge fails
            return ContextSufficiencyResult(
                is_sufficient=False,
                confidence=0.5,
                reasoning=f"Judge assessment failed: {str(e)}",
                suggested_action="RETRIEVE_MORE",
                query_transformations=[],
            )

    def _build_context_summary(self, contexts: list[dict[str, Any]]) -> str:
        """Build a concise summary of retrieved contexts for the judge."""
        if not contexts:
            return "No contexts retrieved."

        summary_parts = []
        for i, ctx in enumerate(contexts[:8]):  # Limit to first 8 contexts
            cid = ctx.get("id", f"{i+1}")
            text = ctx.get("text", "")[:200]  # First 200 chars
            summary_parts.append(f"Context {i + 1} (id={cid}): {text}...")

        return "\n".join(summary_parts)

    def _build_judge_prompt(
        self, question: str, context_summary: str, round_idx: int
    ) -> list[ChatMessage]:
        """Build the prompt for the judge assessment."""

        system_content = """You are a Judge that evaluates whether retrieved contexts contain sufficient information to answer a question accurately.

Your task:
1. Analyze if the contexts provide enough specific information to answer the question
2. Consider whether the question requires information not present in the contexts
3. Suggest query transformations if the contexts are insufficient

Respond ONLY with a JSON object in this exact format:
{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your assessment",
    "suggested_action": "STOP" | "RETRIEVE_MORE" | "TRANSFORM_QUERY",
    "query_transformations": ["alternative query 1", "alternative query 2"],
    "anchor_coverage": 0.0-1.0,  // fraction of key anchors present in contexts
    "conflict_risk": 0.0-1.0,    // 0=no conflict, 1=high conflicting facts across contexts
    "required_anchors": ["list key temporal/entities/units"],
    "mismatch_flags": {"temporal_mismatch": bool, "unit_mismatch": bool, "entity_mismatch": bool}
}

Guidelines:
- is_sufficient=true only if contexts contain specific, relevant information to answer the question
- For factual questions, require specific facts/numbers/names, not just general information
- For complex questions, ensure all parts can be answered
- confidence should reflect certainty of your assessment
- query_transformations should rephrase or break down the question if contexts are insufficient"""

        user_content = f"""Question: {question}

Retrieved Contexts:
{context_summary}

Round: {round_idx + 1}

Assess whether these contexts are sufficient to answer the question accurately."""

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

    def _parse_judge_response(
        self, response: str, question: str
    ) -> ContextSufficiencyResult:
        """Parse the judge's JSON response into a structured result."""
        try:
            # Try to parse JSON response
            data = json.loads(response.strip())

            return ContextSufficiencyResult(
                is_sufficient=bool(data.get("is_sufficient", False)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "No reasoning provided")),
                suggested_action=str(data.get("suggested_action", "RETRIEVE_MORE")),
                query_transformations=(
                    data.get("query_transformations", [])
                    if isinstance(data.get("query_transformations", []), list)
                    else []
                ),
                anchor_coverage=(
                    float(data.get("anchor_coverage"))
                    if data.get("anchor_coverage") is not None
                    else None
                ),
                conflict_risk=(
                    float(data.get("conflict_risk"))
                    if data.get("conflict_risk") is not None
                    else None
                ),
                mismatch_flags=(
                    data.get("mismatch_flags")
                    if isinstance(data.get("mismatch_flags"), dict)
                    else {}
                ),
                required_anchors=(
                    data.get("required_anchors")
                    if isinstance(data.get("required_anchors"), list)
                    else []
                ),
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback parsing for malformed responses
            response_lower = response.lower()

            # Simple heuristics for fallback parsing
            is_sufficient = any(
                word in response_lower
                for word in ["sufficient", "enough", "adequate", "complete"]
            ) and not any(
                word in response_lower
                for word in [
                    "insufficient",
                    "not enough",
                    "inadequate",
                    "incomplete",
                    "missing",
                ]
            )

            confidence = 0.3  # Low confidence for fallback parsing

            return ContextSufficiencyResult(
                is_sufficient=is_sufficient,
                confidence=confidence,
                reasoning=f"Fallback parsing of response: {response[:100]}",
                suggested_action="RETRIEVE_MORE" if not is_sufficient else "STOP",
                query_transformations=[],
                anchor_coverage=None,
                conflict_risk=None,
                mismatch_flags={},
                required_anchors=[],
            )

    def _extract_anchors(self, question: str) -> list[str]:
        """Extract simple lexical anchors (temporal/entity/unit) from the question.

        This is intentionally lightweight (regex/keyword based) to avoid
        heavyweight NLP dependencies.
        """
        ql = (question or "").lower()
        anchors: list[str] = []

        # Years
        import re as _re

        anchors.extend(_re.findall(r"\b(19\d{2}|20\d{2})\b", ql))

        # Scoreline-style triples like 50-40-90 (or 50/40/90)
        anchors.extend(_re.findall(r"\b\d{2}[-\/]\d{2}[-\/]\d{2}\b", ql))

        # Quarters
        quarter_map = {
            "q1": ["jan", "feb", "mar", "first quarter", "1st qtr"],
            "q2": ["apr", "may", "jun", "second quarter", "2nd qtr"],
            "q3": ["jul", "aug", "sep", "third quarter", "3rd qtr"],
            "q4": ["oct", "nov", "dec", "fourth quarter", "4th qtr"],
        }
        for q, toks in quarter_map.items():
            if q in ql or any(t in ql for t in toks):
                anchors.append(q)
                anchors.extend(toks)

        # Months
        months = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "sept",
            "oct",
            "nov",
            "dec",
        ]
        if any(m in ql for m in months):
            anchors.extend([m for m in months if m in ql])

        # Units / target types
        unit_terms = [
            "songs",
            "song",
            "singles",
            "tracks",
            "albums",
            "album",
            "ex-dividend",
            "dividend",
            "grand slam",
            "championship",
            "winner",
        ]
        anchors.extend([t for t in unit_terms if t in ql])

        # Key entities-like tokens (heuristic):
        # - Prefer two-word phrases (lowercase questions may include proper names)
        # - Also include title-cased single tokens from the original question
        try:
            stop = {
                "how",
                "many",
                "much",
                "what",
                "which",
                "who",
                "when",
                "where",
                "why",
                "did",
                "does",
                "do",
                "is",
                "are",
                "was",
                "were",
                "the",
                "a",
                "an",
                "in",
                "on",
                "of",
                "for",
                "to",
                "at",
                "per",
                "game",
                "club",
                "season",
                "seasons",
                "average",
                "averages",
                "made",
                "make",
            }
            toks = _re.findall(r"[a-z][a-z'\-]+", ql)
            candidates: list[str] = []
            for i in range(len(toks) - 1):
                w1, w2 = toks[i], toks[i + 1]
                if w1 in stop or w2 in stop:
                    continue
                if len(w1) >= 3 and len(w2) >= 3:
                    candidates.append(f"{w1} {w2}")
            if candidates:
                anchors.append(max(candidates, key=len))
        except Exception:
            pass

        # Also include title-cased words from the original (if any)
        words = [w.strip() for w in (question or "").split() if w.strip()]
        for w in words:
            if w[:1].isupper() and len(w) > 2 and w.lower() not in anchors:
                anchors.append(w.lower())

        # Dedup while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for a in anchors:
            if a not in seen:
                uniq.append(a)
                seen.add(a)
        return uniq[:20]

    def _compute_anchor_coverage(
        self, anchors: list[str], ctx_texts: list[str]
    ) -> tuple[float, list[str], bool]:
        """Compute fraction of anchors present in contexts and detect temporal mismatch.

        Returns (coverage, missing_anchors, temporal_mismatch)
        """
        if not anchors:
            return (1.0, [], False)
        ctx_lower = " \n".join([t.lower() for t in ctx_texts])
        present = [a for a in anchors if a.lower() in ctx_lower]
        missing = [a for a in anchors if a.lower() not in ctx_lower]

        # Temporal mismatch: if a quarter anchor appears in anchors but none of its months appear in context
        temporal_mismatch = False
        if any(
            a in anchors
            for a in [
                "q1",
                "q2",
                "q3",
                "q4",
                "1st qtr",
                "2nd qtr",
                "3rd qtr",
                "4th qtr",
                "first quarter",
                "second quarter",
                "third quarter",
                "fourth quarter",
            ]
        ):
            q_map = {
                "q1": ["jan", "feb", "mar"],
                "first quarter": ["jan", "feb", "mar"],
                "1st qtr": ["jan", "feb", "mar"],
                "q2": ["apr", "may", "jun"],
                "second quarter": ["apr", "may", "jun"],
                "2nd qtr": ["apr", "may", "jun"],
                "q3": ["jul", "aug", "sep", "sept"],
                "third quarter": ["jul", "aug", "sep", "sept"],
                "3rd qtr": ["jul", "aug", "sep", "sept"],
                "q4": ["oct", "nov", "dec"],
                "fourth quarter": ["oct", "nov", "dec"],
                "4th qtr": ["oct", "nov", "dec"],
            }
            months_present = any(m in ctx_lower for ms in q_map.values() for m in ms)
            if not months_present:
                temporal_mismatch = True

        coverage = len(present) / max(1, len(anchors))
        return (coverage, missing, temporal_mismatch)

    def _detect_unit_mismatch(
        self, question: str, contexts: list[dict[str, Any]]
    ) -> bool:
        ql = (question or "").lower()
        ctx = (" \n".join([c.get("text", "") for c in contexts])).lower()
        if "songs" in ql or "song" in ql or "singles" in ql or "tracks" in ql:
            if (
                ("song" not in ctx)
                and ("songs" not in ctx)
                and ("single" not in ctx)
                and ("singles" not in ctx)
                and ("track" not in ctx)
                and ("tracks" not in ctx)
            ) and ("album" in ctx or "albums" in ctx):
                return True
        return False

    def _detect_entity_mismatch(
        self, question: str, contexts: list[dict[str, Any]]
    ) -> bool:
        ql = (question or "").lower()
        ctx = (" \n".join([c.get("text", "") for c in contexts])).lower()
        # Heuristic: if question mentions a specific entity (simple capitalized word), ensure it's in context
        ents = [w for w in (question or "").split() if w[:1].isupper() and len(w) > 2]
        ents_l = [e.lower() for e in ents]
        if ents_l and not any(e in ctx for e in ents_l):
            return True
        # For tennis "grand slam" questions, require tournament anchor
        if "grand slam" in ql and not any(
            t in ctx
            for t in [
                "australian open",
                "french open",
                "wimbledon",
                "us open",
                "u.s. open",
            ]
        ):
            return True
        return False

    def _estimate_conflict_risk(self, contexts: list[dict[str, Any]]) -> float:
        """Crude conflict risk estimator based on divergent numbers/dates across contexts."""
        import re as _re

        texts = [c.get("text", "") for c in contexts[:6]]
        joined = " \n".join(texts).lower()
        # Extract years and months
        years = _re.findall(r"\b(19\d{2}|20\d{2})\b", joined)
        # If multiple distinct years appear, slight conflict risk
        year_risk = 0.0
        if len(set(years)) >= 3:
            year_risk = 0.3
        # Extract prominent numbers (1-31 for days)
        days = _re.findall(r"\b([1-2]?\d|3[0-1])\b", joined)
        day_risk = 0.0
        if len(set(days)) >= 5:
            day_risk = 0.2
        return min(1.0, year_risk + day_risk)


class QueryTransformer:
    """
    Query transformation module for improving retrieval through query rewriting
    and decomposition.
    """

    def __init__(self, llm_client: OpenAIAdapter):
        self.llm = llm_client

    def transform_query(
        self,
        original_query: str,
        context_assessment: ContextSufficiencyResult,
        failed_contexts: list[dict[str, Any]],
    ) -> list[str]:
        """
        Transform the original query to improve retrieval.

        Args:
            original_query: The original user question
            context_assessment: Judge's assessment of current contexts
            failed_contexts: Contexts that were deemed insufficient

        Returns:
            List of transformed queries to try
        """
        # If judge already provided transformations, use those
        if context_assessment.query_transformations:
            return context_assessment.query_transformations[:3]  # Limit to 3

        # Generate transformations using LLM
        return self._generate_query_transformations(original_query, failed_contexts)

    def _generate_query_transformations(
        self, query: str, failed_contexts: list[dict[str, Any]]
    ) -> list[str]:
        """Generate query transformations using LLM."""

        context_summary = "\n".join(
            [f"- {ctx.get('text', '')[:100]}..." for ctx in failed_contexts[:3]]
        )

        system_content = """You are a Query Transformer that rewrites questions to improve information retrieval.

Your task: Given a question and contexts that were insufficient, generate 2-3 alternative queries that might retrieve better information.

Strategies:
1. Rephrase using synonyms or alternative terms
2. Break complex questions into simpler parts
3. Add specific context or constraints
4. Focus on key entities or concepts

Respond with a JSON array of 2-3 alternative queries:
["alternative query 1", "alternative query 2", "alternative query 3"]"""

        user_content = f"""Original Question: {query}

Insufficient Contexts Found:
{context_summary}

Generate 2-3 alternative queries that might retrieve better information to answer the original question."""

        try:
            prompt = [
                ChatMessage(role="system", content=system_content),
                ChatMessage(role="user", content=user_content),
            ]

            response, usage = self.llm.chat(
                messages=prompt,
                max_tokens=150,
                temperature=0.3,  # Slight creativity for transformations
            )

            # Parse JSON response
            transformations = json.loads(response.strip())
            if isinstance(transformations, list):
                return transformations[:3]  # Limit to 3
            else:
                return [str(transformations)]

        except Exception:
            # Fallback: simple entity-based transformation
            return self._fallback_transformations(query)

    def _fallback_transformations(self, query: str) -> list[str]:
        """Generate simple fallback transformations without LLM."""
        transformations = []

        # Simple rephrasings
        if "who is" in query.lower():
            transformations.append(query.replace("who is", "information about"))
        elif "what is" in query.lower():
            transformations.append(query.replace("what is", "details about"))
        elif "where" in query.lower():
            transformations.append(query.replace("where", "location of"))
        elif "when" in query.lower():
            transformations.append(query.replace("when", "time of"))

        # Add one generic transformation
        transformations.append(f"background information {query}")

        return transformations[:2]  # Return up to 2 fallback transformations


def create_judge(llm_client: OpenAIAdapter, settings: Settings) -> Judge:
    """Factory function to create a Judge instance."""
    return Judge(llm_client, settings)


def create_query_transformer(llm_client: OpenAIAdapter) -> QueryTransformer:
    """Factory function to create a QueryTransformer instance."""
    return QueryTransformer(llm_client)


# === Anchor helpers (exported) ===


def extract_required_anchors(question: str) -> set[str]:
    """Extract required anchors (years, time windows, units, events/categories) from question."""
    q = (question or "").lower()
    anchors: set[str] = set()
    # Years
    import re as _re

    anchors.update(_re.findall(r"\b(?:19|20)\d{2}\b", q))
    # Scoreline-style triples like 50-40-90 (and 50/40/90)
    anchors.update(_re.findall(r"\b\d{2}[-\/]\d{2}[-\/]\d{2}\b", q))
    # Quarters / windows
    for tok in [
        "q1",
        "q2",
        "q3",
        "q4",
        "first week",
        "last month",
        "per game",
        "ex-dividend",
        "domestic",
        "worldwide",
    ]:
        if tok in q:
            anchors.add(tok)
    # Months
    months = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
    for m in months:
        if m in q:
            anchors.add(m)
    # Events / categories
    for tok in [
        "u.s. open",
        "us open",
        "australian open",
        "wimbledon",
        "french open",
        "grand slam",
        "best animated feature",
        "visual effects",
    ]:
        if tok in q:
            anchors.add(tok)
    # Heuristic two-word entity (lowercase questions may include proper names)
    # UPDATED: Use capitalized words from original question as a better heuristic
    try:
        caps = _re.findall(
            r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,3})\b",
            question or "",
        )
        if caps:
            # Add longest capitalized span as a candidate anchor
            anchors.add(max(caps, key=len).lower())
    except Exception:
        pass
    return anchors


def anchors_present_in_texts(
    anchors: set[str], texts: list[str]
) -> tuple[set[str], float]:
    """Return (present_anchors, coverage) where coverage=present/len(anchors)."""
    if not anchors:
        return set(), 1.0
    joined = (" \n".join(texts or [])).lower()
    present = {a for a in anchors if a.lower() in joined}
    cov = len(present) / max(1, len(anchors))
    return present, cov


def validate_factoid_anchors(question: str, texts: list[str]) -> dict[str, bool]:
    """Cheap validators for factoids: time/unit/event presence in cited text.

    Returns flags: {fail_time, fail_unit, fail_event}
    """
    q = (question or "").lower()
    joined = (" \n".join(texts or [])).lower()
    import re as _re

    # Time cues in question
    q_years = _re.findall(r"\b(?:19|20)\d{2}\b", q)
    q_has_qtr = any(
        t in q for t in ["q1", "q2", "q3", "q4", "first week", "last month"]
    ) or any(
        m in q
        for m in [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "sept",
            "oct",
            "nov",
            "dec",
        ]
    )
    t_has_year = any(y in joined for y in q_years)
    t_has_qtr = any(
        t in joined for t in ["q1", "q2", "q3", "q4", "first week", "last month"]
    ) or any(
        m in joined
        for m in [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "sept",
            "oct",
            "nov",
            "dec",
        ]
    )
    fail_time = (bool(q_years) and not t_has_year) or (q_has_qtr and not t_has_qtr)

    # Unit cues: per game / ex-dividend / domestic/worldwide
    q_unit_terms = ["per game", "ex-dividend", "domestic", "worldwide"]
    fail_unit = any(term in q and term not in joined for term in q_unit_terms)

    # Event/category cues: tournaments / award categories
    q_event_terms = [
        "u.s. open",
        "us open",
        "australian open",
        "wimbledon",
        "french open",
        "best animated feature",
        "visual effects",
    ]
    # If any event term appears in question, require it in text
    fail_event = any(term in q and term not in joined for term in q_event_terms)

    return {"fail_time": fail_time, "fail_unit": fail_unit, "fail_event": fail_event}
