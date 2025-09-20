"""Enhanced uncertainty gate for determining when to continue RAG iterations."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentic_rag.config import Settings


class GateAction:
    STOP = "STOP"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    REFLECT = "REFLECT"
    ABSTAIN = "ABSTAIN"
    STOP_LOW_BUDGET = "STOP_LOW_BUDGET"  # New: explicit budget stop


@dataclass
class GateSignals:
    faith: float
    overlap: float
    lexical_uncertainty: float
    completeness: float
    budget_left_tokens: int
    round_idx: int
    has_reflect_left: bool
    novelty_ratio: float
    semantic_coherence: float = 1.0  # New: semantic coherence score
    answer_length: int = 0  # New: response length for context
    question_complexity: float = 0.5  # New: question complexity indicator
    extras: Optional[Dict[str, Any]] = None


class BaseGate(ABC):
    @abstractmethod
    def decide(self, signals: GateSignals) -> str: ...


class UncertaintyGate(BaseGate):
    def __init__(self, settings: Settings):
        self.tau_f = settings.FAITHFULNESS_TAU
        self.tau_o = settings.OVERLAP_TAU
        self.tau_uncertain = settings.UNCERTAINTY_TAU
        self.low_budget_tokens = settings.LOW_BUDGET_TOKENS
        self.max_tokens_total = settings.MAX_TOKENS_TOTAL
        self.strict_stop_requires_judge = getattr(
            settings, "STRICT_STOP_REQUIRES_JUDGE_OK", True
        )
        self.judge_min_conf_for_stop = getattr(settings, "JUDGE_MIN_CONF_FOR_STOP", 0.8)
        self.anchor_cov_tau = getattr(settings, "ANCHOR_COVERAGE_TAU", 0.6)
        self.conflict_risk_tau = getattr(settings, "CONFLICT_RISK_TAU", 0.25)

        # Enhanced adaptive weights
        self.base_weights = {
            "faith": 0.35,
            "overlap": 0.35,
            "lexical": 0.10,
            "completeness": 0.10,
            "semantic": 0.10,
        }

        # Cached patterns for efficiency
        self._uncertainty_pattern = re.compile(
            r"\b(might|maybe|perhaps|possibly|likely|probably|seems|appears|"
            r"suggests|indicates|unclear|uncertain|not sure|don\'t know|"
            r"can\'t say|difficult to determine|unsure|ambiguous)\b",
            re.IGNORECASE,
        )
        self._confidence_pattern = re.compile(
            r"\b(definitely|certainly|clearly|obviously|undoubtedly|"
            r"absolutely|precisely|exactly|specifically|confirmed)\b",
            re.IGNORECASE,
        )

        # Performance cache and settings
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._enable_caching = getattr(settings, "ENABLE_GATE_CACHING", True)

    def decide(self, signals: GateSignals) -> str:
        # Use caching if enabled
        if self._enable_caching:
            cache_key = self._create_cache_key(signals)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                if signals.extras is not None:
                    signals.extras.update(cached_result["extras"])
                return cached_result["decision"]

            # Compute decision and cache it
            decision = self._decide_uncached(signals)
            self._store_in_cache(cache_key, decision, signals.extras or {})
            return decision
        else:
            return self._decide_uncached(signals)

    def _decide_uncached(self, signals: GateSignals) -> str:
        # Early high-confidence stop condition (optimized) with Judge override
        if signals.faith >= self.tau_f and signals.overlap >= self.tau_o:
            judge_ok = True
            if signals.extras is not None:
                js = signals.extras.get("judge_sufficient")
                jc = float(signals.extras.get("judge_confidence", 0.0) or 0.0)
                anchor_cov = signals.extras.get("anchor_coverage")
                conflict_risk = signals.extras.get("conflict_risk")
                mismatch_flags = signals.extras.get("mismatch_flags") or {}

                # Default: if strict mode enabled, require judge agreement and adequate anchors
                if self.strict_stop_requires_judge:
                    # If judge explicitly says insufficient with decent confidence, do not stop
                    if js is False and jc >= 0.6:
                        judge_ok = False
                    # If judge did not run or is unsure, still enforce anchors/conflict checks if available
                    if (
                        anchor_cov is not None
                        and float(anchor_cov) < self.anchor_cov_tau
                    ):
                        judge_ok = False
                    if (
                        conflict_risk is not None
                        and float(conflict_risk) > self.conflict_risk_tau
                    ):
                        judge_ok = False
                    if mismatch_flags and (
                        mismatch_flags.get("temporal_mismatch")
                        or mismatch_flags.get("unit_mismatch")
                        or mismatch_flags.get("entity_mismatch")
                    ):
                        judge_ok = False
                    # Even if judge says sufficient, require minimal confidence
                    if js is True and jc < self.judge_min_conf_for_stop:
                        judge_ok = False

                if judge_ok:
                    signals.extras["uncertainty_score"] = 0.0
                    signals.extras["stop_reason"] = "high_confidence"
                    return GateAction.STOP
                else:
                    # Fall-through to uncertainty calculation (i.e., don't early stop)
                    pass
            else:
                # No extras: allow early stop
                return GateAction.STOP

        # Fast budget check using configured threshold
        if signals.budget_left_tokens < self.low_budget_tokens:
            if signals.extras is not None:
                signals.extras["uncertainty_score"] = 1.0
                signals.extras["stop_reason"] = "budget_exhausted"
            return GateAction.STOP_LOW_BUDGET

        # Enhanced uncertainty calculation with adaptive weights
        weights = self._get_adaptive_weights(signals)
        uncertainty = self._calculate_enhanced_uncertainty(signals, weights)

        # Enhanced novelty and stagnation penalties
        uncertainty = self._apply_context_penalties(uncertainty, signals)

        # Store detailed metrics for logging
        if signals.extras is not None:
            signals.extras["uncertainty_score"] = uncertainty
            signals.extras["adaptive_weights"] = weights
            signals.extras["cache_hit_rate"] = self._cache_hits / max(
                1, self._cache_hits + self._cache_misses
            )

        # Decision logic with enhanced thresholds
        if uncertainty >= self.tau_uncertain * 1.2:  # High uncertainty
            if signals.has_reflect_left and signals.round_idx > 0:
                return GateAction.REFLECT
            else:
                return GateAction.RETRIEVE_MORE
        elif uncertainty >= self.tau_uncertain:  # Medium uncertainty
            if signals.has_reflect_left and signals.semantic_coherence < 0.7:
                return GateAction.REFLECT
            return GateAction.RETRIEVE_MORE
        else:  # Low uncertainty
            return GateAction.STOP

    def _get_adaptive_weights(self, signals: GateSignals) -> Dict[str, float]:
        """Adapt weights based on question complexity and context."""
        weights = self.base_weights.copy()

        # Adjust weights based on question complexity
        if signals.question_complexity > 0.7:  # Complex questions
            weights["faith"] *= 1.2
            weights["semantic"] *= 1.3
        elif signals.question_complexity < 0.3:  # Simple questions
            weights["overlap"] *= 1.2
            weights["completeness"] *= 1.1

        # Adjust based on round - early rounds focus more on retrieval quality
        if signals.round_idx == 0:
            weights["overlap"] *= 1.1
            weights["faith"] *= 0.9

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_enhanced_uncertainty(
        self, signals: GateSignals, weights: Dict[str, float]
    ) -> float:
        """Calculate enhanced uncertainty score with semantic analysis and Judge signals."""
        # Core uncertainty components
        uncertainty = (
            weights["faith"] * (1 - signals.faith)
            + weights["overlap"] * (1 - signals.overlap)
            + weights["lexical"] * signals.lexical_uncertainty
            + weights["completeness"] * (1 - signals.completeness)
            + weights["semantic"] * (1 - signals.semantic_coherence)
        )

        # Integrate Judge signals if available
        if signals.extras:
            judge_sufficient = signals.extras.get("judge_sufficient")
            judge_confidence = signals.extras.get("judge_confidence", 0.5)
            anchor_cov = signals.extras.get("anchor_coverage")
            conflict_risk = signals.extras.get("conflict_risk")
            mismatch_flags = signals.extras.get("mismatch_flags") or {}

            if judge_sufficient is not None:
                # Judge signal: if insufficient with high confidence, increase uncertainty
                if not judge_sufficient and judge_confidence > 0.7:
                    uncertainty += (
                        0.2 * judge_confidence
                    )  # Strong signal for more retrieval
                elif judge_sufficient and judge_confidence > 0.8:
                    uncertainty *= 0.7  # Strong signal for stopping
                elif judge_sufficient and judge_confidence > 0.6:
                    uncertainty *= 0.85  # Moderate signal for stopping

            # Penalize low anchor coverage / high conflict / mismatches
            try:
                if anchor_cov is not None and float(anchor_cov) < self.anchor_cov_tau:
                    uncertainty = min(1.0, uncertainty + 0.15)
            except Exception:
                pass
            try:
                if (
                    conflict_risk is not None
                    and float(conflict_risk) > self.conflict_risk_tau
                ):
                    uncertainty = min(1.0, uncertainty + 0.1)
            except Exception:
                pass
            if mismatch_flags and (
                mismatch_flags.get("temporal_mismatch")
                or mismatch_flags.get("unit_mismatch")
                or mismatch_flags.get("entity_mismatch")
            ):
                uncertainty = min(1.0, uncertainty + 0.15)

        return min(1.0, uncertainty)

    def _apply_context_penalties(
        self, uncertainty: float, signals: GateSignals
    ) -> float:
        """Apply context-aware penalties and bonuses."""
        # Novelty penalty (enhanced)
        if signals.round_idx > 0:
            if signals.novelty_ratio < 0.1:  # Very low novelty
                uncertainty += 0.15
            elif signals.novelty_ratio < 0.2:  # Low novelty
                uncertainty += 0.08

        # Length-based adjustment
        if signals.answer_length < 20:  # Very short answers
            uncertainty += 0.1
        elif (
            signals.answer_length > 200
        ):  # Very long answers might be verbose/uncertain
            uncertainty += 0.05

        # Budget pressure using configured max tokens
        budget_ratio = signals.budget_left_tokens / self.max_tokens_total
        if budget_ratio < 0.2:  # Low budget remaining
            uncertainty *= 1.2  # Increase uncertainty to encourage stopping

        return min(1.0, uncertainty)

    def _create_cache_key(self, signals: GateSignals) -> str:
        """Create a cache key from gate signals."""
        # Create a hash-like key from key signal values (rounded for cache efficiency)
        key_components = [
            f"f:{signals.faith:.2f}",
            f"o:{signals.overlap:.2f}",
            f"lex:{signals.lexical_uncertainty:.2f}",
            f"comp:{signals.completeness:.2f}",
            f"sem:{signals.semantic_coherence:.2f}",
            f"r:{signals.round_idx}",
            f"nov:{signals.novelty_ratio:.2f}",
            f"budget:{signals.budget_left_tokens // 100}00",  # Round to nearest 100
        ]
        return "|".join(key_components)

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached decision if available."""
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        return None

    def _store_in_cache(
        self, cache_key: str, decision: str, extras: Dict[str, Any]
    ) -> None:
        """Store decision in cache."""
        # Implement simple LRU by removing oldest entries when cache gets too large
        if len(self._cache) >= 100:  # Max cache size
            # Remove oldest entry (first in dict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = {"decision": decision, "extras": extras.copy()}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, total_requests),
            "cache_size": len(self._cache),
        }


def make_gate(settings: Settings) -> UncertaintyGate:
    """Create and return an enhanced UncertaintyGate instance."""
    return UncertaintyGate(settings)
