"""Adapter to call external BAUG with the expected signals."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.config import settings


@dataclass
class Signals:
    overlap_est: float
    faith_est: float
    new_hits_ratio: float
    anchor_coverage: float
    budget_left: int
    intent_confidence: float
    slot_completeness: float
    source_of_intent: str
    validators_passed: bool = True
    conflict_risk: float = 0.0
    round_idx: int = 0
    has_reflect_left: bool = True
    lexical_uncertainty: float = 0.0
    completeness: float = 1.0
    semantic_coherence: float = 1.0
    answer_length: int = 0
    question_complexity: float = 0.5
    extras: dict[str, Any] = field(default_factory=dict)
    fine_median: float = 0.0
    prev_overlap: float = 0.0  # New field
    prev_faith: float = 0.0  # New field
    tokens_last_round: int = 0  # New field
    last_action: str = "START"  # New field, to track previous gate decision

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signals:
        extras = data.get("extras") or {}
        if not isinstance(extras, dict):
            extras = {}
        return cls(
            overlap_est=float(data.get("overlap_est", 0.0) or 0.0),
            faith_est=float(data.get("faith_est", 0.0) or 0.0),
            new_hits_ratio=float(data.get("new_hits_ratio", 0.0) or 0.0),
            anchor_coverage=float(data.get("anchor_coverage", 0.0) or 0.0),
            budget_left=int(data.get("budget_left", 0) or 0),
            intent_confidence=float(data.get("intent_confidence", 0.0) or 0.0),
            slot_completeness=float(data.get("slot_completeness", 0.0) or 0.0),
            source_of_intent=str(data.get("source_of_intent", "")),
            validators_passed=bool(data.get("validators_passed", True)),
            conflict_risk=float(data.get("conflict_risk", 0.0) or 0.0),
            round_idx=int(data.get("round_idx", 0) or 0),
            has_reflect_left=bool(data.get("has_reflect_left", True)),
            lexical_uncertainty=float(data.get("lexical_uncertainty", 0.0) or 0.0),
            completeness=float(data.get("completeness", 1.0) or 1.0),
            semantic_coherence=float(data.get("semantic_coherence", 1.0) or 1.0),
            answer_length=int(data.get("answer_length", 0) or 0),
            question_complexity=float(data.get("question_complexity", 0.5) or 0.5),
            extras=extras,
            fine_median=float(data.get("fine_median", 0.0) or 0.0),
            prev_overlap=float(data.get("prev_overlap", 0.0) or 0.0),
            prev_faith=float(data.get("prev_faith", 0.0) or 0.0),
            tokens_last_round=int(data.get("tokens_last_round", 0) or 0),
            last_action=str(data.get("last_action", "START")),
        )


class BAUGAdapter:
    def __init__(self) -> None:
        self._external = os.getenv("BAUG_HANDLER")
        self._callable: Any | None = None
        self._last_decision: str | None = None
        self._last_signals: Signals | None = None
        self._last_reasons: list[str] = []
        if self._external:
            try:
                mod_name, fn_name = self._external.split(":", 1)
                mod = importlib.import_module(mod_name)
                self._callable = getattr(mod, fn_name)
                print(f"BAUG adapter loaded external handler: {self._external}")
            except Exception as exc:
                print(f"Failed to load BAUG_HANDLER '{self._external}': {exc}")
                self._callable = None

    def decide(self, raw_signals: dict[str, Any]) -> str:
        if self._callable is not None:
            try:
                self._last_decision = str(self._callable(raw_signals))
                self._last_reasons = ["external_handler"]
                return self._last_decision
            except Exception as exc:
                print(f"External BAUG call failed, using fallback: {exc}")

        sig = Signals.from_dict(raw_signals)
        self._last_signals = sig
        action, reasons = self._rule_based(sig)
        self._last_decision = action
        self._last_reasons = reasons
        return action

    def should_retrieve(self, raw_signals: dict[str, Any]) -> str:
        """
        Determines if the system should proceed with another retrieval round.
        Implements marginal-gain logic: stop if gain per token is too low
        and anchor coverage is sufficient.
        """
        sig = Signals.from_dict(raw_signals)
        reasons: list[str] = []

        if not settings.BAUG_ANCHOR_GATE_ON:
            return "GO"  # BAUG is off, always retrieve

        # Marginal-gain rule
        # Only apply marginal gain after the first round (round_idx > 0)
        if sig.round_idx > 0:
            gain = (sig.overlap_est + sig.faith_est) - (
                sig.prev_overlap + sig.prev_faith
            )
            # Avoid division by zero, consider gain per 100 tokens
            tokens_for_gain_calc = max(100, sig.tokens_last_round)
            g100 = (gain / tokens_for_gain_calc) * 100
            # If coverage is already good and marginal gain is low, skip retrieval
            if (
                sig.anchor_coverage >= settings.TAU_COVERAGE
                and g100 < settings.LAMBDA_MIN_GAIN
            ):
                reasons.append(f"marginal_gain_low:{g100:.2f}")
                reasons.append(f"coverage_sufficient:{sig.anchor_coverage:.2f}")
                return "SKIP"

        # If anchor coverage is still low, allow retrieval to try and improve it
        if sig.anchor_coverage < settings.TAU_COVERAGE:
            reasons.append(f"low_anchor_coverage:{sig.anchor_coverage:.2f}")
            return "GO"

        # If new hits are very low, skip retrieval (unless coverage is also low, which is handled above)
        if sig.new_hits_ratio < settings.EPS_NEW_HITS:
            reasons.append(f"new_hits_low:{sig.new_hits_ratio:.2f}")
            return "SKIP"

        return "GO"

    def should_generate(self, raw_signals: dict[str, Any]) -> str:
        """
        Determines if the system should proceed with generating an answer.
        """
        sig = Signals.from_dict(raw_signals)
        reasons: list[str] = []

        if not settings.BAUG_ANCHOR_GATE_ON:
            return "GO"  # BAUG is off, always generate

        # On Round 1, always allow at least one generation attempt (we need to extract anchors first)
        if sig.round_idx == 1:
            return "GO"

        # If anchor coverage is still low after Round 1, skip generation and abstain
        if sig.anchor_coverage < settings.TAU_COVERAGE:
            reasons.append(
                f"low_anchor_coverage_post_retrieval:{sig.anchor_coverage:.2f}"
            )
            return "SKIP"

        # Marginal-gain rule for generation: If gain from previous round was low and quality is already good, skip
        if sig.round_idx > 0:
            gain = (sig.overlap_est + sig.faith_est) - (
                sig.prev_overlap + sig.prev_faith
            )
            tokens_for_gain_calc = max(100, sig.tokens_last_round)
            g100 = (gain / tokens_for_gain_calc) * 100

            # If marginal gain is low AND current overlap/faithfulness are already good, skip generation
            if g100 < settings.LAMBDA_MIN_GAIN and (
                sig.overlap_est >= settings.TAU_OVERLAP
                or sig.faith_est >= settings.TAU_FAITH
            ):
                reasons.append(f"marginal_gain_low_pre_generate:{g100:.2f}")
                reasons.append(
                    f"quality_already_sufficient:(overlap={sig.overlap_est:.2f},faith={sig.faith_est:.2f})"
                )
                return "SKIP"

        return "GO"

    def _rule_based(self, sig: Signals) -> tuple[str, list[str]]:
        reasons: list[str] = []

        # 1. Stop if quality is already high
        if (
            sig.overlap_est >= settings.TAU_OVERLAP
            and sig.faith_est >= settings.TAU_FAITH
            and sig.anchor_coverage >= settings.TAU_COVERAGE
        ):
            reasons.append("quality_thresholds_met")
            return "STOP", reasons

        # 2. Stop if marginal gain is too low (and we are not in the first round)
        if sig.round_idx > 0:
            gain = (sig.overlap_est + sig.faith_est) - (
                sig.prev_overlap + sig.prev_faith
            )
            tokens_for_gain_calc = max(100, sig.tokens_last_round)
            g100 = (gain / tokens_for_gain_calc) * 100

            if g100 < settings.LAMBDA_MIN_GAIN:
                reasons.append(f"marginal_gain_too_low:{g100:.2f}")
                return "STOP", reasons

        # 3. Check for temporal conflicts
        validators = sig.extras.get("validators", {})
        temporal_val = validators.get("temporal", {})
        temporal_risk = temporal_val.get("temporal_risk", 0.0)

        # If high temporal risk and overlap/faith not exceptional, be conservative
        if temporal_risk > 0.6:
            if sig.overlap_est < 0.7 or sig.faith_est < 0.8:
                reasons.append(f"temporal_conflict_risk:{temporal_risk:.2f}")
                return "ABSTAIN", reasons

        # 4. If not stopping, decide whether to retrieve more or abstain
        # Abstain if anchor coverage is very low and unlikely to improve
        if sig.anchor_coverage < 0.2 and sig.new_hits_ratio < 0.1:
            reasons.append("low_coverage_and_stagnant_retrieval")
            return "ABSTAIN", reasons

        # 5. Otherwise, retrieve more if we have budget and rounds left
        if (
            sig.round_idx < settings.MAX_ROUNDS - 1
        ):  # -1 because round_idx is 0-indexed in signals
            reasons.append("attempting_to_improve_quality")
            return "RETRIEVE_MORE", reasons

        # 6. Default to STOP if max rounds are reached
        reasons.append("max_rounds_reached")
        return "STOP", reasons

    def kind(self) -> str:
        return "external" if self._callable is not None else "built-in"

    def last_decision(self) -> str | None:
        return self._last_decision

    def last_source_of_intent(self) -> str | None:
        if self._last_signals is None:
            return None
        return self._last_signals.source_of_intent or None

    def last_reasons(self) -> list[str]:
        return list(self._last_reasons)
