"""Adapter to call external BAUG with the expected signals."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.agent.gate import GateAction
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

    def _rule_based(self, sig: Signals) -> tuple[str, list[str]]:
        reasons: list[str] = []
        slot_thresh = getattr(settings, "BAUG_SLOT_COMPLETENESS_MIN", 0.6)
        coverage_min = getattr(settings, "BAUG_STOP_COVERAGE_MIN", 0.3)
        high_overlap_tau = getattr(settings, "BAUG_HIGH_OVERLAP_TAU", 0.7)

        if (
            sig.slot_completeness < slot_thresh
            and sig.budget_left < settings.FACTOID_MIN_TOKENS_LEFT
        ):
            reasons.extend(["low_slot_completeness", "low_budget"])
            return GateAction.ABSTAIN, reasons

        if not sig.validators_passed:
            reasons.append("validators_missing")
            if sig.budget_left < settings.FACTOID_MIN_TOKENS_LEFT:
                reasons.append("low_budget")
                return GateAction.ABSTAIN, reasons
            return GateAction.RETRIEVE_MORE, reasons

        coverage_ok = sig.anchor_coverage >= coverage_min
        overlap_ok = sig.overlap_est >= settings.OVERLAP_TAU
        high_overlap = sig.overlap_est >= high_overlap_tau

        if overlap_ok and coverage_ok:
            reasons.extend(["overlap_ok", "coverage_ok"])
            return GateAction.STOP, reasons
        if high_overlap:
            reasons.append("high_overlap")
            if not coverage_ok:
                reasons.append("low_coverage")
            return GateAction.STOP, reasons

        if sig.round_idx > 0 and sig.new_hits_ratio < settings.NEW_HITS_EPS:
            reasons.append("low_new_hits")
            return "STOP_LOW_GAIN", reasons

        if sig.budget_left < settings.FACTOID_MIN_TOKENS_LEFT:
            reasons.append("low_budget")

        if not coverage_ok:
            reasons.append("low_coverage")
        if sig.overlap_est < settings.OVERLAP_TAU:
            reasons.append("low_overlap")
        if sig.new_hits_ratio < settings.NEW_HITS_EPS:
            reasons.append("low_new_hits")
        if not reasons:
            reasons.append("default")
        return GateAction.RETRIEVE_MORE, reasons

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
