from __future__ import annotations

"""Adapter to call external BAUG with the expected signals.

If no external BAUG is provided, falls back to the in-repo UncertaintyGate
to keep the pipeline runnable. This keeps BAUG as the final authority
for STOP / RETRIEVE_MORE / REFLECT / ABSTAIN.
"""

import importlib
import os
from typing import Any, Dict

from agentic_rag.agent.gate import GateSignals, UncertaintyGate
from agentic_rag.config import settings


class BAUGAdapter:
    def __init__(self) -> None:
        self._external = os.getenv("BAUG_HANDLER")  # dotted path module:function
        self._fallback_gate = UncertaintyGate(settings)
        self._callable = None
        self._last_decision: str | None = None
        if self._external:
            try:
                mod_name, fn_name = self._external.split(":", 1)
                mod = importlib.import_module(mod_name)
                self._callable = getattr(mod, fn_name)
                print(f"🔌 BAUG adapter loaded external handler: {self._external}")
            except Exception as e:
                print(f"⚠️  Failed to load BAUG_HANDLER '{self._external}': {e}")
                self._callable = None

    def decide(self, signals: Dict[str, Any]) -> str:
        """Return BAUG action from signals dict.

        Signals expected keys: overlap_est, faith_est, new_hits_ratio, anchor_coverage,
        conflict_risk, budget_left, round_idx, has_reflect_left, lexical_uncertainty,
        completeness, semantic_coherence, extras (dict with judge info if any).
        """
        if self._callable is not None:
            try:
                self._last_decision = str(self._callable(signals))
                return self._last_decision
            except Exception as e:
                print(f"⚠️  External BAUG call failed, using fallback: {e}")

        # Fallback: map to UncertaintyGate signals
        gs = GateSignals(
            faith=float(signals.get("faith_est", 0.0) or 0.0),
            overlap=float(signals.get("overlap_est", 0.0) or 0.0),
            lexical_uncertainty=float(signals.get("lexical_uncertainty", 0.0) or 0.0),
            completeness=float(signals.get("completeness", 0.0) or 0.0),
            budget_left_tokens=int(signals.get("budget_left", 0) or 0),
            round_idx=int(signals.get("round_idx", 0) or 0),
            has_reflect_left=bool(signals.get("has_reflect_left", True)),
            novelty_ratio=float(signals.get("new_hits_ratio", 0.0) or 0.0),
            semantic_coherence=float(signals.get("semantic_coherence", 1.0) or 1.0),
            answer_length=int(signals.get("answer_length", 0) or 0),
            question_complexity=float(signals.get("question_complexity", 0.5) or 0.5),
            extras=signals.get("extras", {}),
        )
        self._last_decision = self._fallback_gate.decide(gs)
        return self._last_decision

    def kind(self) -> str:
        """Return the gate kind in use: 'external' or 'UncertaintyGate'."""
        return "external" if self._callable is not None else "UncertaintyGate"

    def last_decision(self) -> str | None:
        return self._last_decision
