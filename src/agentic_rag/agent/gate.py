"""Uncertainty gate for determining when to continue RAG iterations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agentic_rag.config import Settings


class GateAction:
    STOP = "STOP"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    REFLECT = "REFLECT"
    ABSTAIN = "ABSTAIN"


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
    extras: Optional[Dict[str, Any]] = None


class BaseGate(ABC):
    @abstractmethod
    def decide(self, signals: GateSignals) -> str: ...


class UncertaintyGate(BaseGate):
    def __init__(self, settings: Settings):
        self.tau_f = settings.FAITHFULNESS_TAU
        self.tau_o = settings.OVERLAP_TAU
        self.tau_uncertain = settings.UNCERTAINTY_TAU
        # Weights for uncertainty components
        self.w_f = 0.4
        self.w_o = 0.4
        self.w_lex = 0.1
        self.w_comp = 0.1

    def decide(self, signals: GateSignals) -> str:
        # High-confidence stop condition
        if signals.faith >= self.tau_f and signals.overlap >= self.tau_o:
            return GateAction.STOP

        # Weighted uncertainty score
        uncertainty = (
            self.w_f * (1 - signals.faith)
            + self.w_o * (1 - signals.overlap)
            + self.w_lex * signals.lexical_uncertainty
            + self.w_comp * (1 - signals.completeness)
        )

        # Novelty penalty
        if signals.round_idx > 0 and signals.novelty_ratio < 0.2:
            uncertainty += 0.1

        # Store uncertainty for logging
        if signals.extras is not None:
            signals.extras["uncertainty_score"] = uncertainty

        if signals.has_reflect_left and uncertainty >= self.tau_uncertain:
            return GateAction.REFLECT

        return GateAction.RETRIEVE_MORE


def make_gate(settings: Settings) -> UncertaintyGate:
    """Create and return an UncertaintyGate instance."""
    return UncertaintyGate(settings)
