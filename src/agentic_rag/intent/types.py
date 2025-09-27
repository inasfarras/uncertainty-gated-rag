from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Intent:
    task_type: str = "factoid"  # "factoid"|"list"|"compare"|"definition"|"why"
    core_entities: list[str] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    canonical_query: str = ""
    ambiguity_flags: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0  # 0..1
    slot_completeness: float = 0.0  # 0..1
    source_of_intent: str = "rule_only"  # "rule_only"|"llm_fallback"|"llm_only"
