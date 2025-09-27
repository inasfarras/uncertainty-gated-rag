from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_ORG: str | None = None

    LLM_MODEL: str = "gpt-4o-mini"
    # Deterministic generation controls
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.0
    EMBED_BACKEND: Literal["openai", "st", "mock"] = "openai"
    EMBED_MODEL: str = "text-embedding-3-small"
    ST_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    FAISS_INDEX_PATH: str = "artifacts/crag_faiss"
    # Overall budget guardrail for a single query
    MAX_TOKENS_TOTAL: int = 3500
    # Max tokens for packed context in prompts (new name; keep old for compat)
    MAX_CONTEXT_TOKENS: int = 1000
    # Backwards-compat name (do not remove); prefer MAX_CONTEXT_TOKENS
    CONTEXT_TOKEN_CAP: int = 2000
    # Max tokens the model can generate
    MAX_OUTPUT_TOKENS: int = 160
    # Loop controls
    MAX_ROUNDS: int = 1
    RETRIEVAL_K: int = 8
    PROBE_FACTOR: int = 4
    GRAPH_K: int = 20
    FAITHFULNESS_TAU: float = 0.65
    OVERLAP_TAU: float = 0.42
    OVERLAP_SIM_TAU: float = 0.60
    UNCERTAINTY_TAU: float = 0.50
    # Enhanced gate settings
    ENABLE_GATE_CACHING: bool = True
    SEMANTIC_COHERENCE_WEIGHT: float = 0.10
    # Judge policy gray-zone bounds and stagnation threshold
    TAU_LO: float = 0.40
    TAU_HI: float = 0.60
    EPSILON_OVERLAP: float = 0.02
    LOW_BUDGET_TOKENS: int = 500
    USE_RERANK: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    USE_HYDE: bool = False
    MMR_LAMBDA: float = 0.45
    RETRIEVAL_POOL_K: int = 24
    # Hybrid search combining vector and BM25
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.7  # Weight for vector vs BM25 (0.7 = 70% vector, 30% BM25)
    ANCHOR_BONUS: float = 0.07  # Score bonus for candidates containing question anchors
    # Anchor orchestrator gate toggle
    ANCHOR_GATE_ON: bool = True
    # Lean-agent controls
    FACTOID_ONE_SHOT_RETRIEVAL: bool = True
    FACTOID_MIN_TOKENS_LEFT: int = 300
    USE_FACTOID_FINALIZER: bool = True
    # Judge policy for RAGAS invocation in agent loop: never | gray_zone | always
    JUDGE_POLICY: Literal["never", "gray_zone", "always"] = "gray_zone"
    JUDGE_MAX_CALLS_PER_Q: int = 1
    # Judge timing & thresholds
    JUDGE_PREGEN: bool = True  # Run Judge before generation
    JUDGE_MIN_CONF_FOR_STOP: float = 0.8  # Require judge OK at/above this to early STOP
    STRICT_STOP_REQUIRES_JUDGE_OK: bool = (
        True  # If True, don't early stop on f/o unless judge agrees
    )
    # Context quality thresholds
    ANCHOR_COVERAGE_TAU: float = 0.6  # Min anchor coverage for sufficiency
    CONFLICT_RISK_TAU: float = 0.25  # Above this, treat as insufficient
    BAUG_STOP_COVERAGE_MIN: float = 0.3  # Min coverage required for BAUG STOP
    BAUG_HIGH_OVERLAP_TAU: float = (
        0.7  # Allow STOP with high overlap even if coverage is low
    )
    BAUG_SLOT_COMPLETENESS_MIN: float = (
        0.6  # Slot completeness threshold for early ABSTAIN
    )
    # Anchor orchestrator thresholds
    NEW_HITS_EPS: float = 0.15  # Min ratio of new docs to continue
    FINE_FILTER_TAU: float = 0.15  # Min median fine score to keep exploring
    ANCHOR_PLATEAU_EPS: float = 0.05  # Min coverage gain between rounds
    MAX_WORKERS: int = 4  # Parallel anchor workers
    # Packing reserves: force-include anchor-bearing contexts when coverage is low
    RESERVE_ANCHOR_SLOTS: int = 2
    PACK_RESERVE_ON_LOW_COVERAGE: bool = True
    # Meta-aware pack fusion weights (final = w_fine*fine_sim + w_title*title_match + w_rank*rank_score)
    PACK_W_FINE: float = 0.90
    PACK_W_TITLE: float = 0.05
    PACK_W_RANK: float = 0.05
    # Per-domain cap in pack (to widen coverage)
    PACK_MAX_PER_DOMAIN: int = 2
    # Doc-level cap when list-like query
    PACK_MAX_PER_DOC_LIST: int = 2
    # Type-aware budgets (optional): overrides for list/factoid types
    LIST_CAP_TOKENS: int = 900
    FACTOID_CAP_TOKENS: int = 850
    LIST_RESERVE_SLOTS: int = 3
    FACTOID_RESERVE_SLOTS: int = 2

    # Scoring/analysis toggles
    USE_FINAL_SHORT_SCORING: bool = True
    FINAL_SHORT_ALLOW_ENTITY: bool = (
        False  # If True, entity extraction used in final_short
    )

    # Removed GATE_KIND - only UncertaintyGate is available

    log_dir: str = "logs"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
