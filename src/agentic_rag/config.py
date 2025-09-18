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
    MAX_ROUNDS: int = 2
    RETRIEVAL_K: int = 8
    PROBE_FACTOR: int = 4
    GRAPH_K: int = 20
    FAITHFULNESS_TAU: float = 0.65
    OVERLAP_TAU: float = 0.40
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
    USE_RERANK: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    USE_HYDE: bool = True
    MMR_LAMBDA: float = 0.4
    RETRIEVAL_POOL_K: int = 50
    # Hybrid search combining vector and BM25
    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.7  # Weight for vector vs BM25 (0.7 = 70% vector, 30% BM25)
    # Judge policy for RAGAS invocation in agent loop: never | gray_zone | always
    JUDGE_POLICY: Literal["never", "gray_zone", "always"] = (
        "always"  # Enable judge by default
    )
    # Removed GATE_KIND - only UncertaintyGate is available

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
