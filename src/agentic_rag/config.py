from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_ORG: str | None = None

    LLM_MODEL: str = "gpt-4o-mini"
    EMBED_BACKEND: Literal["openai", "st", "mock"] = "openai"
    EMBED_MODEL: str = "text-embedding-3-small"
    ST_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    FAISS_INDEX_PATH: str = "artifacts/crag_faiss"
    MAX_TOKENS_TOTAL: int = 3500
    # Cap on total context tokens packed into prompts
    CONTEXT_TOKEN_CAP: int = 2000
    MAX_ROUNDS: int = 2
    RETRIEVAL_K: int = 8
    GRAPH_K: int = 20
    FAITHFULNESS_TAU: float = 0.75
    OVERLAP_TAU: float = 0.50
    OVERLAP_SIM_TAU: float = 0.58
    LOW_BUDGET_TOKENS: int = 500
    USE_RERANK: bool = False

    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
