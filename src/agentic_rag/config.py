from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_ORG: str | None = None

    LLM_MODEL: str = "gpt-4o-mini"
    EMBED_BACKEND: Literal["openai", "st"] = "openai"
    EMBED_MODEL: str = "text-embedding-3-small"
    ST_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    FAISS_INDEX_PATH: str = "artifacts/faiss"
    MAX_TOKENS_TOTAL: int = 3500
    MAX_ROUNDS: int = 2
    RETRIEVAL_K: int = 8
    GRAPH_K: int = 20
    FAITHFULNESS_TAU: float = 0.75
    OVERLAP_TAU: float = 0.50
    LOW_BUDGET_TOKENS: int = 500
    USE_RERANK: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
