"""Configuration settings for the agentic RAG system."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: Literal["openai", "ollama", "local"] = Field(
        default="openai", description="LLM provider to use"
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )

    # Token and Round Limits
    max_tokens_total: int = Field(
        default=3500, description="Maximum total tokens allowed"
    )
    max_rounds: int = Field(default=2, description="Maximum number of RAG rounds")
    low_budget_tokens: int = Field(
        default=500, description="Low budget token threshold"
    )

    # Retrieval Configuration
    retrieval_k: int = Field(default=8, description="Number of documents to retrieve")
    graph_k: int = Field(
        default=20, description="Number of graph neighbors to consider"
    )
    use_rerank: bool = Field(default=True, description="Whether to use reranking")

    # Quality Thresholds
    faithfulness_tau: float = Field(
        default=0.75, description="Faithfulness threshold for quality control"
    )
    overlap_tau: float = Field(
        default=0.50, description="Overlap threshold for duplicate detection"
    )

    # Directory Paths
    data_dir: str = Field(default="./data/corpus", description="Data directory path")
    index_dir: str = Field(
        default="./artifacts/faiss", description="FAISS index directory"
    )
    log_dir: str = Field(default="./artifacts/logs", description="Logs directory path")
    report_dir: str = Field(
        default="./artifacts/reports", description="Reports directory path"
    )

    # System Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = SettingsConfigDict(
        extra="ignore",  # Ignore extra environment variables
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Singleton instance
settings = Settings()
