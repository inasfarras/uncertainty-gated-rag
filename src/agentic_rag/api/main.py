"""FastAPI main application with query endpoint."""

from typing import Any, List, Optional

from agentic_rag.agent.loop import Agent
from agentic_rag.config import settings
from agentic_rag.logging import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str
    max_rounds: Optional[int] = None
    use_rerank: Optional[bool] = None
    retrieval_k: Optional[int] = None
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    answer: str
    confidence: float
    total_rounds: int
    contexts_used: List[dict[str, Any]]
    reasoning_trace: List[str]
    processing_time_ms: float
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    components: dict[str, str]


# Global agent instance (will be initialized on startup)
agent: Optional[Agent] = None


def _configure_startup_event(app: FastAPI) -> None:
    @app.on_event("startup")
    async def startup_event() -> None:
        global agent
        logger.info("Starting up Agentic RAG API")

        try:
            agent = Agent()
            logger.info("Agent initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise


def _configure_shutdown_event(app: FastAPI) -> None:
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("Shutting down Agentic RAG API")


def _configure_health_endpoint(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        components = {
            "agent": "ready" if agent is not None else "not_initialized",
            "llm_provider": settings.EMBED_BACKEND,
            "embed_model": settings.EMBED_MODEL,
        }

        status = "healthy" if agent is not None else "unhealthy"

        return HealthResponse(
            status=status,
            version="0.1.0",
            components=components,
        )


def _configure_query_endpoint(app: FastAPI) -> None:
    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest) -> QueryResponse:
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized. Check service health.",
            )

        logger.info(f"Processing query: {request.query}")

        try:
            # Process query through agent
            import time

            start_time = time.perf_counter()

            # TODO: Pass request parameters to agent
            agent_response = agent.answer(
                request.query,
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            response = QueryResponse(
                query=request.query,
                answer=agent_response.get("final_answer", ""),
                confidence=agent_response.get("final_f", 0.0),
                total_rounds=agent_response.get("rounds", 0),
                contexts_used=[],
                reasoning_trace=[],
                processing_time_ms=processing_time,
                metadata={
                    "request_metadata": request.metadata,
                },
            )

            logger.info(
                f"Query processed successfully in {processing_time:.2f}ms "
                f"with {agent_response.get('rounds', 0)} rounds"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {str(e)}",
            ) from e


def _configure_stats_endpoint(app: FastAPI) -> None:
    @app.get("/stats")
    async def get_stats() -> dict[str, Any]:
        if agent is None:
            return {"error": "Agent not initialized"}

        try:
            stats: dict[str, Any] = {}
            return {
                "agent_stats": stats,
                "config": {
                    "llm_provider": settings.EMBED_BACKEND,
                    "llm_model": settings.LLM_MODEL,
                    "embed_model": settings.EMBED_MODEL,
                    "max_rounds": settings.MAX_ROUNDS,
                    "retrieval_k": settings.RETRIEVAL_K,
                },
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting stats: {str(e)}",
            ) from e


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Agentic RAG API",
        description="API for agentic RAG system with uncertainty-based iteration",
        version="0.1.0",
    )

    _configure_startup_event(app)
    _configure_shutdown_event(app)
    _configure_health_endpoint(app)
    _configure_query_endpoint(app)
    _configure_stats_endpoint(app)

    return app


def initialize_agent() -> Agent:
    """
    Initialize the agentic RAG agent with configured components.

    Returns:
        Initialized agent instance
    """
    return Agent()


# Create the FastAPI app
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentic_rag.api.main:app",
        host=settings.api_host if hasattr(settings, "api_host") else "0.0.0.0",
        port=settings.api_port if hasattr(settings, "api_port") else 8000,
        reload=True,
    )
