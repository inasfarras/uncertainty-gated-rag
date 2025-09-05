"""FastAPI main application with query endpoint."""

from typing import Any, Dict, List, Optional

from agentic_rag.agent.loop import AgenticRAGLoop
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
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    answer: str
    confidence: float
    total_rounds: int
    contexts_used: List[Dict[str, Any]]
    reasoning_trace: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    components: Dict[str, str]


# Global agent instance (will be initialized on startup)
agent: Optional[AgenticRAGLoop] = None


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

    @app.on_event("startup")
    async def startup_event():
        """Initialize components on startup."""
        global agent
        logger.info("Starting up Agentic RAG API")

        try:
            # TODO: Initialize agent components
            # agent = initialize_agent()
            logger.info("Agent initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down Agentic RAG API")

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.

        Returns:
            Health status and component information
        """
        components = {
            "agent": "ready" if agent is not None else "not_initialized",
            "llm_provider": settings.llm_provider,
            "embed_model": settings.embed_model,
        }

        status = "healthy" if agent is not None else "unhealthy"

        return HealthResponse(
            status=status,
            version="0.1.0",
            components=components,
        )

    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest) -> QueryResponse:
        """
        Process a query through the agentic RAG system.

        Args:
            request: Query request with parameters

        Returns:
            Query response with answer and metadata

        Raises:
            HTTPException: If agent is not initialized or processing fails
        """
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
            agent_response = await agent.process_query(
                request.query,
                # max_rounds=request.max_rounds or settings.max_rounds,
                # Additional parameters...
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            # Convert contexts to serializable format
            contexts_data = []
            for context in agent_response.contexts_used:
                context_data = {
                    "query": context.query,
                    "chunks": [
                        {
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "chunk_id": chunk.chunk_id,
                        }
                        for chunk in context.chunks
                    ],
                    "scores": context.scores,
                    "metadata": context.metadata,
                }
                contexts_data.append(context_data)

            response = QueryResponse(
                query=request.query,
                answer=agent_response.answer,
                confidence=agent_response.confidence,
                total_rounds=agent_response.total_rounds,
                contexts_used=contexts_data,
                reasoning_trace=agent_response.reasoning_trace,
                processing_time_ms=processing_time,
                metadata={
                    **agent_response.metadata,
                    "request_metadata": request.metadata,
                },
            )

            logger.info(
                f"Query processed successfully in {processing_time:.2f}ms "
                f"with {agent_response.total_rounds} rounds"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {str(e)}",
            )

    @app.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        if agent is None:
            return {"error": "Agent not initialized"}

        try:
            stats = agent.get_stats()
            return {
                "agent_stats": stats,
                "config": {
                    "llm_provider": settings.llm_provider,
                    "llm_model": settings.llm_model,
                    "embed_model": settings.embed_model,
                    "max_rounds": settings.max_rounds,
                    "retrieval_k": settings.retrieval_k,
                },
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting stats: {str(e)}",
            )

    return app


def initialize_agent() -> AgenticRAGLoop:
    """
    Initialize the agentic RAG agent with configured components.

    Returns:
        Initialized agent instance
    """
    # TODO: Implement agent initialization
    # This would involve:
    # 1. Creating LLM adapter based on settings.llm_provider
    # 2. Creating embedding encoder
    # 3. Loading/creating vector store
    # 4. Creating retriever with optional reranker
    # 5. Creating uncertainty gate
    # 6. Creating retriever switcher (optional)
    # 7. Assembling everything into AgenticRAGLoop

    raise NotImplementedError("Agent initialization not yet implemented")


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
