"""Retriever policy switcher for selecting between vector and graph retrieval \
strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agentic_rag.retriever.vector import ContextChunk, VectorRetriever


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class SwitchingDecision(BaseModel):
    """Decision made by the retriever switcher."""

    strategy: RetrievalStrategy
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = {}


RetrievalContext = List[ContextChunk]


class BaseSwitcher(ABC):
    """Abstract base class for retriever switchers."""

    def __init__(
        self, default_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR
    ) -> None:
        """
        Initialize retriever switcher.

        Args:
            default_strategy: Default retrieval strategy to use
        """
        self.default_strategy = default_strategy

    @abstractmethod
    def decide_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SwitchingDecision:
        """
        Decide which retrieval strategy to use.

        Args:
            query: Input query
            round_number: Current round number (0-indexed)
            previous_contexts: Contexts from previous rounds
            metadata: Optional additional metadata

        Returns:
            Switching decision with strategy and reasoning
        """
        pass


class RetrieverSwitcher(BaseSwitcher):
    """Main retriever switcher with multiple decision strategies."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: Optional[Any] = None,  # TODO: Add graph retriever type
        switching_strategy: str = "round_robin",
        vector_threshold: float = 0.7,
        graph_keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize retriever switcher.

        Args:
            vector_retriever: Vector-based retriever
            graph_retriever: Graph-based retriever (optional)
            switching_strategy: Strategy for switching (
                "round_robin", "adaptive", "keyword"
            )
            vector_threshold: Threshold for vector retrieval confidence
            graph_keywords: Keywords that suggest graph retrieval
            **kwargs: Additional configuration
        """
        super().__init__()
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.switching_strategy = switching_strategy
        self.vector_threshold = vector_threshold

        # Keywords that might benefit from graph retrieval
        self.graph_keywords = graph_keywords or [
            "related",
            "connected",
            "relationship",
            "between",
            "compare",
            "contrast",
            "similar",
            "different",
            "cause",
            "effect",
            "influence",
        ]

    def decide_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SwitchingDecision:
        """Decide retrieval strategy based on configured approach."""
        if self.switching_strategy == "round_robin":
            return self._round_robin_strategy(query, round_number, previous_contexts)
        elif self.switching_strategy == "adaptive":
            return self._adaptive_strategy(
                query, round_number, previous_contexts, metadata
            )
        elif self.switching_strategy == "keyword":
            return self._keyword_strategy(query, round_number, previous_contexts)
        else:
            # Default to vector retrieval
            return SwitchingDecision(
                strategy=RetrievalStrategy.VECTOR,
                confidence=1.0,
                reasoning="Using default vector strategy",
            )

    def _round_robin_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
    ) -> SwitchingDecision:
        """
        Simple round-robin strategy alternating between strategies.

        Args:
            query: Input query
            round_number: Current round number
            previous_contexts: Previous contexts

        Returns:
            Switching decision
        """
        # For now, always use vector since graph retriever is not implemented
        strategy = RetrievalStrategy.VECTOR
        reasoning = (
            f"Round {round_number}: using vector retrieval (graph not yet implemented)"
        )

        return SwitchingDecision(strategy=strategy, confidence=1.0, reasoning=reasoning)

    def _adaptive_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SwitchingDecision:
        """
        Adaptive strategy based on previous round performance.

        Args:
            query: Input query
            round_number: Current round number
            previous_contexts: Previous contexts
            metadata: Additional metadata

        Returns:
            Switching decision
        """
        # TODO: Implement adaptive logic based on previous performance
        # For now, default to vector retrieval
        return SwitchingDecision(
            strategy=RetrievalStrategy.VECTOR,
            confidence=0.8,
            reasoning="Adaptive strategy defaulting to vector (not fully implemented)",
        )

    def _keyword_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
    ) -> SwitchingDecision:
        """
        Keyword-based strategy for selecting retrieval method.

        Args:
            query: Input query
            round_number: Current round number
            previous_contexts: Previous contexts

        Returns:
            Switching decision
        """
        query_lower = query.lower()

        # Check for graph-indicating keywords
        graph_score = sum(
            1 for keyword in self.graph_keywords if keyword in query_lower
        )

        if graph_score > 0 and self.graph_retriever is not None:
            return SwitchingDecision(
                strategy=RetrievalStrategy.GRAPH,
                confidence=min(0.9, 0.5 + graph_score * 0.1),
                reasoning=(
                    f"Found {graph_score} graph-indicating keywords: "
                    f"{self.graph_keywords}"
                ),
            )
        else:
            return SwitchingDecision(
                strategy=RetrievalStrategy.VECTOR,
                confidence=0.9,
                reasoning="No graph keywords found or graph retriever not available",
            )

    def _assess_vector_performance(
        self,
        contexts: List[RetrievalContext],
    ) -> float:
        """
        Assess performance of vector retrieval from previous rounds.

        Args:
            contexts: Previous retrieval contexts

        Returns:
            Performance score (0.0 to 1.0)
        """
        if not contexts:
            return 0.5  # Neutral score

        # Simple heuristic: average of retrieval scores
        total_score = 0.0
        total_chunks = 0

        for context in contexts:
            total_score += sum(
                c["score"] for c in context if c.get("score") is not None
            )
            total_chunks += len(context)

        if total_chunks == 0:
            return 0.5

        return min(1.0, total_score / total_chunks)

    def retrieve_with_strategy(
        self,
        query: str,
        strategy: RetrievalStrategy,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalContext:
        """
        Retrieve using the specified strategy.

        Args:
            query: Query string
            strategy: Retrieval strategy to use
            k: Number of results to return
            **kwargs: Additional retrieval parameters

        Returns:
            Retrieval context
        """
        k_val = k or 8
        if strategy == RetrievalStrategy.VECTOR:
            return self.vector_retriever.retrieve(query, k_val, **kwargs)
        elif strategy == RetrievalStrategy.GRAPH:
            if self.graph_retriever is None:
                # Fallback to vector retrieval
                return self.vector_retriever.retrieve(query, k_val, **kwargs)
            # TODO: Implement graph retrieval
            raise NotImplementedError("Graph retrieval not yet implemented")
        elif strategy == RetrievalStrategy.HYBRID:
            # TODO: Implement hybrid retrieval
            # For now, fallback to vector
            return self.vector_retriever.retrieve(query, k_val, **kwargs)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    def get_available_strategies(self) -> List[RetrievalStrategy]:
        """
        Get list of available retrieval strategies.

        Returns:
            List of available strategies
        """
        strategies = [RetrievalStrategy.VECTOR]

        if self.graph_retriever is not None:
            strategies.extend([RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID])

        return strategies

    def get_stats(self) -> Dict[str, Any]:
        """
        Get switcher statistics.

        Returns:
            Dictionary with switcher statistics
        """
        return {
            "switching_strategy": self.switching_strategy,
            "has_graph_retriever": self.graph_retriever is not None,
            "vector_threshold": self.vector_threshold,
            "graph_keywords_count": len(self.graph_keywords),
            "available_strategies": [s.value for s in self.get_available_strategies()],
        }


class SimpleSwitcher(BaseSwitcher):
    """Simple switcher that always uses the same strategy."""

    def __init__(self, strategy: RetrievalStrategy = RetrievalStrategy.VECTOR) -> None:
        """
        Initialize simple switcher.

        Args:
            strategy: Strategy to always use
        """
        super().__init__(strategy)
        self.strategy = strategy

    def decide_strategy(
        self,
        query: str,
        round_number: int,
        previous_contexts: List[RetrievalContext],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SwitchingDecision:
        """Always return the configured strategy."""
        return SwitchingDecision(
            strategy=self.strategy,
            confidence=1.0,
            reasoning=f"Simple switcher always uses {self.strategy.value}",
        )
