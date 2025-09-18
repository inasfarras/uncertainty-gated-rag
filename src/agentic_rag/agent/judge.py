"""
Judge module for assessing context sufficiency and triggering remedial actions.

This module implements a lightweight LLM-based judge that evaluates whether
retrieved context is sufficient to answer a given question.
"""

import json
from typing import Any, Dict, List, Optional

from agentic_rag.config import Settings
from agentic_rag.models.adapter import ChatMessage, OpenAIAdapter


class ContextSufficiencyResult:
    """Result of context sufficiency assessment."""

    def __init__(
        self,
        is_sufficient: bool,
        confidence: float,
        reasoning: str,
        suggested_action: str,
        query_transformations: Optional[List[str]] = None,
    ):
        self.is_sufficient = is_sufficient
        self.confidence = confidence
        self.reasoning = reasoning
        self.suggested_action = suggested_action
        self.query_transformations = query_transformations or []


class Judge:
    """
    Judge module that assesses context sufficiency and suggests remedial actions.

    The Judge evaluates whether retrieved context contains sufficient information
    to answer a question accurately. If insufficient, it suggests query transformations
    or decompositions to improve retrieval.
    """

    def __init__(self, llm_client: OpenAIAdapter, settings: Settings):
        self.llm = llm_client
        self.settings = settings

    def assess_context_sufficiency(
        self, question: str, contexts: List[Dict[str, Any]], round_idx: int = 0
    ) -> ContextSufficiencyResult:
        """
        Assess whether the retrieved contexts are sufficient to answer the question.

        Args:
            question: The original user question
            contexts: List of retrieved context blocks with 'id' and 'text'
            round_idx: Current retrieval round (0-based)

        Returns:
            ContextSufficiencyResult with assessment details
        """
        # Build context summary for the judge
        context_summary = self._build_context_summary(contexts)

        # Create judge prompt
        prompt = self._build_judge_prompt(question, context_summary, round_idx)

        try:
            # Get judge assessment
            response, usage = self.llm.chat(
                messages=prompt,
                max_tokens=200,  # Judge needs less output than generation
                temperature=0.0,  # Deterministic assessment
            )

            # Parse judge response
            return self._parse_judge_response(response, question)

        except Exception as e:
            # Fallback: assume insufficient if judge fails
            return ContextSufficiencyResult(
                is_sufficient=False,
                confidence=0.5,
                reasoning=f"Judge assessment failed: {str(e)}",
                suggested_action="RETRIEVE_MORE",
                query_transformations=[],
            )

    def _build_context_summary(self, contexts: List[Dict[str, Any]]) -> str:
        """Build a concise summary of retrieved contexts for the judge."""
        if not contexts:
            return "No contexts retrieved."

        summary_parts = []
        for i, ctx in enumerate(contexts[:8]):  # Limit to first 8 contexts
            text = ctx.get("text", "")[:200]  # First 200 chars
            summary_parts.append(f"Context {i + 1}: {text}...")

        return "\n".join(summary_parts)

    def _build_judge_prompt(
        self, question: str, context_summary: str, round_idx: int
    ) -> List[ChatMessage]:
        """Build the prompt for the judge assessment."""

        system_content = """You are a Judge that evaluates whether retrieved contexts contain sufficient information to answer a question accurately.

Your task:
1. Analyze if the contexts provide enough specific information to answer the question
2. Consider whether the question requires information not present in the contexts
3. Suggest query transformations if the contexts are insufficient

Respond ONLY with a JSON object in this exact format:
{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your assessment",
    "suggested_action": "STOP" | "RETRIEVE_MORE" | "TRANSFORM_QUERY",
    "query_transformations": ["alternative query 1", "alternative query 2"]
}

Guidelines:
- is_sufficient=true only if contexts contain specific, relevant information to answer the question
- For factual questions, require specific facts/numbers/names, not just general information
- For complex questions, ensure all parts can be answered
- confidence should reflect certainty of your assessment
- query_transformations should rephrase or break down the question if contexts are insufficient"""

        user_content = f"""Question: {question}

Retrieved Contexts:
{context_summary}

Round: {round_idx + 1}

Assess whether these contexts are sufficient to answer the question accurately."""

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]

    def _parse_judge_response(
        self, response: str, question: str
    ) -> ContextSufficiencyResult:
        """Parse the judge's JSON response into a structured result."""
        try:
            # Try to parse JSON response
            data = json.loads(response.strip())

            return ContextSufficiencyResult(
                is_sufficient=data.get("is_sufficient", False),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                suggested_action=data.get("suggested_action", "RETRIEVE_MORE"),
                query_transformations=data.get("query_transformations", []),
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback parsing for malformed responses
            response_lower = response.lower()

            # Simple heuristics for fallback parsing
            is_sufficient = any(
                word in response_lower
                for word in ["sufficient", "enough", "adequate", "complete"]
            ) and not any(
                word in response_lower
                for word in [
                    "insufficient",
                    "not enough",
                    "inadequate",
                    "incomplete",
                    "missing",
                ]
            )

            confidence = 0.3  # Low confidence for fallback parsing

            return ContextSufficiencyResult(
                is_sufficient=is_sufficient,
                confidence=confidence,
                reasoning=f"Fallback parsing of response: {response[:100]}",
                suggested_action="RETRIEVE_MORE" if not is_sufficient else "STOP",
                query_transformations=[],
            )


class QueryTransformer:
    """
    Query transformation module for improving retrieval through query rewriting
    and decomposition.
    """

    def __init__(self, llm_client: OpenAIAdapter):
        self.llm = llm_client

    def transform_query(
        self,
        original_query: str,
        context_assessment: ContextSufficiencyResult,
        failed_contexts: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Transform the original query to improve retrieval.

        Args:
            original_query: The original user question
            context_assessment: Judge's assessment of current contexts
            failed_contexts: Contexts that were deemed insufficient

        Returns:
            List of transformed queries to try
        """
        # If judge already provided transformations, use those
        if context_assessment.query_transformations:
            return context_assessment.query_transformations[:3]  # Limit to 3

        # Generate transformations using LLM
        return self._generate_query_transformations(original_query, failed_contexts)

    def _generate_query_transformations(
        self, query: str, failed_contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate query transformations using LLM."""

        context_summary = "\n".join(
            [f"- {ctx.get('text', '')[:100]}..." for ctx in failed_contexts[:3]]
        )

        system_content = """You are a Query Transformer that rewrites questions to improve information retrieval.

Your task: Given a question and contexts that were insufficient, generate 2-3 alternative queries that might retrieve better information.

Strategies:
1. Rephrase using synonyms or alternative terms
2. Break complex questions into simpler parts
3. Add specific context or constraints
4. Focus on key entities or concepts

Respond with a JSON array of 2-3 alternative queries:
["alternative query 1", "alternative query 2", "alternative query 3"]"""

        user_content = f"""Original Question: {query}

Insufficient Contexts Found:
{context_summary}

Generate 2-3 alternative queries that might retrieve better information to answer the original question."""

        try:
            prompt = [
                ChatMessage(role="system", content=system_content),
                ChatMessage(role="user", content=user_content),
            ]

            response, usage = self.llm.chat(
                messages=prompt,
                max_tokens=150,
                temperature=0.3,  # Slight creativity for transformations
            )

            # Parse JSON response
            transformations = json.loads(response.strip())
            if isinstance(transformations, list):
                return transformations[:3]  # Limit to 3
            else:
                return [str(transformations)]

        except Exception:
            # Fallback: simple entity-based transformation
            return self._fallback_transformations(query)

    def _fallback_transformations(self, query: str) -> List[str]:
        """Generate simple fallback transformations without LLM."""
        transformations = []

        # Simple rephrasings
        if "who is" in query.lower():
            transformations.append(query.replace("who is", "information about"))
        elif "what is" in query.lower():
            transformations.append(query.replace("what is", "details about"))
        elif "where" in query.lower():
            transformations.append(query.replace("where", "location of"))
        elif "when" in query.lower():
            transformations.append(query.replace("when", "time of"))

        # Add one generic transformation
        transformations.append(f"background information {query}")

        return transformations[:2]  # Return up to 2 fallback transformations


def create_judge(llm_client: OpenAIAdapter, settings: Settings) -> Judge:
    """Factory function to create a Judge instance."""
    return Judge(llm_client, settings)


def create_query_transformer(llm_client: OpenAIAdapter) -> QueryTransformer:
    """Factory function to create a QueryTransformer instance."""
    return QueryTransformer(llm_client)
