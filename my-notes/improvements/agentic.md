# Agentic Framework Improvement Plan
**Date**: September 18, 2025

This document outlines the planned enhancements to evolve the current RAG pipeline into a more dynamic, multi-agent framework.

---

### 1. Activate and Enhance the "Judge" (Self-Correction) Module
- **Immediate Goal**: Modify the agent loop to **always** invoke the Judge module after the initial retrieval step.
- **Functionality**: The Judge will be a lightweight LLM call that assesses whether the retrieved context is sufficient to answer the question.
- **Logic**:
    - If `is_sufficient`, proceed to the generation step.
    - If `is_not_sufficient`, trigger a remedial action instead of giving up.

### 2. Implement Remedial Actions (Query Transformation)
When the Judge deems the context insufficient, the agent should not immediately abstain. It should instead try to improve the context. The next steps are to implement the following strategies:

- **Query Re-writing**: The agent should rephrase the original question to explore different angles. For example, changing "where did the ceo of salesforce previously work?" to "Marc Benioff career before Salesforce".
- **Query Decomposition**: For complex, multi-hop questions, the agent should break the query down into a series of simpler, sequential questions.
    - **Example**: For "how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?", the agent should first ask "Which seasons did Steve Nash make the 50-40-90 club?" and then use those results to ask about the stats for each specific season.

### 3. Improve Core Retriever
A better initial retrieval will reduce the number of times the agent needs to self-correct.
- **Implement Hybrid Search**: Augment the existing FAISS-based vector search with a keyword-based search algorithm like BM25. This will improve performance on queries with specific, must-match terms (names, acronyms, etc.) that dense retrieval can sometimes miss.
- **Add a Re-ranker**: After the initial retrieval (from either vector or hybrid search), use a more powerful cross-encoder model to re-rank the top ~50 candidates to find the most relevant passages. This adds a small amount of latency but can significantly improve the quality of the context passed to the Judge and Generator.

### 4. Evolve to a Multi-Agent System
The long-term architectural goal is to refactor the system into a **multi-agent framework**. This approach treats the problem-solving process as a collaboration between specialized agents, managed by an orchestrator.

-   **`Orchestrator`**: The main agent loop (`loop.py`) that directs traffic.
-   **`QueryAnalyzerAgent`**: Implements Query Transformation.
-   **`SearchAgent`**: Implements the Core Retriever improvements.
-   **`CritiqueAgent`**: The formal implementation of the "Judge" module.
-   **`SynthesisAgent`**: Generates the final answer from high-quality, pre-vetted context.
