# Agentic RAG Framework Notes
**Date**: September 18, 2025

## Current Implementation

The current system operates as a basic Retrieval-Augmented Generation (RAG) pipeline. For a given user query, it performs a single pass of vector-based retrieval to fetch relevant context, and then generates an answer based on that context.

A recent evaluation run (`1758126979`) highlighted the limitations of this approach. While effective for simple, single-hop questions, it consistently fails on complex queries that require synthesizing information from multiple sources or understanding nuanced statistical questions.

The analysis showed that the core problem is **retrieval failure**. The system retrieves documents that are thematically related but factually insufficient to answer the question, leading to a high rate of "I don't know" responses (30% Abstain Rate).

Crucially, the evaluation showed that the more advanced agentic features like the "Judge" module were **not being invoked at all** (`Judge%Invoked: 0.000`).

## Next Steps for Improvement

To address these limitations, the immediate priority is to evolve the system from a simple pipeline into a more dynamic, "agentic" framework capable of reasoning about its retrieval quality and adapting its strategy.

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

---

## Evolving to a Multi-Agent System
**Date**: September 18, 2025

The limitations of the single-agent approach suggest that the next architectural evolution is to refactor the system into a **multi-agent framework**. This approach treats the problem-solving process as a collaboration between specialized agents, each with a distinct role, managed by an orchestrator.

### Limitations of a Single Agent
1.  **Cognitive Overload**: A single agent's prompt becomes convoluted as it tries to be an expert at query analysis, retrieval, critique, and synthesis simultaneously.
2.  **Lack of Specialized Skills**: A general-purpose agent is a jack-of-all-trades. Different tasks, like skepticism (critique) and prose generation (synthesis), benefit from specialized prompting and logic.
3.  **Brittle Reasoning**: A single agent can get stuck in reasoning loops and struggles to "step back" and question its own flawed premises.

### Proposed Multi-Agent Architecture: A "Society of Agents"
The existing "Next Steps" map directly to the roles in a multi-agent system:

-   **`Orchestrator`**: The main agent loop (`loop.py`) that directs traffic and manages the state between other agents. It does not perform the tasks itself.
-   **`QueryAnalyzerAgent`**: Implements the **Query Transformation** step. Its sole job is to analyze the initial query for complexity, ambiguity, or the need for decomposition.
-   **`SearchAgent`**: Implements the **Core Retriever** improvements. This agent takes a query and determines the best retrieval strategy (e.g., vector, keyword, hybrid) to gather information.
-   **`CritiqueAgent`**: This is the formal implementation of the **"Judge" module**. Its only purpose is to assess the context provided by the `SearchAgent` and decide if it's sufficient to answer the question. It is designed to be skeptical.
-   **`SynthesisAgent`**: This agent is only activated when the `CritiqueAgent` gives approval. It takes the high-quality, pre-vetted context and generates the final, well-cited answer for the user.

This modular design will make the system more robust, easier to test, and more scalable. New capabilities can be added by creating new specialized agents without disrupting the entire system.
