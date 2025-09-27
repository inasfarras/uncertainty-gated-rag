# Agentic RAG Framework Notes
**Date**: September 18, 2025

## Current Implementation

The current system operates as a basic Retrieval-Augmented Generation (RAG) pipeline. For a given user query, it performs a single pass of vector-based retrieval to fetch relevant context, and then generates an answer based on that context.

A recent evaluation run (`1758126979`) highlighted the limitations of this approach. While effective for simple, single-hop questions, it consistently fails on complex queries that require synthesizing information from multiple sources or understanding nuanced statistical questions.

The analysis showed that the core problem is **retrieval failure**. The system retrieves documents that are thematically related but factually insufficient to answer the question, leading to a high rate of "I don't know" responses (30% Abstain Rate).

Crucially, the evaluation showed that the more advanced agentic features like the "Judge" module were **not being invoked at all** (`Judge%Invoked: 0.000`).
