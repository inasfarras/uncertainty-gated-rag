# Project Notes

This directory contains notes, design documents, and experiment logs for the Uncertainty-Gated RAG project.

## Directory Structure

-   `documentation_report.md`: The primary, cumulative report documenting the project's progress, major updates, and architectural changes. This is the canonical source of truth for the project's history.

-   `/design`: Contains high-level design documents and conceptual notes. These files describe the architecture and "how it should work."
    -   `agentic.md`: Notes on the agentic RAG framework and the evolution to a multi-agent system.
    -   `gate.md`: Detailed notes on the Uncertainty Gate's implementation.

-   `/experiments`: Contains logs, command sequences, and results from specific experimental runs.
    -   Files should be named using the convention: `YYYY-MM-DD_short-description.md`.
    -   These notes should be self-contained and allow someone to reproduce a specific experiment.

-   `/reference`: Contains supplementary documentation and explanations of core concepts.
    -   `metrics.md`: A detailed explanation of all evaluation metrics used in the project.
