# Core Evaluation Metrics Explained
**Date**: September 18, 2025

This document details the mechanism behind the key evaluation metrics used in this project.

---

## 1. Core Metrics

### Overlap (`final_o`)
-   **What it Measures**: How well the generated answer is **grounded in the retrieved context**. It quantifies the proportion of sentences in the answer directly supported by a cited document.
-   **Calculation**:
    1.  The answer is split into sentences.
    2.  Each sentence is validated: it must have exactly one valid citation and must not be an "I don't know" statement.
    3.  A **cosine similarity** check is performed between the sentence embedding and the cited context's embedding.
    4.  The final score is the ratio of supported sentences to the total number of sentences.
    \[ \text{Overlap} = \frac{\text{Number of Supported Sentences}}{\text{Total Number of Sentences}} \]
-   **Interpretation**: A high overlap (e.g., 1.0) indicates that the answer's claims are strongly supported by the retrieved context. This can be high even if Exact Match (EM) or F1 Score is low, because overlap focuses on contextual grounding rather than lexical identity to the gold answer.

### Faithfulness (`final_f`)
-   **What it Measures**: The factual accuracy of the answer **based on the provided context**.
-   **Calculation (Fallback Method)**: To avoid the high cost of LLM-based evaluation, a proxy is used.
    -   If the agent incorrectly says "I don't know," the score is **0.0**.
    -   Otherwise, the score is calculated as:
    \[ \text{Faithfulness} = \min(1.0, 0.6 + 0.4 \times \text{Overlap}) \]
    -   This provides a baseline score of **0.6** for a plausible but unsupported answer, which increases with better context grounding.

### F1 Score (`Avg F1`)
-   **What it Measures**: The token-level overlap between the generated answer and the ground-truth answer, balancing precision and recall. It's a more forgiving metric than Exact Match.
-   **Calculation**: Standard F1 score based on shared tokens after normalization (lowercase, punctuation removal).
-   **Purpose**: The primary metric for **task accuracy**.

### Abstain Rate
-   **What it Measures**: The proportion of questions where the system chose not to provide an answer (e.g., responded with "I don't know").
-   **Calculation**: `(Number of Abstain Responses) / (Total Number of Questions)`
-   **Purpose**: A key indicator of the uncertainty gate's behavior. A high rate isn't necessarily bad if it correlates with a reduction in hallucinations.

## 2. Secondary & Diagnostic Metrics

### Exact Match (`Avg EM`)
-   **What it Measures**: Whether the generated answer matches the ground-truth answer **perfectly** after normalization.
-   **Purpose**: A strict metric, useful for short, factual questions.

### Total Tokens (`Avg Total Tokens`)
-   **What it Measures**: The average number of LLM tokens (prompt + completion) used per question across all rounds.
-   **Purpose**: A proxy for **computational cost**.

### Latency (`P50 Latency (ms)`)
-   **What it Measures**: The **median** time taken to process a question from start to finish.
-   **Purpose**: Measures the system's speed and responsiveness.

### IDK + Citation Count (`IDK+Cit Count`)
-   **What it Measures**: A sanity check. Counts the number of times an "I don't know" sentence incorrectly contains a citation.
-   **Purpose**: This value **must be 0**. If it's higher, it indicates a flaw in the prompt or generation logic.

## 3. Best Practices (Rule-of-Thumb)
-   **Thresholds**: The similarity threshold for overlap (`τ_sim`) should be around **0.6–0.7** to avoid being too lenient.
-   **Reporting**: For primary tables, report on `N`, `Faithfulness`, `Overlap`, `F1`, `Tokens`, `Latency`, and `Abstain Rate`. Leave stricter or more detailed metrics like `EM` for an appendix.
-   **Paired Analysis**:
    -   Compare **Faithfulness/Overlap** vs. **F1/EM** to distinguish between answers that are faithful to bad context versus answers that are simply wrong.
    -   Compare **Tokens/Latency** vs. **Abstain Rate** to evaluate the efficiency of the uncertainty gate.
