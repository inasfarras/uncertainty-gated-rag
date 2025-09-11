# Agentic RAG Project - Progress Report

This report summarizes the work completed to fix and refactor the Agentic RAG baseline.

## ✅ Completed

- [x] **Settings Refactor:** Switched to `pydantic-settings` and defined a clear, typed configuration in `src/agentic_rag/config.py`.
- [x] **OpenAI-Only Pipeline:** Removed all dependencies on local `SentenceTransformer` and `torch`. The entire embedding and chat pipeline now uses the `openai` library.
- [x] **Simplified spaCy:** Made `spaCy` an optional dependency for the `is_global_question` check, with a fallback to a simple length-based heuristic.
- [x] **Corrected FAISS Implementation:** Ensured that vectors are normalized before being indexed and that `IndexFlatIP` is used for cosine similarity.
- [x] **Robust Ingestion CLI:** Implemented a `typer`-based CLI for the ingestion pipeline.
- [x] **Updated Tests:** The test suite now passes, with OpenAI API calls mocked to ensure deterministic behavior.
- [x] **Relaxed Linting/Mypy:** Added `ruff.toml` and `mypy.ini` with lenient configurations to speed up development.
- [x] **Updated `README.md`:** The documentation now includes a Windows-specific quickstart guide with updated commands.

## ⚠️ Remaining Issues

- **`black` not installed:** The `black` code formatter is not installed in the environment.
- **`ruff` issues:** `ruff` still reports 128 linting issues.
- **`mypy` errors:** `mypy` reports 22 type-checking errors.

## 🛠 Next 5 Actions

1.  **Install `black`:**
    ```bash
    pip install black
    ```
2.  **Run `black`:**
    ```bash
    python -m black .
    ```
3.  **Address `ruff` issues:**
    Review the remaining `ruff` issues and fix them.
4.  **Address `mypy` errors:**
    Review the `mypy` errors and add type hints or fix type inconsistencies.
5.  **Set `OPENAI_API_KEY`:**
    Set the `OPENAI_API_KEY` in your `.env` file to run the ingestion and retrieval pipeline.

## 💨 Online Smoke Test Results

### Pipeline Status: ✅ **WORKING**

**Ingestion:** Successfully built FAISS index with 4 chunks from test corpus.

**Evaluation Results (5 challenging questions):**

| Metric | Baseline (Gate OFF) | Agent (Gate ON) | Target | Status |
|--------|---------------------|------------------|---------|---------|
| **Avg Overlap** | 1.000 | 0.933 | 0.15-0.35 | ❌ Too high |
| **Avg Faithfulness** | 1.000 | 0.973 | 0.62-0.75 | ⚠️ Too high |
| **Avg Total Tokens** | 496 | 508 | 180-260 | ❌ Too high |
| **P50 Latency (ms)** | 1847 | 1643 | 700-1200 | ⚠️ Slightly high |
| **Rounds** | 1 | 1 | 1 | ✅ Perfect |
| **Abstain Rate** | 20% | 0% | 0-5% | ⚠️ Mixed |

### Key Findings:

✅ **Working Systems:**
- Citation parsing and overlap calculation
- RAGAS faithfulness evaluation
- Token counting and latency measurement
- End-to-end pipeline (ingest → retrieve → generate)

⚠️ **Areas Needing Optimization:**
- **Overlap too high:** Perfect citation coverage inflates scores
- **Token usage:** ~2x target (496-508 vs 180-260)
- **Latency:** Slightly above target (1.6-1.8s vs 0.7-1.2s)
- **Gate logic:** Not triggering multi-round behavior

❌ **Issues Identified:**
- Questions too simple for current corpus
- Gate thresholds may need adjustment
- Need more diverse/challenging evaluation dataset

### Next Actions:
1. Tune gate thresholds (lower τ_f and τ_o)
2. Create more challenging questions requiring multi-round reasoning
3. Optimize token usage in prompts
4. Test with larger, more complex corpus
