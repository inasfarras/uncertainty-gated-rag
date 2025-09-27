# Agentic RAG Implementation Summary (Updated)
Date: September 27, 2025
Status: COMPLETE

## Overview
This update introduces a Hybrid Question Interpreter (rules-first, LLM-fallback) and wires its intent signals through the Anchor → Retrieval → Supervisor → BAUG stack. Goals: resolve a majority of queries via rules, cut wrong-on-answerable, keep tokens ≤1.2× baseline, and improve overlap and short-answer F1.

## What’s New

- Hybrid Intent Module (`src/agentic_rag/intent/`)
  - `types.py` Intent schema.
  - `rules.py` regex/heuristics for years, season ranges (e.g., 2016–17), units (per game, %, USD), awards/tournament cues and aliases, task_type inference, completeness/confidence scoring.
  - `llm.py` deterministic JSON-only fallback (temperature=0, top_p=0, max_tokens≤256) with robust handling of invalid JSON.
  - `interpreter.py` rules-first, budget-aware LLM fallback, conservative merge policy.

- Anchors informed by Intent (`anchors/predictor.py`)
  - `propose_anchors(intent)` prioritizes `slots` and `core_entities`; dedup + score deterministically.

- Supervisor Orchestrator (`supervisor/orchestrator.py`)
  - `intent = interpret(question, llm_budget_ok=tokens_left>=300)` then `anchors = propose_anchors(intent)`.
  - Passes `intent_confidence`, `slot_completeness`, `source_of_intent` and validator outputs to BAUG and telemetry.
  - Lightweight validators before STOP: (1) awards/tournament require {event+year+category/division} in cited text; (2) numeric/time require {unit+time window} in cited text.

- BAUG Adapter (`gate/adapter.py`)
  - Typed `Signals`; built‑in rule policy:
    - slot_completeness<0.6 and budget<300 → ABSTAIN
    - validators OK and overlap≥τ → STOP
    - new_hits_ratio<ε → STOP_LOW_GAIN
    - else RETRIEVE_MORE (≤1 REFLECT allowed)

- Telemetry + Tests
  - Telemetry: intent_confidence, slot_completeness, source_of_intent, validators_passed, llm_calls, stop_reason.
  - Tests under `tests/intent/` exercise rules, LLM fallback/merge, and signals flow.

## Determinism & Budget
- LLM gen: temperature=0, top_p=0; max output ≤160 tokens; packed context cap ≈1000 tokens.
- Interpreter LLM: max_tokens≤256; rules-only preferred; at most one REFLECT round.

## Results (CRAG, N=30)
- Baseline: F1=0.133, EM=0.000, Overlap=0.482, Faith=0.557, Abstain=40%, Tokens≈1102, P50≈1561ms.
- Anchor + Hybrid Intent (BAUG ON): F1=0.269, EM=0.100, Overlap=0.589, Faith=0.643, Abstain=33%, Tokens≈1132, P50≈1530ms.
- Anchor (Gate OFF): Same accuracy; P50≈1656ms. BAUG primarily improves termination (slightly faster) without extra tokens.

## How to Run
1) Rebuild FAISS/BM25: `python -X utf8 -m agentic_rag.ingest.ingest --input data/crag_corpus_html --out artifacts/crag_faiss --backend openai`
2) Anchor (BAUG ON): `python -X utf8 -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system anchor --profile anchor_balanced --n 30`
3) Anchor (Gate OFF): `python -X utf8 -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system anchor --gate-off --n 30 --override "USE_HYBRID_SEARCH=False" --override "RETRIEVAL_K=8" --override "PROBE_FACTOR=1" --override "RETRIEVAL_POOL_K=24" --override "MAX_CONTEXT_TOKENS=1100" --override "MAX_OUTPUT_TOKENS=90" --override "MMR_LAMBDA=0.0" --override "MAX_WORKERS=6"`

## Files Touched (Key)
- New: `src/agentic_rag/intent/*`, `tests/intent/*`.
- Updated: `anchors/predictor.py`, `anchors/validators.py`, `supervisor/orchestrator.py`, `gate/adapter.py`, telemetry recorder.

## Notes
- LLM JSON robustness and rule-only early exits are validated in tests.
- ≥60% rule‑only on small subsets; share varies by domain in full runs.
