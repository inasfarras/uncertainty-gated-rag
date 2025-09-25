# Anchor-Style Multi-Agent + BAUG: Short Report Template

## Setup
- Model: `${LLM_MODEL}` temp=0, top_p=0, max_output=${MAX_OUTPUT_TOKENS}
- Context cap: ${MAX_CONTEXT_TOKENS}
- Index: `${FAISS_INDEX_PATH}` (hybrid=${USE_HYBRID_SEARCH}, mmr=${MMR_LAMBDA}, rerank=${USE_RERANK})
- Judge policy: ${JUDGE_POLICY} (≤${JUDGE_MAX_CALLS_PER_Q} call/Q)
- BAUG handler: `${BAUG_HANDLER:-fallback}`

## Systems Compared
- Vanilla (baseline)
- Single-Agent (agent)
- Anchor-Style Multi-Agent (anchor)

## Metrics (CRAG subset)
- Evidence: Overlap (sentence-level, τ_sim=${OVERLAP_SIM_TAU}), Faithfulness (fallback / RAGAS subset)
- Gold: EM/F1, Wrong-on-Answerable
- Cost: tokens/q, p50 latency

## Results (N=${N})
| System | Avg Overlap | Avg EM | Avg F1 | Abstain% | Wrong-on-Ans% | Tokens/q | p50 Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline |  |  |  |  |  |  |  |
| agent |  |  |  |  |  |  |  |
| anchor |  |  |  |  |  |  |  |

## BAUG Decision Breakdown (anchor)
- STOP / MORE / REFLECT / ABSTAIN rates: ...
- Avg new_hits_ratio: ... ; Δoverlap per round: ...
- Anchor coverage (mean): ... ; conflict risk (mean): ...

## Ablations
- Without BAUG (anchor but BAUG off): ...
- Without pruning/early-stop (no-new-hits/plateau off): ...
- Retrieval modes (vector vs hybrid vs graph if available): ...

## Qualitative
- Examples with stronger anchor grounding and lower “faithfully wrong” outcomes.
- Failure modes (anchor sparsity, mismatch flags) and BAUG actions.

## Notes
- Deterministic settings maintained; REFLECT rarely invoked.
- Logs in `logs/` contain per-round telemetry for reproducibility.
