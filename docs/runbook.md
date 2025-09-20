# Agentic RAG – Anchor-Aware Improvements (Quick Runbook)

This runbook summarizes how to validate the anchor-aware changes that reduce abstains on answerables and eliminate “faithfully wrong” answers.

## Defaults

- Retrieval pool: `RETRIEVAL_POOL_K=50` (config), rerank/MMR as before
- Anchor bonus in fusion: `ANCHOR_BONUS=0.07`
- HyDE: disabled for factoids (dates/numbers/single-entity)
- One-shot anchor-constrained retrieval: enabled when factoid + missing anchors + `tokens_left >= 300`

## Validate on the 30-Item Subset

1) Run agent (best-of scoring OFF by default):

```
python -m agentic_rag.eval.runner \
  --dataset data/crag_questions.jsonl \
  --system agent --gate-on --judge-policy always
```

2) Inspect logs in `logs/<ts>_agent.jsonl` and `logs/<ts>_agent_summary.csv`.

3) Per-question reasons (Node):

```
node scripts/analyze_per_question.js logs/<ts>_agent.jsonl data/crag_questions.jsonl
```

## What to Look For

- Anchor-constrained retrievals firing only once per question when needed:
  - `used_anchor_constrained_search=true`
  - `required_anchors`, `missing_anchors`, `anchor_coverage`, `fail_time/unit/event`
  - `used_hyde`, `used_hybrid`, `used_rerank`, `used_mmr` in step logs

- Answer-level outcomes (not just metrics):
  - Nash 3PA → `4`
  - MSFT ex-div Q1-2024 → `Feb 14, 2024`
  - 2004 Best Animated Feature → `Finding Nemo`
  - USO vs AO → USO/Nadal (or abstain)
  - Pixar average gross → abstain if unit/window is ambiguous in CTX
  - Southern Africa list → either complete (if CTX supports) or abstain (no partial)

## Optional Scoring Toggle

Use short-answer scoring (dates/numbers/entities) with:

```
python -m agentic_rag.eval.runner ... --use-final-short
```

The CSV will include both `em_full/f1_full` and `em_short/f1_short`, and a `pred_source` column.
