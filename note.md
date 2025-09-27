# Budget-Aware Uncertainty Gate (BAUG) — Anchor-Centric Agentic RAG Plan

*A self-contained brief you can share with your advisor **and** paste into Cursor/Codex to guide implementation.*

---

## 1) Thesis in one page

### Problem

* **Vanilla RAG** can be **inconsistent** with retrieved evidence (answers drift beyond the context).
* **Agentic RAG loops** (retrieve → generate → reflect → retrieve…) reduce hallucinations, but can **inflate cost** (tokens/latency) without consistent gains.

### Gap

We lack a **simple, external, metric-driven controller** that is:

1. **Context-aware** beyond model confidence (does the answer actually cite/support from context?),
2. **Budget-aware** (token/latency caps + early exits),
3. Supports multiple **actions** (STOP / RETRIEVE\_MORE / REFLECT / ABSTAIN), not just retrieve/not.

### Approach: **Budget-Aware Uncertainty Gate (BAUG)**

* **Signals:**

  * **Support Overlap**: fraction of answer sentences supported by retrieved chunks via exact citation + semantic similarity ≥ τ\_sim.
  * **Faithfulness**: alignment between answer claims and retrieved evidence (fallback formula and/or lightweight judge).
  * **Budget**: context/output caps; **no-new-hits** and **stagnation** short-circuits.
* **Policy:** Choose **STOP / RETRIEVE\_MORE / REFLECT / ABSTAIN** within budget and log decisions.
* **Evaluation:** Pareto of (evidence consistency) vs (tokens/latency); plus gold-facing EM/F1.

### Why anchor-centric agentic RAG?

Recent work shows **critique/reflect loops** can help (Self-RAG) and **self-grading/corrective steps** (CRAG) are plug-and-play; **graph-guided** retrieval (GraphRAG) improves precision on structured corpora. Your BAUG can sit **on top** as the controller for cost-quality trade-offs. (Self-RAG: critique/reflect loop; CRAG: corrective self-grading; GraphRAG: graph-structured retrieval and hierarchical summaries.) ([ICLR Proceedings][1]) ([OpenReview][2]) ([Microsoft][3])

---

## 2) What the latest agentic-RAG literature implies for your design

* **Self-RAG** introduces reflection tokens + a scaffolding loop to *retrieve, generate, and critique*; integrate REFLECT only when BAUG asks for it. ([ICLR Proceedings][1])
* **Corrective-RAG (CRAG)** separates **evaluate→correct** steps and is explicitly *plug-and-play* with any RAG pipeline—perfect “decision points” to insert BAUG. ([OpenReview][4])
* **GraphRAG** builds a knowledge graph + community hierarchy, then uses graph summaries for retrieval; keep a **vector/BM25 fallback** and allow BAUG to select the cheaper path under budget. ([Microsoft][3])
* **Agentic RAG surveys (2025)** synthesize patterns (planning, reflection, tool use, multi-agent orchestration). Use them to justify your modular, **multi-agent** plan (predictor ↔ retrievers ↔ supervisor) with an external controller (BAUG). ([arXiv][5])

---

## 3) Migration target: **Anchor-Centric, Multi-Agent RAG** (KG-aware when available)

> **Intent:** You keep your current retrieval + evaluation stack, but wrap it with a thin **multi-agent layer** that proposes and tests **anchors** (entities/years/units/events/tournaments) and **terminates early** when paths aren’t promising—while your **BAUG** remains the final decision-maker.

### Agent roles & messages

* **Anchor-Predictor**

  * Input: question (+ optional first-pass hits)
  * Output: ranked `{anchor, confidence}` list (e.g., “U.S. Open 2017”, “Best Animated Feature (2004)”, “per game”, “Q1 2024”).
* **Retriever Workers (parallel)**

  * For each anchor, run **multi-hop** exploration:

    * **Graph** mode (if KG or hyperlink graph exists): follow edges/links two hops, then collect passages.
    * **Vector/BM25** mode: retrieve passages emphasizing anchor tokens.
  * **Pruning**: rough (lexical/topical) → fine (semantic entailment/overlap); terminate paths below τ or with low novelty (**new\_hits\_ratio < ε**). (This mirrors CRAG’s “selectively focus/filter” and GraphRAG’s structured constraints.) ([OpenReview][4]) ([Microsoft][3])
* **Supervisor**

  * Merge and score paths by **anchor-coverage**, **support overlap**, **conflict risk**, and **expected utility per token**.
  * Calls **BAUG** with signals (`overlap_est`, `faith_est`, `new_hits_ratio`, `anchor_coverage`, `budget_left`) to choose **STOP / RETRIEVE\_MORE / REFLECT / ABSTAIN**.
  * If BAUG says REFLECT: run a **Self-RAG-style critique-revise** *once*; otherwise stop or abstain. ([ICLR Proceedings][1])

### Determinism + budget

* **Single generation by default** (temp=0, top\_p=0, max\_output≈160).
* **Context cap ≈ 1k tokens**, strict packing.
* **Judge policy**: gray-zone (≤1 call/Q) if used at all.
* Always log **tokens\_by\_stage** (predictor / retrieval / generation) and **latency p50**.

---

## 4) Modules & interfaces (drop-in to your repo)

> Keep your current names if different; the key is the **interfaces** and **telemetry**.

### `anchors/predictor.py`

* `propose_anchors(question, top_m) -> list[{"text": str, "score": float}]`
  Heuristics: extract **years, quarters/months, units** (“per game”, “ex-dividend”), **events/categories** (“U.S. Open”, “Best Animated Feature”), and **named entities**; optionally re-rank with a small cross-encoder.

### `retrieval/agent.py`

* `explore(anchor, question, hop_budget) -> list[Path]`

  * Hop 0: retrieve K passages filtered by anchor tokens; prune **rough** (keyword/topical).
  * Hop 1: expand via KG edges or hyperlinks; prune **fine** (semantic/entailment vs question).
  * Track `new_hits_ratio`, `path_score`, `terminated_by` (low score/budget/time).

### `supervisor/orchestrator.py`

* `compose(paths) -> EvidenceBundle` (packed chunks + per-anchor stats).
* `decide(bundle, budget) -> Action` uses **BAUG**; return **STOP / MORE / REFLECT / ABSTAIN** with a reason code.

### `gate/adapter.py` (your BAUG in the loop)

* `collect_signals(bundle, usage) -> Signals`
* `decide(signals) -> Action` delegates to BAUG; emit `stop_reason` (e.g., `STOP_OVERLAP_OK`, `STOP_NO_NEW_HITS`, `ABSTAIN_MISSING_ANCHORS`, `STOP_LOW_BUDGET`).

### `logging/telemetry.py`

* JSONL per round:
  `anchors_proposed`, `anchors_selected`, `path_count`, `pruned_count`, `new_hits_ratio`,
  `anchor_coverage`, `overlap_est`, `faith_est`, `tokens_left/used`, `stop_reason`, `latency_ms`.

---

## 5) Precision features that directly fix your error modes

1. **Anchor-boost** in fusion
   After combining vector/BM25 scores, add a small **bonus** for passages containing required anchors (coverage-scaled). This reduces *“faithful-to-wrong-anchor”* answers (e.g., AO vs USO; 2004 ceremony vs film release year). (Use GraphRAG/CRAG insights to justify anchor constraints + selective focus.) ([Microsoft][3])

2. **One-shot anchor-constrained retrieval (factoids only)**
   Before **ABSTAIN** on dates/numbers/entities: if `tokens_left ≥ 300` and anchors in the **question** are **missing** in cited text, run **one** extra retrieval by **appending missing anchors** (no LLM rewrite). If it yields new hits, regenerate **once**, else abstain.

3. **Type-specific validators** in supervisor (cheap)

   * **Awards/tournaments**: ceremony-year + category/tournament must be present in cited text to STOP.
   * **Numerics/time windows**: require unit/time anchors (“per game”, “Q1 2024”, “domestic/worldwide”) to STOP.
     These enforce **“sufficiency of the context”**—a theme in CRAG’s corrective filtering—before BAUG lets you stop. ([OpenReview][4])

4. **Short-answer finalizer (scoring only)**
   Extract the minimal span for EM/F1 (date/number/title). Keep the full cited answer for humans; use the **short** form for scoring so EM/F1 reflects factuality, not verbosity.

---

## 6) Metrics (what to report)

* **Evidence-facing:** Overlap, Faithfulness (fallback or judge).
* **Gold-facing:** EM/F1 (short), **Wrong-on-Answerable** (not IDK and F1==0).
* **Contract checks:** IDK+Cit violations (should be 0).
* **Cost:** Tokens/Q, p50 ms.
* **CRAG-style:** Accuracy / Hallucination / Missing / Score (if you mirror their judge).
  This aligns with CRAG’s *corrective* notion (evaluate sufficiency before answering) and with agentic surveys’ emphasis on **reasoning + tool use** trade-offs. ([DataCamp][6])

---

## 7) Runbook (repro in three commands)

```bash
# 1) Prepare CRAG subset & index
python scripts/prepare_crag.py --split train --static-only --n 100
python -m agentic_rag.ingest.ingest --input data/crag_corpus --out artifacts/crag_faiss --backend openai

# 2) Baselines
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --n 100

# 3) Anchor-centric Agent + BAUG (lean, single-round, precision-first)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 100 \
  --override MAX_ROUNDS=1 USE_RERANKER=False USE_HYDE=False \
  RETRIEVAL_POOL_K=24 RETRIEVAL_K=8 MMR_LAMBDA=0.45 \
  OVERLAP_TAU=0.42 MAX_CONTEXT_TOKENS=1000 MAX_OUTPUT_TOKENS=160 \
  ANCHOR_BONUS=0.07 FACTOID_ONE_SHOT_RETRIEVAL=True FACTOID_MIN_TOKENS_LEFT=300 \
  JUDGE_POLICY=gray_zone JUDGE_MAX_CALLS_PER_Q=1 USE_FACTOID_FINALIZER=True
```

---

## 8) Two-week milestones

**Week 1 (scaffolding & signals)**

* D1–D2: Add `anchors/predictor.py`; emit top-m anchors + confidence; unit tests for extraction.
* D3–D4: Implement `retrieval/agent.py` with rough→fine pruning; log `terminated_by`, `new_hits_ratio`.
* D5: Wire **anchor-boost** into fusion; **deterministic** gen; strict context/output caps.
* D6–D7: Add `supervisor/orchestrator.py` + `gate/adapter.py` → BAUG call; JSONL telemetry fields.

**Week 2 (precision & eval)**

* D8: One-shot anchor-constrained retrieval (factoids).
* D9: Type-validators (awards/tournaments; numeric/unit/time windows).
* D10: Short-answer finalizer (EM/F1 only).
* D11–D12: CRAG subset run (N=100); A/B vs vanilla; **Pareto plot** (Overlap vs Tokens).
* D13–D14: Error analysis (wrong-on-answerable, partial lists) and a 1-page update.

---

## 9) Risks & mitigations

| Risk                         | Impact                | Mitigation                                                                                         |
| ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| KG not available             | Graph hops impossible | Use hyperlink/section hops + anchor-boost; keep vector/BM25 fallback. ([Microsoft][3])             |
| Anchor sparsity or ambiguity | Too many false paths  | Confidence threshold + rough→fine pruning; BAUG ABSTAIN if coverage low. ([OpenReview][4])         |
| Cost growth from extras      | Token budget blown    | Single generation by default; judge gray-zone (≤1/Q); early termination; strict caps. ([arXiv][5]) |
| EM/F1 penalized by verbosity | Understated quality   | Short-answer finalizer for scoring only.                                                           |

---

## 10) Ready-to-paste **Cursor/Codex prompt**

> **Role:** Senior RAG architect. Create a **migration plan + code skeleton** to refactor my agentic RAG into an **anchor-centric, multi-agent** system with a **Budget-Aware Uncertainty Gate (BAUG)** as the final decision policy.
> **Do not** depend on my current file paths—propose modules & interfaces as below; keep generation deterministic (temp=0, top\_p=0) and budget-aware.

**Deliverables**

1. **System blueprint** (ASCII diagram) with: `Anchor-Predictor → Retriever-Workers (rough→fine) → Supervisor (+BAUG)`; early termination rules and evidence composition.
2. **Module stubs & interfaces** (`anchors/predictor.py`, `retrieval/agent.py`, `supervisor/orchestrator.py`, `gate/adapter.py`, `logging/telemetry.py`, `agent/finalize.py`) with function signatures, input/output dataclasses, and TODOs.
3. **Anchor-boost** in retrieval fusion and a **one-shot anchor-constrained retrieval** before ABSTAIN on factoids.
4. **Type validators** (awards/tournaments; numerics/units/time windows) that must pass before STOP; otherwise try the one-shot search then ABSTAIN.
5. **Scoring finalizer** to extract short spans (date/number/title) for EM/F1 **only**—do not change displayed answers.
6. **Telemetry** fields listed above; a tiny **runbook** with three commands for N=100 CRAG.
7. **Acceptance criteria:**

   * Tokens/Q ≤ 1.2× vanilla;
   * Overlap & F1(short) ≥ vanilla;
   * Wrong-on-Answerable ↓;
   * IDK+Cit = 0.

**Ground your choices** in:

* **Self-RAG** (critique/reflect loop) ([ICLR Proceedings][1])
* **CRAG** (corrective filtering; plug-and-play) ([OpenReview][4])
* **GraphRAG** (graph-guided retrieval; community summaries) ([Microsoft][3])
* **Agentic RAG surveys (2025)** (planning, orchestration, uncertainty) ([arXiv][5])

---

## 11) References (for your report)

* **Self-RAG (ICLR 2024)** — learn to retrieve, generate, **critique**; core reference for REFLECT hooks. ([ICLR Proceedings][1])
* **Corrective-RAG (OpenReview)** — **self-grading** + corrective steps; *plug-and-play* with RAG pipelines (paper + PDF). ([OpenReview][2])
* **GraphRAG (Microsoft)** — build a knowledge graph + **hierarchical summaries** to guide retrieval (docs + blog). ([Microsoft][3])
* **Agentic RAG survey (2025)** — taxonomy of **agentic** RAG (planning, tool use, multi-agent); use to justify your architecture. ([arXiv][5])
* **AutoRAG (2024)** — framework for auto-tuning RAG modules; useful for your retrieval ablations. ([arXiv][7])
* **PaperQA2 (2024)** — high-accuracy, tool-using **RAG agent** for scientific QA; a strong practical baseline to compare against if needed. ([GitHub][8])

---

### Final note

This plan lets you **stop tuning the whole agent**, and instead focus on the **Budget-Aware Uncertainty Gate**—the core of your thesis—while standing on published, open frameworks (Self-RAG / CRAG / GraphRAG). Your evaluation stays centered on **evidence consistency vs. budget**, which is exactly the contribution you’re claiming.

[1]: https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf?utm_source=chatgpt.com "SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND ..."
[2]: https://openreview.net/forum?id=JnWJbrnaUE&utm_source=chatgpt.com "Corrective Retrieval Augmented Generation"
[3]: https://microsoft.github.io/graphrag/?utm_source=chatgpt.com "Welcome - GraphRAG"
[4]: https://openreview.net/pdf?id=JnWJbrnaUE&utm_source=chatgpt.com "CORRECTIVE RETRIEVAL AUGMENTED GENERATION"
[5]: https://arxiv.org/abs/2501.09136?utm_source=chatgpt.com "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"
[6]: https://www.datacamp.com/tutorial/corrective-rag-crag?utm_source=chatgpt.com "Corrective RAG (CRAG) Implementation With LangGraph"
[7]: https://arxiv.org/abs/2410.20878?utm_source=chatgpt.com "AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline"
[8]: https://github.com/Future-House/paper-qa?utm_source=chatgpt.com "Future-House/paper-qa: High accuracy RAG for answering ..."
