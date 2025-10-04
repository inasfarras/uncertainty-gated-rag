"""Evaluation rubrics and prompt templates for GPT-based judging."""

from __future__ import annotations

RUBRIC_GOLD_AWARE = """You are an impartial grader that compares the AGENT_ANSWER against the GOLD_ANSWER and the cited EVIDENCE. Citations use the format [CIT:<evidence_id>] and must match the `id` field inside EVIDENCE to count as grounded. Focus on semantic agreement: treat paraphrases and correct, evidence-backed elaborations as valid. Only penalize when information contradicts the gold/evidence or omits critical facts.

Score each dimension from 0 to 5 integers. Provide brief rationales. Apply the rules strictly.

Dimensions:
- correctness (0 incorrect or contradicts gold/evidence, 5 factually consistent with the gold answer even if phrasing differs).
- citation_support (0 no citations or incorrect grounding, 5 all claims that need support cite matching evidence).
- completeness (0 misses the main gold facts, 5 covers every critical fact required by the gold answer).
- conciseness (0 rambling or speculative, 5 concise and focused).
- hallucination_risk (0 introduces unsupported speculation, 5 additions are justified or clearly scoped).

Rules:
- Jika AGENT_ANSWER berisi "I don't know" padahal GOLD_ANSWER jelas, set correctness <= 1.
- Terima detail tambahan selama konsisten dengan GOLD_ANSWER dan didukung EVIDENCE; hanya kurangi skor jika bertentangan.
- Jika tidak ada sitasi relevan [CIT:id] yang cocok dengan EVIDENCE, set citation_support <= 1.
- Jika jawaban bertentangan dengan EVIDENCE atau GOLD_ANSWER, set correctness = 0 dan citation_support <= 1.

Overall score (0-100) = 8*correctness + 5*citation_support + 3*completeness + 2*conciseness + 2*hallucination_risk.

Kembalikan JSON ketat dengan skema:
{
  "scores": { ... },
  "rationales": { ... },
  "flags": {
    "missing_citation": bool,
    "contradiction_with_evidence": bool,
    "gold_mismatch_but_supported": bool
  },
  "used_citations": [ints]
}

Jangan keluarkan teks lain di luar objek JSON."""

PROMPT_TEMPLATE = """### Task
Evaluate the AGENT_ANSWER with access to GOLD_ANSWER and supporting EVIDENCE. Decide whether the agent's answer is factually equivalent to the gold answer even when the wording differs or it adds correct, supported detail.

### Guidance
- Compare key facts (entities, dates, numbers, relations) for semantic agreement.
- Accept paraphrases and extra correct context if they stay consistent with the gold answer.
- Reject or down-score only when the answer contradicts the gold/evidence or omits critical facts.
- Verify citations: each [CIT:<id>] must reference matching evidence `id` text that truly supports the claim.
- Keep rationales concise and reference evidence ids when useful.

### QUESTION
{QUESTION}

### GOLD_ANSWER
{GOLD_ANSWER}

### AGENT_ANSWER
{AGENT_ANSWER}

### EVIDENCE
{EVIDENCE}

Apply the rubric and return JSON only."""
