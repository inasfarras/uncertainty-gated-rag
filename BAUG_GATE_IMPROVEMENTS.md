# BAUG Gate Improvements - Fix High Abstention Rate

## Problem Identified

**55% Abstention Rate** on 20-question test with anchor system.

### Root Cause
BAUG gate was **immediately stopping** without attempting to generate an answer when `new_hits_ratio < 0.10`, even when the system had already retrieved relevant context.

Example failure:
- **Question**: "which movie won the oscar best visual effects in 2021?"
- **Retrieved context**: "Tenet, scoring a Best Visual Effects Oscar win"
- **System answer**: "I don't know"
- **Stop reason**: `STOP_LOW_GAIN` due to `low_new_hits`

## Changes Made

### 1. **Fixed BAUG Gate Logic** (`src/agentic_rag/gate/adapter.py` lines 146-158)

**Before:**
```python
if sig.round_idx > 0 and sig.new_hits_ratio < settings.NEW_HITS_EPS:
    reasons.append("low_new_hits")
    return "STOP_LOW_GAIN", reasons  # ← Aborts immediately!
```

**After:**
```python
if sig.round_idx > 0 and sig.new_hits_ratio < settings.NEW_HITS_EPS:
    reasons.append("low_new_hits")
    # If we already have good overlap and coverage, stop and generate answer
    if high_overlap and coverage_ok:
        reasons.append("sufficient_context")
        return GateAction.STOP, reasons
    # If overlap is decent but coverage is low, try one more round
    if sig.overlap_est >= (settings.OVERLAP_TAU * 0.7):
        reasons.append("partial_overlap")
        return GateAction.RETRIEVE_MORE, reasons
    # Otherwise, stop with low gain
    return "STOP_LOW_GAIN", reasons
```

**Impact**: System now attempts to answer when it has sufficient context, even if not finding new documents.

### 2. **Relaxed Thresholds** (`src/agentic_rag/config.py`)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `NEW_HITS_EPS` | 0.10 | 0.05 | Less aggressive early stopping |
| `FINE_FILTER_TAU` | 0.15 | 0.10 | Allow lower-quality docs through filtering |
| `ANCHOR_PLATEAU_EPS` | 0.05 | 0.03 | Continue retrieval with smaller gains |
| `MAX_CONTEXT_TOKENS` | 1000 | 1500 | Pack more context per query |
| `RERANK_KEEP_K` | 15 | 20 | Keep more candidates after reranking |
| `RETRIEVAL_POOL_K` | 24 | 32 | Larger initial candidate pool |

### 3. **Maintained CUDA Optimizations**
- Reranker on CUDA with FP16 (2x speedup)
- Embedder on CUDA with batching (3-5x speedup)
- No performance regression

## Expected Improvements

### Before Fix:
- **Abstain Rate**: 55% (11/20 questions)
- **Mean Score**: 27/100
- **Hallucinations**: 5/20 (25%)
- **Perfect Matches**: 1/20 (5%)

### Target After Fix:
- **Abstain Rate**: 20-30% (more answers attempted)
- **Mean Score**: 35-45/100 (better quality)
- **Hallucinations**: <20% (maintained)
- **Perfect Matches**: 10-15% (2-3x improvement)

## Testing

### Run Comparison Test:
```powershell
# Test with improvements
python -m agentic_rag.eval.runner --dataset test_questions.jsonl --n 20 --system anchor --gate-on

# Compare results
python scripts/run_eval_with_judge.py --dataset test_questions.jsonl --system anchor --n 20 --judge-require-citation false
```

### Key Metrics to Watch:
1. **Abstain Rate** - Should decrease from 55% to ~25%
2. **Avg Overall Score** - Should increase from 27 to 35+
3. **Safe IDK** - Should decrease (fewer unnecessary abstentions)
4. **Hallucinations** - Should remain stable or decrease (not increase!)

## Validation Criteria

✅ **Success**:
- Abstain rate < 35%
- Mean overall score > 30
- Hallucination rate < 30%
- At least 2 perfect matches

⚠️ **Needs Tuning**:
- Abstain rate 35-45%
- Hallucinations 30-40%

❌ **Regression**:
- Abstain rate > 50% (no improvement)
- Hallucinations > 40% (quality degraded)

## Rollback Plan

If results regress:
```bash
git diff src/agentic_rag/gate/adapter.py
git diff src/agentic_rag/config.py
git checkout HEAD -- src/agentic_rag/gate/adapter.py src/agentic_rag/config.py
```

## Related Issues

- High abstention on factoid questions with clear answers
- "Tenet" Oscar question failure (answer in context, said IDK)
- Numeric questions stopping too early
- Context quality filtering too aggressive
