# Hallucination Fixes: Gate + Retrieval

## Summary of All Changes

### Phase 1: BAUG Gate Fixes (COMPLETED - No Impact)
Gate fixes were implemented but showed **zero improvement** (hallucinations still 25.5%).
**Root cause:** Gate logic was fine - the problem is retrieval finding wrong documents.

### Phase 2: Retrieval Quality Fixes (CURRENT)

**New approach:** Instead of gate logic, fix the retrieval to find RIGHT documents.

**Changes made:**
1. **Favor BM25 exact matching** - HYBRID_ALPHA: 0.6→0.4 (60% BM25 vs 40% vector)
2. **Boost anchor-bearing docs** - ANCHOR_BONUS: 0.07→0.20 (3x stronger)
3. **Increase diversity** - MMR_LAMBDA: 0.45→0.2 (escape wrong-doc clusters)

See `RETRIEVAL_IMPROVEMENTS.md` for full technical details.

---

## Phase 1 Details (Gate Fixes - Ineffective)

### Issue Identified
Analysis of 50-sample run showed:
- **30% hallucination rate** (15/50)
- **33% of hallucinations had PERFECT coverage (≥0.6)** but still wrong answers
- Root cause: System was finding partially relevant docs (right topic, wrong specifics)

### Fixes Implemented

#### 1. ✅ Prevent High-Overlap + Low-Coverage Stops (Already Done)
**File:** `src/agentic_rag/gate/adapter.py`
**Lines:** 125-134

Changed BAUG to NOT stop when overlap is high but coverage is low.
- Before: Stopped if overlap ≥ 0.85 (regardless of coverage)
- After: Requires BOTH high overlap AND adequate coverage

#### 2. ✅ Add Conflict Detection (NEW)
**File:** `src/agentic_rag/gate/adapter.py`
**Lines:** 117-122

Added check for conflicting evidence before allowing stop:
```python
if sig.conflict_risk > 0.4 and sig.round_idx < 2:
    reasons.append("conflicting_evidence")
    return GateAction.RETRIEVE_MORE
```

Prevents stopping when documents contradict each other (e.g., different numbers for same metric).

#### 3. ✅ Raised Default Thresholds (NEW)
**File:** `src/agentic_rag/config.py`

**Changed:**
- `BAUG_STOP_COVERAGE_MIN`: 0.35 → **0.55** (require more anchors)
- `BAUG_HIGH_OVERLAP_TAU`: 0.7 → **0.85** (stricter high-overlap threshold)
- `NEW_HITS_EPS`: 0.15 → **0.10** (allow more rounds before stopping)
- **Added:** `BAUG_CONFLICT_THRESHOLD: 0.4` (new parameter)

## Expected Impact

| Metric | Before Fix | Expected After |
|--------|-----------|----------------|
| Hallucinations | 30% (15/50) | **~18-22%** |
| Avg Tokens | 2002 | ~2100-2200 |
| Avg Rounds | 1.7 | ~1.9-2.1 |
| Composite | 50.43 | ~52-54 |

## Test Commands

### Quick Test (n=50)
```bash
python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3 --override "RETRIEVAL_K=24 PROBE_FACTOR=4 USE_RERANK=True RERANK_CANDIDATE_K=80 RERANK_KEEP_K=18 MMR_LAMBDA=0.2 HYBRID_ALPHA=0.5 ANCHOR_BONUS=0.12 RESERVE_ANCHOR_SLOTS=3 PACK_RESERVE_ON_LOW_COVERAGE=True GATE_RETRIEVAL_K_BONUS=10"
```

### Full Evaluation (n=200) - RECOMMENDED
```bash
python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 200 --judge-policy gray_zone --max-rounds 3 --override "RETRIEVAL_K=24 PROBE_FACTOR=4 USE_RERANK=True RERANK_CANDIDATE_K=80 RERANK_KEEP_K=18 MMR_LAMBDA=0.2 HYBRID_ALPHA=0.5 ANCHOR_BONUS=0.12 RESERVE_ANCHOR_SLOTS=3 PACK_RESERVE_ON_LOW_COVERAGE=True GATE_RETRIEVAL_K_BONUS=10"
```

**Note:** Using default config values now, no need to override BAUG thresholds.

### Optional: Even Stricter (if hallucinations still high)
If hallucinations are still above 20%, try:
```bash
--override "... BAUG_STOP_COVERAGE_MIN=0.65 NEW_HITS_EPS=0.05 ..."
```

## What Changed Under the Hood

### Before (Old BAUG Logic):
```python
if overlap >= 0.85:
    return STOP  # ❌ Could stop with low coverage!
```

### After (New BAUG Logic):
```python
# Check 1: Conflicting evidence?
if conflict_risk > 0.4:
    return RETRIEVE_MORE  # ✅ Get more docs to resolve conflict

# Check 2: High overlap?
if overlap >= 0.85 and coverage >= 0.55:
    return STOP  # ✅ Only stop with BOTH high overlap AND coverage
elif overlap >= 0.85 and coverage < 0.55:
    return RETRIEVE_MORE  # ✅ Keep searching for better docs
```

## Monitoring

After running full n=200 evaluation, check these metrics:

1. **Hallucination rate** - Target: < 20%
2. **Abstain rate** - Should stay ~22-26% (not increase too much)
3. **Token usage** - Expected: ~2000-2200 (slight increase is OK)
4. **Composite score** - Target: > 52

If hallucinations drop below 20% without abstain rate shooting up, the fix worked!

## Next Steps if Issues Persist

If hallucination rate is still > 20% after full evaluation:

1. **Analyze remaining hallucinations** using `HALLUCINATION_FIX_RECOMMENDATIONS.md`
2. **Consider scope-aware coverage** (advanced fix - check if entities match question scope)
3. **Strengthen validators** (add numerical consistency checks)
4. **Cross-document verification** (require multiple docs to agree on numerical facts)

## Files Modified

1. `src/agentic_rag/gate/adapter.py` - Added conflict detection, fixed high-overlap logic
2. `src/agentic_rag/config.py` - Raised default thresholds
