<!-- 76978411-5a29-4a0d-a128-aaddc90e108b 888e7d5b-7d88-4167-905c-b4cb47df71c5 -->
# Fix BAUG Latency and Hallucination Issues

## Problem Summary

From the N=5 experiment, two critical issues were identified:

1. **Latency Measurement Bug**: BAUG configurations show `p50_latency_ms: 0` and empty `latencies: []` arrays
2. **Temporal Hallucination**: Question 5 (Dow Jones best performer "today") - BAUG answered with Walmart from "Tuesday" instead of 3M at 1.38%

## Root Cause Analysis

### Latency Issue

- Line 1047 in `orchestrator.py`: `"p50_latency_ms": int(np.median(latencies)) if latencies else 0`
- Line 169 in `orchestrator.py`: `latencies: list[int] = []` initialized but never appended to
- Line 696: `latency_ms = tmr()` captures per-round latency but is never added to the `latencies` list
- The latency is logged per-round but not accumulated in the summary

### Temporal Hallucination Issue

- No temporal filtering exists to detect "today" questions
- Retrieved documents contain mixed temporal data from different dates
- BAUG doesn't validate temporal consistency between question and context
- Existing `temporal_mismatch` check in `mismatch_flags()` only checks for year mismatches, not relative time words like "today"

## Implementation Plan

### 1. Fix Latency Tracking

**File**: `src/agentic_rag/supervisor/orchestrator.py`

**Location**: Line 696-699 (after LLM call)

**Change**:

```python
# Current code (line 696-699):
latency_ms = tmr()
tokens_last_round_val = usage.get("total_tokens", 0)
total_tokens += tokens_last_round_val
tokens_left -= tokens_last_round_val

# Fix: Add latency to list
latency_ms = tmr()
latencies.append(int(latency_ms))  # ADD THIS LINE
tokens_last_round_val = usage.get("total_tokens", 0)
total_tokens += tokens_last_round_val
tokens_left -= tokens_last_round_val
```

**Why**: The `latency_ms` variable is computed but never appended to the `latencies` list, causing the summary to show 0ms.

### 2. Add Temporal Validation

**File**: `src/agentic_rag/anchors/validators.py`

**Location**: After line 68 (end of `mismatch_flags` function)

**Add new function**:

```python
def temporal_context_validator(question: str, texts: Iterable[str]) -> dict:
    """
    Validate temporal consistency for time-sensitive questions.
    Detects relative time words (today, yesterday, this week, etc.) and checks
    if retrieved contexts contain conflicting temporal markers.

    Returns dict: {
        has_relative_time: bool,
        relative_time_words: list[str],
        has_conflicting_dates: bool,
        date_mentions: list[str],
        temporal_risk: float (0.0-1.0)
    }
    """
    ql = (question or "").lower()
    tl = "\n".join((t or "").lower() for t in texts)

    # Relative time indicators
    relative_time_patterns = [
        "today", "tonight", "this morning", "this afternoon",
        "yesterday", "tomorrow",
        "this week", "this month", "this year",
        "last week", "last month", "last year",
        "now", "current", "currently", "recent", "recently",
        "latest", "newest"
    ]

    found_relative = [p for p in relative_time_patterns if p in ql]
    has_relative_time = bool(found_relative)

    if not has_relative_time:
        return {
            "has_relative_time": False,
            "relative_time_words": [],
            "has_conflicting_dates": False,
            "date_mentions": [],
            "temporal_risk": 0.0
        }

    import re as _re

    # Extract day-of-week mentions from contexts (Monday, Tuesday, etc.)
    day_pattern = r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    days_in_context = _re.findall(day_pattern, tl)

    # Extract specific date mentions
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # MM/DD/YYYY
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",  # ISO format
    ]

    date_mentions = []
    for pattern in date_patterns:
        date_mentions.extend(_re.findall(pattern, tl))

    # Check for conflicting temporal markers
    # If question asks for "today" but context mentions specific past days
    has_conflicting_dates = False
    if "today" in ql or "current" in ql or "now" in ql:
        # Contexts mentioning specific past days are suspicious
        if len(set(days_in_context)) > 1:  # Multiple different days mentioned
            has_conflicting_dates = True
        elif days_in_context and "tuesday" in days_in_context:  # Example: "Tuesday's data" for "today"
            has_conflicting_dates = True

    # Risk score: 0-1 based on temporal inconsistency severity
    temporal_risk = 0.0
    if has_conflicting_dates:
        temporal_risk = 0.7
    elif days_in_context or date_mentions:
        temporal_risk = 0.4  # Some temporal markers present, moderate risk

    return {
        "has_relative_time": has_relative_time,
        "relative_time_words": found_relative,
        "has_conflicting_dates": has_conflicting_dates,
        "date_mentions": days_in_context + date_mentions,
        "temporal_risk": temporal_risk
    }
```

### 3. Integrate Temporal Validation into BAUG Decision

**File**: `src/agentic_rag/supervisor/orchestrator.py`

**Location**: Line 783-797 (where validators are called)

**Change**:

```python
# After line 784 (after numeric_req = ...)
temporal_val = temporal_context_validator(question, texts_for_validation)

# Update line 786-790:
validators_passed = not (
    award_req.get("missing")
    or numeric_req.get("missing")
    or list_req.get("missing")
    or temporal_val.get("has_conflicting_dates", False)  # ADD THIS
)

# Update line 791-796:
validators_snapshot = {
    "award": award_req,
    "numeric": numeric_req,
    "list": list_req,
    "temporal": temporal_val,  # ADD THIS
    "passed": validators_passed,
}
```

**Why**: This integrates temporal validation into the quality gates, preventing BAUG from confidently answering when temporal conflicts are detected.

### 4. Enhance BAUG Quality Thresholds for Temporal Conflicts

**File**: `src/agentic_rag/gate/adapter.py`

**Location**: Line 98-103 (in `_rule_based` method, after line 140)

**Add temporal conflict handling**:

```python
# After extracting sig = Signals.from_dict(raw_signals)
# Check if validators indicate temporal conflicts
validators = sig.extras.get("validators", {})
temporal_val = validators.get("temporal", {})
temporal_risk = temporal_val.get("temporal_risk", 0.0)

# If high temporal risk and overlap/faith not exceptional, be conservative
if temporal_risk > 0.6:
    if sig.overlap_est < 0.7 or sig.faith_est < 0.8:
        reasons.append(f"temporal_conflict_risk:{temporal_risk:.2f}")
        return "ABSTAIN", reasons
```

### 5. Update Experiment Report Script

**File**: `run_baug_full_experiment.ps1`

**Location**: Line 175-183 (metrics calculation)

**Add warning detection**:

```powershell
# After line 177 (after avg_latency calculation)
$has_latency_issue = $metrics.avg_latency -eq 0 -and $metrics.avg_rounds -gt 1
if ($has_latency_issue) {
    Write-Host "  [WARN] $($config.Name): Latency measurement appears broken (0ms with multiple rounds)" -ForegroundColor Yellow
}

# In the report generation, add diagnostic section
$report_content += ""
$report_content += "## Diagnostic Warnings"

foreach ($config in $configs) {
    $metrics = $analysis[$config.Name]
    if ($metrics) {
        if ($metrics.avg_latency -eq 0 -and $metrics.avg_rounds -gt 1) {
            $report_content += "- WARNING [$($config.Name)]: Latency measurement showing 0ms despite $($metrics.avg_rounds.ToString('F1')) rounds - possible measurement bug"
        }
    }
}
```

## Testing Plan

1. **Latency Fix Verification**:

   - Run `.\run_baug_full_experiment.ps1 -N 5`
   - Check that BAUG configurations show non-zero latency values
   - Verify `latencies` array contains values for each round

2. **Temporal Validation**:

   - Test with question 5 (Dow Jones "today")
   - Verify temporal validator detects "today" in question
   - Confirm conflicting dates are flagged when "Tuesday" appears in context
   - Check that BAUG abstains or retrieves more when temporal_risk > 0.6

3. **End-to-End**:

   - Run full N=50 experiment
   - Compare results with N=5 baseline
   - Verify both issues are resolved

## Files Modified

1. `src/agentic_rag/supervisor/orchestrator.py` - Fix latency tracking, integrate temporal validation
2. `src/agentic_rag/anchors/validators.py` - Add temporal context validator
3. `src/agentic_rag/gate/adapter.py` - Add temporal conflict handling to BAUG
4. `run_baug_full_experiment.ps1` - Add diagnostic warnings (optional)

## Success Criteria

- BAUG configurations show realistic latency measurements (> 0ms)
- `latencies` array populated with per-round measurements
- Temporal questions with "today" trigger temporal validation
- Conflicting temporal data (e.g., "Tuesday" vs "today") flags temporal_risk
- BAUG abstains or retrieves more when temporal conflicts detected
- Question 5 hallucination prevented by temporal validator

### To-dos

- [ ] Add latencies.append(int(latency_ms)) after line 696 in orchestrator.py to fix 0ms bug
- [ ] Create temporal_context_validator() function in validators.py to detect relative time words and conflicting dates
- [ ] Call temporal validator in orchestrator.py and add results to validators_snapshot
- [ ] Add temporal_risk check in BAUGAdapter._rule_based() to abstain on high temporal conflicts
- [ ] Update run_baug_full_experiment.ps1 to detect and report latency/temporal issues
- [ ] Run experiment with N=5 and verify BAUG shows non-zero latency
- [ ] Test with Dow Jones question and verify temporal conflicts are detected and handled
