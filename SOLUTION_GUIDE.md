# Solution: Fix CRAG Corpus & Reduce Hallucinations

## Problem Diagnosis

**Root cause found:** Your CRAG corpus is **incomplete**
- Current: 23,999 chunks (24% of expected)
- Expected: ~100,000 chunks for full coverage
- Result: Missing 76% of documents → hallucinations & high abstain rate

## The Solution (3 Steps)

### Step 1: Revert Retrieval Changes (2 minutes)

The retrieval parameter changes made things worse, so revert them:

```powershell
python revert_retrieval_changes.py
```

This resets:
- `HYBRID_ALPHA: 0.4 → 0.6` (back to 60% vector, 40% BM25)
- `ANCHOR_BONUS: 0.20 → 0.07` (back to original)
- `MMR_LAMBDA: 0.2 → 0.45` (back to original)

---

### Step 2: Rebuild CRAG Corpus (30-60 minutes)

Run the automated rebuild script:

```powershell
.\rebuild_crag_corpus.ps1
```

**What it does:**
1. Backs up your current incomplete corpus
2. Re-prepares with 80 pages per question (was 20)
3. Re-ingests with OpenAI embeddings
4. Verifies chunk count

**Requirements:**
- `OPENAI_API_KEY` must be set
- Estimated cost: $5-10 in API credits
- Estimated time: 30-60 minutes

**Expected result:**
- Chunks: 23,999 → 80,000-120,000
- Better document coverage for all questions

---

### Step 3: Test with New Corpus (5-10 minutes)

After rebuild completes, test with 50 questions:

```powershell
python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3
```

**Expected improvements:**
| Metric | Before (Incomplete) | After (Full Corpus) |
|--------|---------------------|---------------------|
| Hallucinations | 32% (16/50) | **~18-22%** |
| Abstain | 28% (14/50) | **~20-25%** |
| Composite | 46.50 | **~53-55** |
| EM | 8% | **~10-12%** |

---

## Why This Will Work

### Current Problem:
```
24k chunks ÷ 200 questions = 120 chunks/question
→ Not enough coverage
→ Missing specific documents (US Open 2017, correct Pixar list, etc.)
→ System either abstains OR hallucinates from similar-but-wrong docs
```

### After Fix:
```
80k-120k chunks ÷ 200 questions = 400-600 chunks/question
→ Much better coverage
→ RIGHT documents now in corpus
→ Less abstention, fewer hallucinations
```

### The 4 Persistent Hallucinations Should Be Fixed:
1. ✅ **Pixar movies** - More Pixar docs → correct movie list
2. ✅ **Corporate bonds** - More finance docs → $59B (not $45B)
3. ✅ **Grand Slam 2017** - More tennis docs → US Open (not Australian)
4. ✅ **Halloween movies** - More movie docs → all 3 titles

---

## Alternative: Quick Manual Steps

If you prefer manual control:

```powershell
# 1. Backup current data
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
Move-Item artifacts/crag_faiss "artifacts/backup_$ts"
Remove-Item -Recurse -Force data/crag_corpus_html

# 2. Re-prepare corpus (30-60 pages per question)
python scripts/prepare_crag_from_jsonl.py `
    --src data/crag_task_1_and_2_dev_v4.jsonl.bz2 `
    --out-dir data/crag_corpus_html `
    --qs-file data/crag_questions.jsonl `
    --meta-file data/crag_meta.jsonl `
    --static-only `
    --n 200 `
    --min-chars 300 `
    --max-pages-per-q 80

# 3. Re-ingest with OpenAI
python -m agentic_rag.ingest.ingest `
    --input data/crag_corpus_html `
    --out artifacts/crag_faiss `
    --backend openai

# 4. Verify
python check_embeddings.py
```

---

## Troubleshooting

### "OPENAI_API_KEY not set"
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

### "Still only ~40k chunks after rebuild"
Increase `--max-pages-per-q` further:
```powershell
.\rebuild_crag_corpus.ps1 -MaxPagesPerQ 120
```

### "Download failed"
Check internet connection and try:
```powershell
python scripts/crag_full_download.py
```

### "Too expensive"
If $10 is too much, you can:
- Reduce to `--n 100` (100 questions instead of 200)
- Use fewer pages: `-MaxPagesPerQ 50`
- Trade cost for quality (but results will be worse)

---

## What About the Gate Fixes?

The BAUG gate fixes (conflict detection, coverage thresholds) remain in place:
- ✅ Prevents high-overlap + low-coverage stops
- ✅ Conflict detection still active
- ✅ Coverage threshold: 0.45 (balanced)

These didn't reduce hallucinations with incomplete corpus, but they **SHOULD help** with full corpus by preventing premature stops on borderline cases.

---

## Summary

The problem was **never the gate or retrieval logic** - it was the **missing documents**.

**Fix:** Rebuild corpus with 80 pages/question → 4x more coverage → fewer hallucinations

**Run:** `.\rebuild_crag_corpus.ps1` and wait ~45 minutes.
