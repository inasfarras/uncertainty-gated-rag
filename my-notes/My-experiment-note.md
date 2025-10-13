# CRAG Benchmark Experiment Notes

Complete workflow for setting up, preparing data, and evaluating the Agentic RAG system on the CRAG benchmark.

---

## 1. Environment Setup

Create a Python 3.11 virtual environment and install dependencies.

```powershell
# Create and activate virtual environment (one-time setup)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Data Preparation & Ingestion

### Step 1: Download CRAG Dataset
The dataset should already be at `data/crag_task_1_and_2_dev_v4.jsonl.bz2` (2706 questions).

### Step 2: Prepare Full Corpus
Convert the CRAG dataset into text documents with metadata.

```powershell
python scripts/prepare_crag_from_jsonl.py `
  --src data/crag_task_1_and_2_dev_v4.jsonl.bz2 `
  --out-dir data/crag_corpus_full `
  --qs-file data/crag_questions_full.jsonl `
  --meta-file data/crag_meta_full.jsonl `
  --min-chars 200 `
  --fallback-snippet
```

**Output:**
- `data/crag_corpus_full/` → ~13,279 text documents
- `data/crag_questions_full.jsonl` → 2,706 questions
- `data/crag_meta_full.jsonl` → Document metadata (URL, title, rank)

### Step 3: Ingest into FAISS (Offline Embeddings)

**Using Sentence-Transformers (FREE, no API key required):**

```powershell
# Ingest full corpus with offline embeddings
python -m agentic_rag.ingest.ingest `
  --input data/crag_corpus_full `
  --out artifacts/crag_faiss_full `
  --backend st
```

**Using OpenAI Embeddings (requires API key):**

```powershell
# Alternative: OpenAI embeddings (faster, but costs $$)
python -m agentic_rag.ingest.ingest `
  --input data/crag_corpus_full `
  --out artifacts/crag_faiss_full `
  --backend openai
```

**Output:**
- `artifacts/crag_faiss_full/` → FAISS vector index with ~13k chunks

### Step 4: Update Configuration

After ingesting, update `src/agentic_rag/config.py` to use the full corpus:

```python
FAISS_INDEX_PATH: str = "artifacts/crag_faiss_full"
EMBED_BACKEND: Literal["openai", "st", "mock"] = "st"  # Use "st" for offline
```

Also update paths in `src/agentic_rag/supervisor/orchestrator.py`:

```python
CRAG_META_PATH = "data/crag_meta_full.jsonl"
CRAG_QUESTIONS_PATH = "data/crag_questions_full.jsonl"
```

---

## 3. Run Evaluations (Auto: Eval + Judge + Benchmarks)

The `run_eval_with_judge.py` script automates the full evaluation pipeline:
1. Runs the RAG system
2. Evaluates with the judge
3. Computes benchmarks
4. Displays validation summary

### 3.1 Anchor System (Recommended)

**With BAUG ON (Smart Multi-Round RAG):**

```powershell
python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions_full.jsonl `
  --system anchor `
  -- --n 100 --gate-on
```

**With BAUG OFF (Naive Multi-Round RAG for comparison):**

```powershell
python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions_full.jsonl `
  --system anchor `
  -- --n 100 --gate-off
```

### 3.2 Baseline System (Simple Single-Round RAG)

```powershell
python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions_full.jsonl `
  --system baseline `
  -- --n 100
```

### 3.3 Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Path to questions JSONL file | Required |
| `--system` | System to evaluate (`anchor`, `baseline`) | Required |
| `--judge-require-citation` | Require citations in judge | `false` |
| `--validator-limit` | Max validation examples to show | 5 |
| `--n` | Number of questions (after `--`) | 0 (all) |
| `--override` | Override config values (after `--`) | - |

**Common Overrides:**
- `RETRIEVAL_K=20` → Retrieve top 20 documents
- `PROBE_FACTOR=3` → Retrieve 3× more for reranking
- `USE_RERANK=True` → Enable BGE reranker
- `EMBED_BACKEND=st` → Use sentence-transformers

### 3.4 Full Experiment (Automated)

**Single Command for Complete Analysis:**

The new `run_baug_full_experiment.ps1` script automates the entire experiment workflow, running all three configurations and generating a comprehensive comparison report.

```powershell
# Run full experiment with 50 questions
.\run_baug_full_experiment.ps1 -N 50

# Run with 100 questions
.\run_baug_full_experiment.ps1 -N 100

# Run with 200 questions
.\run_baug_full_experiment.ps1 -N 200

# Run with 1000 questions
.\run_baug_full_experiment.ps1 -N 1000

# Skip baseline if already completed
.\run_baug_full_experiment.ps1 -N 200 -SkipBaseline

# Skip BAUG configurations if only need baseline
.\run_baug_full_experiment.ps1 -N 100 -SkipBAUG:$true

# Regenerate report from most recent N=100 logs without re-running
.\run_baug_full_experiment.ps1 -N 100 -ReanalyzeOnly
```

**What the script does:**
1. Runs baseline system evaluation
2. Runs anchor + BAUG ON evaluation
3. Runs anchor + BAUG OFF evaluation
4. Analyzes results from all three configurations
5. Generates comprehensive report: `logs/comparisons/full_report_n{N}.md`

**New in v2.1:**
- **`-ReanalyzeOnly`**: Skips steps 1-3 and regenerates the report from the most recent existing log files for the specified `-N`. This is useful for fixing the report's formatting or analysis logic without re-running the full experiment.

**Example output report structure:**
```markdown
# BAUG Full Experiment Report - N=50 Questions

## Executive Summary
## Experimental Setup
## Results Summary (comparison table)
## Key Performance Metrics Comparison
## BAUG Impact Analysis
## Critical Findings
## Recommendations
## Technical Details
```

**Tips for running experiments at scale:**
- Start with N=50 for quick validation
- Use N=100 for standard comparison
- Run N=200 for statistical significance
- Use N=1000 for comprehensive analysis
- Use `-SkipBaseline` to save time on subsequent runs

---

## 4. Evaluation Metrics

After running the evaluation, you'll see:

```
Benchmark Summary
Count: 100
Avg Faithfulness: 0.652
Avg Overlap: 0.520
Avg RAB: 58.60
Avg AQB: 55.96
Avg Composite: 50.43
Avg F1_short: 0.236
Avg SupportOverlap: 0.074
Counts - match: 24, partial_match(correct): 21, safe_idk: 13, hallucination: 15
```

**Key Metrics:**
- **EM (Exact Match)**: Percentage of perfect answers
- **F1**: Token overlap with ground truth
- **Abstain Rate**: Percentage of "I don't know" responses
- **Hallucination**: Incorrect answers flagged by judge
- **Safe IDK**: Correct abstentions on unanswerable questions

---

## 5. System Architecture

### Components:
1. **Retrieval**: Hybrid search (FAISS + BM25) with MMR diversification
2. **Reranking**: BGE cross-encoder reranker (optional)
3. **BAUG (Budget-Aware Uncertainty Gate)**: Post-generation decision making:
   - **Quality Check**: STOP if overlap/faith/coverage > thresholds
   - **Marginal Gain**: STOP if quality improvement per token < threshold
   - **Intelligent Abstention**: ABSTAIN if coverage very low and stagnant
   - **Smart Retrieval**: RETRIEVE_MORE only when beneficial
4. **Answer Generation**: GPT-4o-mini with citation enforcement
5. **Multi-Round Logic**: Up to MAX_ROUNDS with intelligent early stopping

### Current Configuration (Default):
```python
RETRIEVAL_K: int = 8           # Top-K retrieval
PROBE_FACTOR: int = 4          # Retrieve 4×K for reranking
MAX_ROUNDS: int = 2            # Max retrieval rounds
USE_RERANK: bool = True        # Enable BGE reranker
EMBED_BACKEND: str = "st"      # Offline embeddings
TEMPERATURE: float = 0.0       # Deterministic generation

# BAUG Thresholds (for cost-benefit analysis)
TAU_OVERLAP: float = 0.35      # Overlap threshold for quality check
TAU_FAITH: float = 0.55        # Faithfulness threshold for quality check
TAU_COVERAGE: float = 0.55     # Anchor coverage threshold
LAMBDA_MIN_GAIN: float = 0.05  # Minimum marginal gain per 100 tokens
EPS_NEW_HITS: float = 0.12     # Minimum new hits ratio
```

---

## 6. Output Files

All outputs are saved to `logs/<system>/<timestamp>_<system>.*`:

- `<timestamp>_<system>.jsonl` → Full predictions with metadata
- `<timestamp>_<system>_summary.csv` → Metrics summary (CSV)
- `<timestamp>_<system>_qa_pairs.csv` → Question-answer pairs
- `<timestamp>_<system>_judge_gold.jsonl` → Judge evaluations
- `<timestamp>_<system>_benchmarks.jsonl` → Final benchmark scores
- `<timestamp>_<system>_summary.md` → Markdown summary

---

## 7. Troubleshooting

### Issue: Low EM/F1 Scores
**Possible causes:**
- Corpus size too small → Process full dataset (2706 questions)
- Retrieval K too low → Increase `RETRIEVAL_K=20`
- Reranker disabled → Enable with `USE_RERANK=True`
- Wrong embeddings → Ensure FAISS index matches `EMBED_BACKEND`

### Issue: High Abstain Rate
**Possible causes:**
- BAUG too conservative → Lower `TAU_COVERAGE` or `TAU_OVERLAP`
- Insufficient retrieval → Increase `PROBE_FACTOR=6`
- Poor retrieval quality → Check corpus preparation

### Issue: High Token Usage (BAUG not saving costs)
**Possible causes:**
- Marginal gain threshold too low → Increase `LAMBDA_MIN_GAIN=0.08`
- Quality thresholds too strict → Lower `TAU_OVERLAP` or `TAU_FAITH`
- MAX_ROUNDS too high → Reduce `MAX_ROUNDS=1` for testing

### Issue: Hallucinations
**Possible causes:**
- BAUG allows low-quality answers → Raise `TAU_OVERLAP` or `TAU_FAITH`
- Context too short → Increase `MAX_CONTEXT_TOKENS`
- LLM temperature too high → Set `TEMPERATURE=0.0`

---

## 8. Dataset Statistics

**Full CRAG Benchmark:**
- Questions: **2,706** (complete CRAG Task 1 & 2 Dev v4)
- Documents: ~13,279 (5 search results per question)
- Avg document size: ~5,800 chars (after HTML→text conversion)
- Domains: Web search results (factoid questions)
- **Evaluation Time**: ~2-4 hours for full 2,706 questions (depending on system)

**Filtering:**
- Documents < 200 chars: ~73 (0.5%)
- Empty HTML: ~38 (0.3%)
- Total retention: ~99.2%

---

## 9. Quick Reference

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Quick test: BAUG ON vs OFF comparison (5 questions each)
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --n 5 --gate-on
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --n 5 --gate-off

# Medium test (100 questions) - Good for development/debugging
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --n 100 --gate-on

# Large test (500 questions) - Substantial evaluation
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --n 500 --gate-on

# FULL BENCHMARK (2,706 questions) - Complete CRAG evaluation
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --gate-on

# FULL BENCHMARK comparison - BAUG OFF (2,706 questions)
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --gate-off

# FULL BENCHMARK baseline (2,706 questions) - Simple single-round RAG
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system baseline

# Run with custom BAUG thresholds (100 questions for tuning)
python scripts/run_eval_with_judge.py --dataset data/crag_questions_full.jsonl --system anchor -- --n 100 --gate-on --override "TAU_COVERAGE=0.45 LAMBDA_MIN_GAIN=0.08"

# Check recent logs
ls logs/anchor/ | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

---

## Notes

- **Embeddings**: Using sentence-transformers (offline) saves API costs but may be slower than OpenAI.
- **Evaluation Time**: ~2-3 seconds per question (BAUG ON), ~3-4 seconds (BAUG OFF), ~1-2 seconds (baseline).
- **Cost**: ~$0.01-0.02 per question with gpt-4o-mini (LLM only, embeddings free with ST).
- **Full Benchmark Cost**: ~$27-54 for 2,706 questions (LLM costs only).
- **BAUG Performance**: Typically 15-25% token savings vs naive multi-round, with equal or better quality.
- **Dataset**: CRAG Task 1 & 2 Dev v4 (static web pages only, no dynamic queries).

## Expected Results Summary

Based on our 5-question tests:
- **BAUG ON**: ~3224 tokens, 40% abstain, F1=0.444, EM=0.400
- **BAUG OFF**: ~4094 tokens, 40% abstain, F1=0.244, EM=0.200
- **Token Savings**: ~21% reduction with BAUG
- **Quality Improvement**: 82% better F1, 100% better EM with BAUG
