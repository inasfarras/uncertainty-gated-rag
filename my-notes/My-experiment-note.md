# My Experiment Notes

This file contains the sequence of commands to set up the environment, process the CRAG dataset, and run the RAG system evaluations.

## 1. Environment Setup

First, create a virtual environment and install the required dependencies.

```powershell
# Create and activate the virtual environment (run once)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation & Ingestion

Download the full CRAG dataset, prepare the corpus, and ingest it into the FAISS vector store.

```powershell
# Step 1: Download the dataset from the original source
python scripts/crag_full_download.py

# Step 2: Prepare the corpus from the downloaded file (creates .txt files)
python scripts/prepare_crag_from_jsonl.py --static-only --n 200

# Step 3: Ingest the corpus into FAISS using OpenAI embeddings
python -m agentic_rag.ingest.ingest --input data/crag_corpus_html --out artifacts/crag_faiss --backend openai
```

## 3. Run Evaluations

Run the evaluation for both the baseline RAG system and the agentic system with UncertaintyGate.

```powershell
# Run the baseline system (simple RAG without gating)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --n 200

# Run the agent system with UncertaintyGate ON (smart stopping, MMR, reranking, HyDE, REFLECT)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 200

# Run the agent system with gate OFF (for comparison - always runs full rounds)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-off --n 200
```

## 4. Test Runs (Small Scale)

For quick testing and debugging, use smaller sample sizes:

```powershell
# Quick test with gate ON (10 questions)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 10

# Quick test with gate OFF (10 questions)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-off --n 10
```

## 5. Enhanced Features

The current agent system includes:
- **UncertaintyGate**: Smart decision making (STOP/REFLECT/RETRIEVE_MORE)
- **MMR Diversification**: Reduces redundancy in retrieved contexts (λ=0.4)
- **BGE Reranking**: Cross-encoder reranking for better relevance (optional)
- **HyDE Query Rewriting**: Hypothetical document embedding for better retrieval (optional)
- **REFLECT**: Self-correction mechanism for improving answers
- **Enhanced Debug Output**: Detailed progress tracking with emojis

---

## 6. Analysis

After running the evaluations, you can analyze the results using the `analyze_run.js` script. This script computes EM/F1 scores and categorizes the results.

```powershell
# Usage: node scripts/analyze_run.js <run.jsonl> <dataset.jsonl>
node scripts/analyze_run.js logs/1758373375_agent.jsonl data/crag_questions.jsonl
```

For a more detailed, per-question reason analysis, use the `analyze_per_question.js` script.

```powershell
# Usage: node scripts/analyze_per_question.js <run.jsonl> <dataset.jsonl>
node scripts/analyze_per_question.js logs/1758373375_agent.jsonl data/crag_questions.jsonl
```

## 7. 2025-09-25 Anchor System Debug (CRAG)

What changed
- Chunking: ingestion chunks reduced to 300 tokens (overlap 50) to isolate per-season/table rows for better selection.
- Anchors: added 50-40-90 detection (hyphen/slash), season ranges (e.g., 2005–06), and two-word entity anchors (e.g., steve nash).
- Retriever: hybrid fusion adds anchors like 3pa/three-point attempts; multi-chunk per doc selection + in-doc scanning; slicing long texts around season/3PA tokens; reserve rule to force one season+3PA chunk when special patterns present.
- Orchestrator prompt: instructs computing numeric answers from per-season/table rows and citing the chunk with the numbers.
- BM25: rebuilt to match FAISS after re-ingest (corpus now ~23,999 chunks).

Repro quick run (anchor)
```powershell
python -m agentic_rag.eval.runner ^
  --dataset data/crag_questions.jsonl ^
  --system anchor --n 3 ^
  --override "MAX_ROUNDS=3 JUDGE_POLICY=always USE_HYBRID_SEARCH=True ^
    RETRIEVAL_K=8 PROBE_FACTOR=2 RETRIEVAL_POOL_K=64 MAX_CONTEXT_TOKENS=1600 ^
    USE_RERANK=False MMR_LAMBDA=0.0 HYBRID_ALPHA=0.55 ANCHOR_BONUS=0.12"
```

Observed (latest)
- CEO Oracle (qid=161a89f3-…): answered correctly with [CIT].
- 50-40-90 Nash (qid=7bb29eb4-…): answers with a numeric average (5.0) + [CIT]; next step is deterministic averaging from per-season 3PA rows to match gold.
- Physics movie (qid=a2486535-…): still abstain; try HYDE and slightly lower HYBRID_ALPHA to bias lexical BM25.

Next steps
- Add numeric aggregator (config-guarded) to parse 3PA per-season rows and compute averages; cite the chunk.
- Strengthen reserve: always include at least one chunk with (season token AND 3PA) when special anchors are detected.
- Expand anchors for device/sci-fi phrasing (device/machine/invention; manipulate gravity/time/matter; The Core).

## 8. 2025-10-02 Anchor BAUG Sweep (200Q)

Commands used for the latest 200-question comparison (baseline vs. anchor gate toggles):

```powershell
# Baseline (pure vector, lowest tokens)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --n 200

# Anchor stack, BAUG gate OFF (control)
python -m agentic_rag.eval.runner ^
  --profile anchor_balanced_off --dataset data/crag_questions.jsonl --n 200 ^
  --override "RETRIEVAL_K=10 PROBE_FACTOR=2 RESERVE_ANCHOR_SLOTS=3 PACK_W_FINE=0.88 PACK_W_TITLE=0.07 PACK_W_RANK=0.05"

# Anchor stack, BAUG gate ON (target)
python -m agentic_rag.eval.runner ^
  --profile anchor_balanced --dataset data/crag_questions.jsonl --n 200 ^
  --judge-policy gray_zone ^
  --override "RETRIEVAL_K=10 PROBE_FACTOR=2 RESERVE_ANCHOR_SLOTS=3 BAUG_STOP_COVERAGE_MIN=0.35 PACK_W_FINE=0.88 PACK_W_TITLE=0.07 PACK_W_RANK=0.05"
```

Key metrics (answerable rows, averages):
- **Baseline** � faith 0.590, overlap 0.519, F1 0.158, tokens ~1,078.
- **Anchor gate OFF** � faith 0.611, overlap 0.531, F1 0.205, tokens ~1,158.
- **Anchor gate ON** � faith 0.621, overlap 0.547, F1 0.209, tokens ~1,157.

BAUG now beats both baseline and gate-off while holding token budget flat.

## 9. Gold-Aware GPT Judge

Use the new CLI to score any predictions JSONL against gold answers and cited passages:

```powershell
python -m eval.cli_judge -p logs/baseline/1759331314_baseline.jsonl --debug
```

Notes:
- `--gold` defaults to `data/crag_questions.jsonl`; override if you use another dataset.
- Output is written as `<predictions>_judge_gold.jsonl` alongside the input file.
- `--debug` prints `[judge] processed X/Y` so you can monitor progress.
- Additional options: `--parallel N`, `--require-citation true`, `--model gpt-4o-mini`, `--temp 0.2`.

### Benchmarks from Judge Output

```powershell
python -m eval.compute_benchmarks `
  --input logs/baseline/1759331314_baseline_judge_gold.jsonl `
  --out   logs/baseline/1759331314_baseline_benchmarks.jsonl
```

Outputs per-question RAB/AQB/Composite and prints the summary with abstain/safe/hallucination counts.
### Batch Judge + Benchmark Commands (baseline / anchor gate-off / gate-on)

```powershell
$runs = @(
    @{ Folder = "baseline"; File = "1759331314_baseline" },
    @{ Folder = "anchor";   File = "1759343917_anchor" },
    @{ Folder = "anchor";   File = "1759343984_anchor" }
)

foreach ($run in $runs) {
    $pred = "logs/$($run.Folder)/$($run.File).jsonl"
    $judge = "logs/$($run.Folder)/$($run.File)_judge_gold.jsonl"
    $bench = "logs/$($run.Folder)/$($run.File)_benchmarks.jsonl"

    python -m eval.cli_judge -p $pred --debug
    python -m eval.compute_benchmarks --input $judge --out $bench
}
```

### 10. Auto Run (Eval + Judge + Benchmarks)

```powershell (gate on)
python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions.jsonl `
  --system anchor `
  --judge-require-citation false `
  --judge-debug `
  --validator-limit 5 `
  -- --gate-on --n 200 --override "RETRIEVAL_K=10 PROBE_FACTOR=2"
```
python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions.jsonl `
  --system anchor `
  --judge-require-citation false `
  --judge-debug `
  --validator-limit 5 `
  --validator-f1-threshold 0.5 `
  -- --gate-on --n 5 --override "RETRIEVAL_K=10 PROBE_FACTOR=2 USE_RERANK=False"

Gate dimatikan:

python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions.jsonl `
  --system anchor `
  --judge-require-citation false `
  --judge-debug `
  --validator-limit 5 `
  -- --gate-off --n 200 --override "RETRIEVAL_K=10 PROBE_FACTOR=2  USE_RERANK=False"


Baseline (tanpa gating dan tanpa override tambahan):

python scripts/run_eval_with_judge.py `
  --dataset data/crag_questions.jsonl `
  --system baseline `
  --judge-require-citation false `
  --judge-debug `
  --validator-limit 5 `
  -- --n 200

- Setelah benchmark, script otomatis menampilkan ringkasan validator (jawaban salah, safe IDK, hallucination) hingga jumlah baris yang ditentukan `--validator-limit`.
- File output akhir tetap berada di `logs/<system>/` (prediksi, judge_gold, benchmarks).
- Tambahkan argumen runner setelah "--" (contoh: `--gate-on`, `--n 200`, `--override ...`).
- Script otomatis menangkap file log terbaru (`logs/<system>/<timestamp>_<system>.jsonl`), menjalankan `eval.cli_judge`, dan menghitung benchmark (`eval.compute_benchmarks`).
