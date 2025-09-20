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
