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

Run the evaluation for both the baseline RAG system and the agentic system.

```powershell
# Run the baseline system (gate OFF)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system baseline --gate-off --n 200

# Run the agent system (gate ON)
python -m agentic_rag.eval.runner --dataset data/crag_questions.jsonl --system agent --gate-on --n 200
```

---
