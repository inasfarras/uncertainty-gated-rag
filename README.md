# Agentic RAG

Agentic Retrieval-Augmented Generation with an **uncertainty gate** that turns “I’m not sure” signals into **actions** (re-retrieve, switch, reflect, stop) under a **token/time budget**—with transparent logging and evaluation.

## ✨ Key Features
- **End-to-end RAG**: ingest → FAISS → retrieve → generate
- **Agentic loop**: simple planner + **uncertainty gate** (thresholds & budget)
- **OpenAI-only** path (chat + embeddings), **no local Torch** required
- **Offline/mock mode** to develop without an API key
- **Evaluation**: faithfulness & overlap (RAGAS if available, fallback if not)
- **Logging**: JSONL + CSV with tokens, latency, and per-step gate actions
- **CLI & Makefile** for fast smoke tests and experiments

---

## 📦 Requirements
- Python **3.11+**
- [Poetry](https://python-poetry.org/docs/#installation)
- (Online mode) **OPENAI_API_KEY**
- Windows, macOS, or Linux

---

## 🚀 Quickstart

1. **Clone the repo and install dependencies**
    ```bash
    git clone https://github.com/your-username/agentic-rag.git
    cd agentic-rag
    poetry install
    ```

2. **Activate the virtual environment**
    ```bash
    poetry shell
    ```

3. **Configure environment**
    ```bash
    cp .env.example .env
    # Edit .env and set OPENAI_API_KEY=sk-xxxxx (for online mode)
    ```

4. **Prepare a tiny corpus**

    Create a few `.txt` files in the `data/corpus/` directory.

5. **Run a smoke test**
    ```bash
    # Offline (mock embeddings; no API key)
    make smoke-mock

    # Online (OpenAI embeddings)
    make smoke-online
    ```
    The smoke tests will ingest the corpus, run the baseline and agent systems on a sample dataset, and output logs to the `logs/` directory.

---

## 🧰 Operating Modes

### Online (recommended)

* Embeddings & chat via **OpenAI** (default).
* Make sure `.env` includes `OPENAI_API_KEY`.

### Offline / Mock

* For development without an API key.
* `--backend mock` yields **deterministic** embeddings from a hash.
* Great for CI and unit tests.

---

## ⚙️ Configuration (`.env` / `config.py`)

| Key/Setting         | Default                  | Description                               |
| ------------------- | ------------------------ | ----------------------------------------- |
| `OPENAI_API_KEY`    | `""`                     | Required for online mode                  |
| `LLM_MODEL`         | `gpt-4o-mini`            | OpenAI chat model                         |
| `EMBED_BACKEND`     | `openai`                 | `openai` \| `mock`                        |
| `EMBED_MODEL`       | `text-embedding-3-small` | OpenAI embedding model                    |
| `MAX_TOKENS_TOTAL`  | `3500`                   | Token budget per query                    |
| `MAX_ROUNDS`        | `2`                      | Max agent iterations                      |
| `RETRIEVAL_K`       | `8`                      | Initial top-k for vector search           |
| `FAITHFULNESS_TAU`  | `0.75`                   | Confidence threshold (faithfulness)       |
| `OVERLAP_TAU`       | `0.50`                   | Confidence threshold (overlap)            |
| `OVERLAP_SIM_TAU`   | `0.7`                    | Cosine sim threshold for sentence support |
| `LOW_BUDGET_TOKENS` | `500`                    | Stop when remaining tokens < this         |

> CLI flags can override most of these at runtime (e.g., `--tau-f`, `--tau-o`, `--max-rounds`).

---

## 🧪 Evaluation & Outputs

### Sample dataset

`data/sample.jsonl`:

```jsonl
{"id":"q1","question":"What is the main purpose of the documents in the corpus?"}
{"id":"q2","question":"Name one explicit fact mentioned in any file."}
{"id":"q3","question":"Summarize the corpus in one sentence."}
```

### Run evaluation

```bash
# Baseline
python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system baseline --gate-off --n 3

# Agent
python -m agentic_rag.eval.runner --dataset data/sample.jsonl --system agent --gate-on --n 3
```

### Artifacts

* **JSONL per run**: `logs/<timestamp>_<system>.jsonl`
  One record per query (summary) plus **per-step gate logs** (for agent).
* **CSV summary**: `logs/<timestamp>_<system>_summary.csv`
  Aggregates: `final_f` (faithfulness), `final_o` (overlap), `total_tokens`, `latency_ms (p50)`

> If **RAGAS** is unavailable, faithfulness falls back to
> `min(1.0, 0.6 + 0.4 * overlap)`.

---

## 🧠 How It Works (Short)

1. **Retrieve** top-k chunks (FAISS; cosine/IP over **normalized** embeddings).
2. **Generate** an answer **only from the provided context**. The prompt instructs the model to append a citation `[CIT:<chunk_id>]` to every sentence.
3. **Signals**: compute `overlap` (share of sentences supported by context) & `faithfulness`.
4. **Gate** decides **STOP / RETRIEVE\_MORE / SWITCH\_GRAPH (optional) / REFLECT** under a **budget**.

---

## 🧭 Makefile

| Command         | Description                                     |
| --------------- | ----------------------------------------------- |
| `make format`     | Formats code with `black`.                      |
| `make lint`       | Lints and fixes code with `ruff`.               |
| `make type`       | Type-checks with `mypy`.                        |
| `make smoke-mock` | Runs the full offline pipeline (mock backend).  |
| `make smoke-online` | Runs the full online pipeline (OpenAI backend). |

---

## 🔧 Troubleshooting

* **401/Unauthorized** → verify `OPENAI_API_KEY` and environment loading.
* **Connection/SSL** → check network/proxy; set `OPENAI_BASE_URL` if using a gateway.
* **Rate limiting** → reduce corpus/questions; add small delays for ingestion.
* **Overlap always 0** → ensure the prompt forces citations, lower `OVERLAP_SIM_TAU` (e.g., 0.55–0.6), and ask questions that are **answerable from the corpus**.

---

## 🗂 Project Structure

```
src/agentic_rag/
  agent/       # loop, gate, switcher, prompt builder
  embed/       # chunking & embeddings (OpenAI/mock)
  eval/        # signals (overlap/faithfulness) & runner
  ingest/      # loader & FAISS index builder
  models/      # OpenAIAdapter (chat/embeddings/usage)
  retriever/   # vector retriever
  store/       # FAISS store (build/load/search)
  utils/       # JSONL & timing helpers
api/           # FastAPI (optional)
logs/          # JSONL & CSV outputs
artifacts/     # FAISS index + meta
data/          # corpus & datasets
```

---

## 📄 License

MIT

```
