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
- (Online mode) **OPENAI_API_KEY**
- Windows, macOS, or Linux

---

## 🚀 Quickstart

1. **Clone the repo**
    ```bash
    git clone https://github.com/your-username/agentic-rag.git
    cd agentic-rag
    ```

2. **Create and activate a virtual environment**
    ```powershell
    # Windows PowerShell
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    ```bash
    # macOS / Linux
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -U pip wheel setuptools
    pip install -r requirements.txt
    ```

4. **Configure environment**
    ```bash
    cp .env.example .env
    # Edit .env and set OPENAI_API_KEY=sk-xxxxx (for online mode)
    ```

5. **Prepare a tiny corpus**

    Create a few `.txt` files in the `data/corpus/` directory.

6. **Run a smoke test**
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

## 📚 Using CRAG (Quivr/CRAG)

This project includes first-class support for the [Quivr/CRAG](https://huggingface.co/datasets/Quivr/CRAG) dataset.

1.  **Log in to Hugging Face (if needed)**
    ```bash
    # This may be required for gated or private datasets
    pip install huggingface_hub
    huggingface-cli login
    ```
    You will need to provide a token with read access.

2.  **Prepare the dataset**
    ```bash
    python scripts/prepare_crag.py --out-dir data/crag_corpus --qs-file data/crag_questions.jsonl --split test --static-only --n 200
    ```
    This downloads the dataset, converts HTML search results to clean text, and filters for high-quality static content. The documents are saved to `data/crag_corpus/` and questions to `data/crag_questions.jsonl`.

#### Live Fetching Mode (`--fetch-live`)

The `prepare_crag.py` script includes an optional `--fetch-live` mode. If a search result in the dataset is missing its HTML content (`page_result`), this mode will attempt to download the page from its original URL.

> **Disclaimer**: Live fetching interacts with external websites. This feature is intended for research purposes only.
> - **Be responsible**: You are responsible for complying with all applicable laws, website Terms of Service, and `robots.txt` files.
> - **Respect copyrights**: Do not use this feature to download content you do not have the right to access.
> - **Avoid high rates**: The script includes a default 1-second delay between requests to avoid overwhelming servers. Do not significantly lower this value or "hammer" websites. Abuse may lead to your IP address being blocked.

Example usage:
```bash
python scripts/prepare_crag.py --split train --static-only --n 200 --min-chars 300 --fetch-live --max-live-per-q 3
```

3.  **Ingest the corpus (choose one)**
    ```bash
    # OpenAI (recommended)
    make crag-ingest-openai

    # Mock (offline)
    make crag-ingest-mock
    ```

4.  **Run evaluation**
    ```bash
    # Baseline
    make crag-run-baseline

    # Agent
    make crag-run-agent
    ```

**Tips:**
*   Start with `--static-only` to avoid noisy, time-sensitive web content.
*   If `overlap` is frequently 0, try lowering `OVERLAP_SIM_TAU` to `0.55`–`0.60`.
*   Ensure your prompt template forces the model to use citations like `[CIT:chunk_id]` for every claim.

### Using CRAG (Full HTML)

> **Note**: The Corrective RAG (CRAG) dataset is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). It is intended for non-commercial, academic use. Please review the license terms before using the dataset.

This project supports the full CRAG dataset, which includes complete HTML content for web pages. Follow these steps to download, prepare, and evaluate using this dataset:

1.  **Download the dataset**
    ```bash
    make crag-full-download
    ```
    *(Alternatively: `python scripts/crag_full_download.py`)*

2.  **Prepare the corpus, questions, and metadata**
    ```bash
    make crag-full-prepare
    ```
    *(Alternatively: `python scripts/prepare_crag_from_jsonl.py --static-only --n 200`)*

3.  **Run the end-to-end pipeline using the Makefile targets:**

    ```bash
    # Ingest the text corpus into FAISS
    # Choose one of the following backends:
    make crag-full-ingest-openai   # Requires OPENAI_API_KEY
    # or
    make crag-full-ingest-mock     # For offline/testing

    # Run evaluations
    make crag-full-run-baseline
    make crag-full-run-agent
    ```

    You can also run the entire OpenAI-based pipeline with a single command:
    ```bash
    make crag-full-all-openai
    ```

#### Tips for a Custom Run

-   **Start with static content**: For more stable and reproducible evaluations, the default `crag-full-prepare` target uses the `--static-only` flag to filter out queries about real-time or fast-changing information.
-   **Tune retrieval parameters**: If you find that the system's answers are too generic or lack specific citations, you might need to adjust the retrieval parameters. In particular, lowering the `OVERLAP_SIM_TAU` threshold (e.g., to `0.55`–`0.60`) in the configuration can help retrieve more relevant chunks.
-   **Enforce citations**: Ensure that your model's prompt strongly encourages or requires citations in the format `[CIT:doc_id]` to improve traceability and accuracy.
-   **Adjust dataset size**: You can process more or fewer questions by changing the `--n` argument in the `crag-full-prepare` target in the `Makefile`. Similarly, modify `--max-pages-per-q` to control the number of web pages saved for each question.


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

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
