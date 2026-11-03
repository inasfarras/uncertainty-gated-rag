# Uncertainty-Gated RAG

Retrieval-augmented generation with a budgeted uncertainty gate. The package in `src/agentic_rag` (version 0.1.0) ingests plain-text documents into a FAISS index, retrieves passages, and generates answers that cite those passages. An uncertainty gate can stop, retrieve again, reflect, or abstain when a token budget or a confidence threshold is reached. The anchor system uses a second controller, BAUG (`BAUGAdapter` in `src/agentic_rag/gate/adapter.py`), to make the same class of decision from overlap, faithfulness, anchor coverage, and remaining budget.

Repository: [https://github.com/inasfarras/uncertainty-gated-rag](https://github.com/inasfarras/uncertainty-gated-rag)

## Features

- **Ingest.** Recursive `*.txt` loading, tiktoken chunks (300 tokens, 50-token overlap), L2-normalized embeddings, and a FAISS `IndexFlatIP` index (`src/agentic_rag/ingest/ingest.py`, `src/agentic_rag/store/faiss_store.py`). Each index directory contains `index.faiss`, `meta.parquet`, and `chunks.parquet`.
- **Embedding backends** in `src/agentic_rag/embed/encoder.py`:
  - `openai` — `EMBED_MODEL`, default `text-embedding-3-small`
  - `st` — sentence-transformers, default `sentence-transformers/all-MiniLM-L6-v2`
  - `mock` — deterministic MD5 hash vectors (dimension 1536) for offline runs
- **Three systems** selected by `python -m agentic_rag.eval.runner --system`: `baseline`, `agent`, and `anchor`.
- **Agent gate** (`UncertaintyGate` in `src/agentic_rag/agent/gate.py`). Actions: `STOP`, `RETRIEVE_MORE`, `REFLECT`, `ABSTAIN`, `STOP_LOW_BUDGET`.
- **Anchor system** (`AnchorSystem` in `src/agentic_rag/supervisor/orchestrator.py`) with intent interpretation, anchor coverage, and `BAUGAdapter` decisions `STOP`, `RETRIEVE_MORE`, and `ABSTAIN`. Set `BAUG_HANDLER` to `module:function` to replace the built-in rules with an external callable.
- **Retrieval options** on `VectorRetriever`: FAISS search, BM25 hybrid search (`USE_HYBRID_SEARCH`), MMR (`MMR_LAMBDA`), HyDE (`USE_HYDE`), and an optional BGE reranker (`USE_RERANK`, package `FlagEmbedding`). The baseline constructor turns hybrid search, reranking, and MMR off for that run.
- **Metrics** in `src/agentic_rag/eval/signals.py`: citation overlap, a faithfulness fallback, exact match, and token F1. RAGAS faithfulness runs when `--faith-report` is `ragas` or `both`.
- **Logs.** One JSONL file and one CSV summary per run, under a directory chosen from the system and gate flag.
- **CRAG preparation** via `scripts/prepare_crag.py`, `scripts/crag_full_download.py`, and `scripts/prepare_crag_from_jsonl.py`, with matching Makefile targets.
- **HTTP modules** under `src/agentic_rag/api/`. `app.py` registers `GET /health`. `main.py` also registers `POST /query` and `GET /stats`.

## Architecture

1. **Ingest.** `Ingestor` reads `**/*.txt`, sanitizes each file stem into a document id, chunks the text, embeds the chunks, and builds the index.
2. **Retrieve.** `VectorRetriever` loads `settings.FAISS_INDEX_PATH` (default `artifacts/crag_faiss_full`) and searches by inner product over normalized vectors.
3. **Generate.** Agent and anchor prompts require an answer drawn from the supplied context. A non-abstaining sentence carries one `[CIT:<id>]` citation whose id matches a context id. The abstention string is `I don't know`, with no citation.
4. **Signals.** Overlap is the fraction of answer sentences supported by a citation or by cosine similarity at or above `OVERLAP_SIM_TAU`. Fallback faithfulness is `min(1.0, 0.6 + 0.4 * overlap)`. If the answer is "I don't know" and a gold answer is present, that fallback is `0.0`.
5. **Agent gate.** `UncertaintyGate` uses faithfulness, overlap, lexical uncertainty, and the remaining token budget (`MAX_TOKENS_TOTAL`, `LOW_BUDGET_TOKENS`). `MAX_ROUNDS` (default 3) caps iterations. `--gate-on` / `--gate-off` enables or disables this gate for `--system agent`.
6. **Anchor / BAUG.** The anchor system adds anchor coverage and conflict checks. `BAUGAdapter` then returns `STOP`, `RETRIEVE_MORE`, or `ABSTAIN`. For `--system anchor`, the same `--gate-on` / `--gate-off` flag sets `ANCHOR_GATE_ON`.

```
src/agentic_rag/
  agent/        loop, uncertainty gate, judge, switcher, finalize
  anchors/      anchor prediction and validators
  api/          FastAPI modules
  embed/        chunking and embeddings
  eval/         runner, overlap, faithfulness, EM/F1
  gate/         BAUG adapter
  ingest/       .txt loader and FAISS builder
  intent/       rule and LLM intent interpretation
  rerank/       optional BGE reranker
  retrieval/    retrieval agent and MMR
  retriever/    vector, BM25, HyDE, MMR
  store/        FAISS index read/write
  supervisor/   anchor orchestrator
  telemetry/    round and summary logging
  utils/        JSON, timing, telemetry helpers
scripts/        CRAG download/prepare and log analysis
```

Chat and embedding calls import `agentic_rag.models.adapter`. Anchor evaluation also imports `agentic_rag.data.meta`. Those packages are not in the tracked tree. `.gitignore` ignores directories named `models/` and `data/`.

## Setup

CI in `.github/workflows/ci.yml` uses Python 3.11.

```bash
git clone https://github.com/inasfarras/uncertainty-gated-rag
cd uncertainty-gated-rag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

The library is not declared as an installable distribution (`requirements.txt` has no path to `src/`). Export `PYTHONPATH` before `python -m agentic_rag`:

```bash
export PYTHONPATH=src
```

```powershell
$env:PYTHONPATH = "src"
```

`requirements.txt` omits imports that the source loads at runtime. Add these when you hit an import error: `pydantic-settings` (`src/agentic_rag/config.py`), `sentence-transformers` (imported by `src/agentic_rag/embed/encoder.py` for every backend; the `st` backend also imports `torch`), and `tqdm` (`src/agentic_rag/ingest/ingest.py`). Optional imports: `FlagEmbedding` for the BGE reranker, and `spacy` with the `en_core_web_sm` model on one path in `src/agentic_rag/agent/loop.py`. Importing the BM25 retriever downloads the NLTK packages `punkt`, `punkt_tab`, and `stopwords`.

Copy the environment template when you use the OpenAI backend:

```bash
cp .env.example .env
```

`.env.example` sets `OPENAI_API_KEY` (empty), `OPENAI_BASE_URL`, `OPENAI_ORG`, `LLM_MODEL=gpt-4o-mini`, `EMBED_BACKEND=openai`, and `EMBED_MODEL=text-embedding-3-small`. `Settings` loads that file. With no `.env`, `EMBED_BACKEND` stays at the `config.py` default, `st`.

`make install` runs `poetry install`. This checkout has no `pyproject.toml` and no `poetry.lock`. Makefile targets `format`, `lint`, `type`, `test`, and `run-api` prefix their commands with `poetry run` for the same reason. The corresponding tools are already listed in `requirements.txt`. With the virtualenv active and `PYTHONPATH=src`:

```bash
ruff check .
black .
mypy src/agentic_rag
pytest
```

The CI test invocation is `pytest -q -m "not slow and not e2e"`. Those markers are defined in `pytest.ini`.

### Configuration

Defaults below are the field values in `src/agentic_rag/config.py`. A copied `.env.example` overrides `EMBED_BACKEND` to `openai`. The runner accepts `--tau-f`, `--tau-o`, `--max-rounds`, and repeatable `--override KEY=VAL`.

| Setting | Default in `config.py` | Role |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | OpenAI credential |
| `OPENAI_BASE_URL` | unset | Optional API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBED_BACKEND` | `st` | `openai`, `st`, or `mock` |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ST_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `FAISS_INDEX_PATH` | `artifacts/crag_faiss_full` | Index directory the retriever loads |
| `MAX_TOKENS_TOTAL` | `3500` | Token budget for one query |
| `MAX_ROUNDS` | `3` | Iteration cap |
| `RETRIEVAL_K` | `8` | Top-k passages |
| `FAITHFULNESS_TAU` | `0.55` | Faithfulness threshold |
| `OVERLAP_TAU` | `0.35` | Overlap threshold |
| `OVERLAP_SIM_TAU` | `0.60` | Sentence-to-context cosine threshold |
| `LOW_BUDGET_TOKENS` | `500` | Remaining-token floor for `STOP_LOW_BUDGET` |
| `USE_HYBRID_SEARCH` | `true` | Mix FAISS with BM25 (`HYBRID_ALPHA` 0.6) |
| `USE_RERANK` | `true` | BGE reranker when `FlagEmbedding` is installed |
| `USE_HYDE` | `false` | Hypothetical-document query expansion |
| `JUDGE_POLICY` | `gray_zone` | Settings default. The runner CLI defaults to `never` unless `--judge-policy` is set |

Further anchor, packing, and judge fields live on `Settings` in `src/agentic_rag/config.py`.

## Usage

`data/` is gitignored, so a fresh clone has no corpus and no question file. Create both locally.

```bash
mkdir -p data/corpus
printf 'The uncertainty gate can stop, retrieve again, reflect, or abstain.\n' > data/corpus/overview.txt
printf '%s\n' '{"id":"q1","question":"What can the uncertainty gate do?","gold":"stop, retrieve again, reflect, or abstain"}' > data/sample.jsonl
```

Each JSONL row needs `id` and `question`. `gold` or `answer` is optional and feeds exact match and token F1.

Build the index with the flags defined on the ingest command (`--input`, `--out`, `--backend`). The retriever reads `FAISS_INDEX_PATH`, so point that variable at the directory you just wrote:

```bash
export PYTHONPATH=src
export FAISS_INDEX_PATH=artifacts/faiss
python -m agentic_rag.ingest.ingest --input data/corpus --out artifacts/faiss --backend mock
```

`--backend openai` uses `OPENAI_API_KEY`. `--backend st` uses `ST_EMBED_MODEL` on CPU or CUDA.

Run the evaluator. `--dataset` is required. `--n 0` (the default) reads the whole file.

```bash
python -m agentic_rag.eval.runner \
  --dataset data/sample.jsonl \
  --system baseline \
  --gate-off \
  --backend mock \
  --n 3

python -m agentic_rag.eval.runner \
  --dataset data/sample.jsonl \
  --system agent \
  --gate-on \
  --backend mock \
  --n 3

python -m agentic_rag.eval.runner \
  --dataset data/sample.jsonl \
  --system anchor \
  --gate-on \
  --backend mock \
  --n 3
```

`--profile` applies a built-in override bundle and may switch `--system`: `anchor_fast`, `anchor_balanced`, `anchor_balanced_off`, `anchor_hybrid_light`, `baseline_plain`.

```bash
python -m agentic_rag.eval.runner \
  --dataset data/sample.jsonl \
  --profile baseline_plain \
  --backend mock \
  --n 3
```

### Makefile

Export `PYTHONPATH=src` before these targets. The smoke and CRAG targets call `python` directly. `install`, `format`, `lint`, `type`, `test`, and `run-api` call Poetry, which this repository does not configure.

| Target | Command it runs |
| --- | --- |
| `smoke-mock` | Ingest `data/corpus` → `artifacts/faiss` with `--backend mock`, then baseline (`--gate-off`) and agent (`--gate-on`) on `data/sample.jsonl` with `--n 3` |
| `smoke-online` | Same steps with `--backend openai` on ingest |
| `crag-prepare` | `scripts/prepare_crag.py --out-dir data/crag_corpus --qs-file data/crag_questions.jsonl --split test --static-only --n 200` |
| `crag-ingest-openai` / `crag-ingest-mock` | Ingest `data/crag_corpus` → `artifacts/crag_faiss` |
| `crag-run-baseline` / `crag-run-agent` | Evaluate `data/crag_questions.jsonl` with `--n 200` |
| `crag-all-mock` | `crag-prepare`, mock ingest, baseline, agent |
| `crag-full-download` | `scripts/crag_full_download.py` |
| `crag-full-prepare` | `scripts/prepare_crag_from_jsonl.py` with `--src data/crag_task_1_and_2_dev_v4.jsonl.bz2 --out-dir data/crag_corpus_html --qs-file data/crag_questions.jsonl --meta-file data/crag_meta.jsonl --static-only --n 200 --min-chars 500 --max-pages-per-q 20` |
| `crag-full-ingest-openai` / `crag-full-ingest-mock` | Ingest `data/crag_corpus_html` → `artifacts/crag_faiss` |
| `crag-full-run-baseline` / `crag-full-run-agent` | Same runner invocation as the non-full CRAG run targets |
| `crag-full-all-openai` | Download, prepare, OpenAI ingest, baseline, agent |

Smoke targets pass `--backend` to ingest only. The following runner process uses `EMBED_BACKEND` from the environment or from `config.py` unless you pass `--backend` yourself. They also expect `data/corpus` and `data/sample.jsonl`.

`make run-baseline` and `make run-agent` invoke the runner with no arguments. Pass `--dataset` on the `python -m agentic_rag.eval.runner` command instead. `make ingest` passes `--input-dir` and `--out-dir`; the Typer command defines `--input` and `--out`.

CRAG ingest targets write `artifacts/crag_faiss`. Set `FAISS_INDEX_PATH=artifacts/crag_faiss` before a run that should read that index. The config default remains `artifacts/crag_faiss_full`.

`make run-api` serves `agentic_rag.api.app:create_app` through Poetry and reads `API_HOST` and `API_PORT` from the environment. That factory exposes `GET /health`. `python -m agentic_rag.api.main` serves `agentic_rag.api.main:app` on `0.0.0.0:8000` and adds `POST /query` and `GET /stats`. Both import the agent stack, including `agentic_rag.models`.

### CRAG

`scripts/prepare_crag.py` loads the Hugging Face dataset `Quivr/CRAG` (config `crag_task_1_and_2` by default). Public access needs no token. If the Hub requests one, set `HF_TOKEN`.

```bash
python scripts/prepare_crag.py \
  --out-dir data/crag_corpus \
  --qs-file data/crag_questions.jsonl \
  --split test \
  --static-only \
  --n 200
```

`--fetch-live` downloads a page when a row has no `page_result`. The default pause is 1 second (`--sleep`) and the default cap is 3 pages per question (`--max-live-per-q`). Comply with the site's terms, `robots.txt`, and copyright.

`scripts/crag_full_download.py` fetches `crag_task_1_and_2_dev_v4.jsonl.bz2` from `https://raw.githubusercontent.com/facebookresearch/CRAG`. `scripts/prepare_crag_from_jsonl.py` converts that file to text, questions, and metadata. Use the license published with the upstream dataset; this repository does not include the data.

```bash
python scripts/crag_full_download.py
python scripts/prepare_crag_from_jsonl.py \
  --src data/crag_task_1_and_2_dev_v4.jsonl.bz2 \
  --out-dir data/crag_corpus_html \
  --qs-file data/crag_questions.jsonl \
  --meta-file data/crag_meta.jsonl \
  --static-only \
  --n 200 \
  --min-chars 500 \
  --max-pages-per-q 20
```

## Evaluation

The runner chooses the log directory from the system name and the gate flag:

- `logs/baseline/`
- `logs/agent_gate_on/` or `logs/agent_gate_off/`
- `logs/anchor/`

Each run writes `<unix_timestamp>_<tag>.jsonl` (one JSON object per question) and `<unix_timestamp>_<tag>_summary.csv`. CSV columns include `final_f`, `final_o`, `em`, `f1`, `total_tokens`, `latency_ms`, `rounds`, and `action`, plus `em_short` / `f1_short` when those fields exist. `final_f_ragas` is added only after a RAGAS evaluation.

```bash
python -m agentic_rag.eval.runner \
  --dataset data/crag_questions.jsonl \
  --system agent \
  --gate-on \
  --backend openai \
  --faith-report both \
  --ragas-eval-subset 20 \
  --n 200
```

`--faith-report` defaults to `fallback`. `--judge-policy` accepts `never` (the CLI default), `gray_zone`, and `always`. `--use-final-short` / `--no-use-final-short` toggles short-answer EM/F1; the CLI default is short-answer scoring on.

## License

MIT License. Copyright (c) 2025 Inas Farras. See [LICENSE](LICENSE).
