# Rebuild CRAG Corpus with OFFLINE Embeddings (FREE)
# Uses sentence-transformers instead of OpenAI

param(
    [int]$MaxPagesPerQ = 80,
    [int]$N = 200,
    [int]$MinChars = 300,
    [string]$EmbedModel = "sentence-transformers/all-MiniLM-L6-v2"  # Fast default model
)

$ErrorActionPreference = 'Stop'

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Success($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

try {
    Info "Starting OFFLINE CRAG corpus rebuild..."
    Info "Using sentence-transformers (FREE, no API key needed)"
    Info "Model: $EmbedModel"
    Info ""

    # Check if corpus is already prepared
    if (-not (Test-Path "data\crag_corpus_html")) {
        Info "Corpus not found. Preparing corpus first..."

        python scripts\prepare_crag_from_jsonl.py `
            --src data\crag_task_1_and_2_dev_v4.jsonl.bz2 `
            --out-dir data\crag_corpus_html `
            --qs-file data\crag_questions.jsonl `
            --meta-file data\crag_meta.jsonl `
            --static-only `
            --n $N `
            --min-chars $MinChars `
            --max-pages-per-q $MaxPagesPerQ

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to prepare corpus"
        }

        Success "Corpus prepared"
    } else {
        Success "Corpus already exists: data\crag_corpus_html"
    }

    # Ingest with sentence-transformers
    Info ""
    Info "Ingesting corpus with OFFLINE embeddings..."
    Info "This is FREE and runs on your local machine"
    Info "Estimated time: 20-40 minutes (depends on CPU/GPU)"
    Info ""

    # Set embedding model
    $env:ST_EMBED_MODEL = $EmbedModel

    python -m agentic_rag.ingest.ingest `
        --input data\crag_corpus_html `
        --out artifacts\crag_faiss `
        --backend st

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to ingest corpus"
    }

    Success "Ingestion complete!"

    # Verify
    Info ""
    Info "Verifying embeddings..."
    python check_embeddings.py

    Info ""
    Success "=========================================="
    Success "OFFLINE CORPUS REBUILD COMPLETE!"
    Success "=========================================="
    Info ""
    Info "Next step: Test with new corpus"
    Info ""
    Info "python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3"
    Info ""
    Info "Note: Offline embeddings are 95-98% as good as OpenAI"
    Info "The bigger win is having full corpus coverage!"

} catch {
    Write-Host "[ERR] $_" -ForegroundColor Red
    exit 1
}
