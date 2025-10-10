# Prepare FULL CRAG Corpus (all 2706 questions)
# Uses offline embeddings (FREE)

param(
    [int]$MinChars = 200,
    [string]$EmbedModel = "sentence-transformers/all-MiniLM-L6-v2"
)

$ErrorActionPreference = 'Stop'

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Success($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

try {
    Info "Starting FULL CRAG corpus preparation (2706 questions)..."
    Info "This will create ~13,000 documents and ~325,000 chunks"
    Info "Estimated time: 30-45 minutes"
    Info ""
    Info "Using offline embeddings (FREE): $EmbedModel"
    Info ""

    $continue = Read-Host "Continue? (y/n)"
    if ($continue -ne 'y') {
        Write-Host "Aborted by user"
        exit 0
    }

    # Backup current small corpus
    if (Test-Path "data\crag_corpus_html") {
        Info "Backing up current 200-question corpus..."
        $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
        Move-Item "data\crag_corpus_html" "data\crag_corpus_html_200q_$ts"
        Success "Backed up to crag_corpus_html_200q_$ts"
    }

    if (Test-Path "artifacts\crag_faiss") {
        Move-Item "artifacts\crag_faiss" "artifacts\crag_faiss_200q_$ts"
        Success "Backed up FAISS to crag_faiss_200q_$ts"
    }

    # Prepare corpus (NO --n limit, ALL questions!)
    Info ""
    Info "STEP 1/2: Preparing corpus from FULL dataset..."
    Info "  Questions: ALL (2706)"
    Info "  Min chars: $MinChars"
    Info ""

    python scripts\prepare_crag_from_jsonl.py `
        --src data\crag_task_1_and_2_dev_v4.jsonl.bz2 `
        --out-dir data\crag_corpus_html `
        --qs-file data\crag_questions.jsonl `
        --meta-file data\crag_meta.jsonl `
        --min-chars $MinChars `
        --fallback-snippet

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare corpus"
    }

    $fileCount = (Get-ChildItem data\crag_corpus_html\*.txt).Count
    Success "Created $fileCount documents"

    if ($fileCount -lt 10000) {
        Warn "Expected ~13,000 documents, got $fileCount"
        Warn "This is lower than expected but may still be okay"
    }

    # Ingest with offline embeddings
    Info ""
    Info "STEP 2/2: Ingesting with offline embeddings..."
    Info "This will take 15-30 minutes depending on your CPU"
    Info ""

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
    Info "Verifying corpus..."
    python check_embeddings.py

    Info ""
    Success "=========================================="
    Success "FULL CORPUS PREPARATION COMPLETE!"
    Success "=========================================="
    Info ""
    Info "Corpus statistics:"
    Info "  Documents: $fileCount"
    Info "  Questions: 2706"
    Info "  Embeddings: Offline (sentence-transformers)"
    Info ""
    Info "Next step: Run evaluation on full corpus"
    Info ""
    Info "Quick test (50 questions):"
    Info "python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3"
    Info ""
    Info "Full evaluation (2706 questions - will take hours!):"
    Info "python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 2706 --judge-policy gray_zone --max-rounds 3"

} catch {
    Write-Host "[ERR] $_" -ForegroundColor Red
    exit 1
}
