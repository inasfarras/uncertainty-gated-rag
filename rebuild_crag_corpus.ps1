# Rebuild CRAG Corpus with Full Coverage
# This script backs up current data and rebuilds with more pages per question

param(
    [int]$MaxPagesPerQ = 80,  # Increase from 20 to 80 for better coverage
    [int]$N = 200,            # Number of questions
    [int]$MinChars = 300      # Minimum characters per page
)

$ErrorActionPreference = 'Stop'

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Success($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }

try {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'

    Info "Starting CRAG corpus rebuild process..."
    Info "This will:"
    Info "  1. Backup current incomplete corpus"
    Info "  2. Re-prepare corpus with $MaxPagesPerQ pages per question (was ~20)"
    Info "  3. Re-ingest with OpenAI embeddings"
    Info "  4. Verify chunk count"
    Info ""
    Info "Expected result: 80k-120k chunks (currently: 24k)"
    Info "Estimated time: 30-60 minutes"
    Info "Estimated cost: $5-10 in OpenAI API credits"
    Info ""

    # Confirm with user
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -ne 'y') {
        Warn "Aborted by user"
        exit 0
    }

    # ===== STEP 1: BACKUP =====
    Info ""
    Info "STEP 1/4: Backing up current data..."

    $backupDir = "artifacts\backup_$ts"
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

    if (Test-Path "artifacts\crag_faiss") {
        Info "  Backing up artifacts/crag_faiss..."
        Move-Item -Force "artifacts\crag_faiss" "$backupDir\crag_faiss"
        Success "  Backed up to $backupDir\crag_faiss"
    } else {
        Warn "  No existing FAISS index to backup"
    }

    if (Test-Path "data\crag_corpus_html") {
        Info "  Backing up data/crag_corpus_html..."
        Move-Item -Force "data\crag_corpus_html" "$backupDir\crag_corpus_html"
        Success "  Backed up to $backupDir\crag_corpus_html"
    } else {
        Warn "  No existing corpus to backup"
    }

    # ===== STEP 2: CHECK DATASET =====
    Info ""
    Info "STEP 2/4: Checking CRAG dataset..."

    $datasetPath = "data\crag_task_1_and_2_dev_v4.jsonl.bz2"
    if (-not (Test-Path $datasetPath)) {
        Warn "  Dataset not found. Downloading from HuggingFace..."
        python scripts\crag_full_download.py
        if ($LASTEXITCODE -ne 0) {
            Fail "Failed to download dataset"
            exit 1
        }
    } else {
        Success "  Dataset exists: $datasetPath"
    }

    # ===== STEP 3: PREPARE CORPUS =====
    Info ""
    Info "STEP 3/4: Preparing corpus (this will take 15-30 minutes)..."
    Info "  Max pages per question: $MaxPagesPerQ"
    Info "  Questions: $N"
    Info "  Min chars per page: $MinChars"

    python scripts\prepare_crag_from_jsonl.py `
        --src $datasetPath `
        --out-dir data\crag_corpus_html `
        --qs-file data\crag_questions.jsonl `
        --meta-file data\crag_meta.jsonl `
        --static-only `
        --n $N `
        --min-chars $MinChars `
        --max-pages-per-q $MaxPagesPerQ

    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to prepare corpus"
        exit 1
    }

    # Check if corpus was created
    $txtFiles = Get-ChildItem -Path "data\crag_corpus_html" -Filter "*.txt" -Recurse | Measure-Object
    Success "  Created $($txtFiles.Count) text files"

    if ($txtFiles.Count -lt 1000) {
        Warn "  Warning: Only $($txtFiles.Count) files created. Expected 5000+"
        Warn "  This might indicate an issue with corpus preparation"
    }

    # ===== STEP 4: INGEST WITH EMBEDDINGS =====
    Info ""
    Info "STEP 4/4: Ingesting corpus with OpenAI embeddings (15-30 minutes)..."
    Info "  This will cost approximately $5-10 in API credits"

    # Check if OpenAI API key is set
    if (-not $env:OPENAI_API_KEY) {
        Fail "OPENAI_API_KEY environment variable not set!"
        Info "Please set it with: `$env:OPENAI_API_KEY='your-key-here'"
        exit 1
    }

    python -m agentic_rag.ingest.ingest `
        --input data\crag_corpus_html `
        --out artifacts\crag_faiss `
        --backend openai

    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to ingest corpus"
        exit 1
    }

    # ===== STEP 5: VERIFY =====
    Info ""
    Info "STEP 5/5: Verifying embeddings..."

    python check_embeddings.py

    if ($LASTEXITCODE -ne 0) {
        Warn "Verification script failed, but corpus may still be valid"
    }

    # Check chunk count
    if (Test-Path "artifacts\crag_faiss\chunks.parquet") {
        $chunkCount = (python -c "import pandas as pd; df = pd.read_parquet('artifacts/crag_faiss/chunks.parquet'); print(len(df))")

        if ([int]$chunkCount -lt 50000) {
            Warn "Only $chunkCount chunks created (expected 80k-120k)"
            Warn "You may need to increase --max-pages-per-q further"
        } elseif ([int]$chunkCount -lt 80000) {
            Warn "$chunkCount chunks created (target: 80k+, but this might be acceptable)"
        } else {
            Success "$chunkCount chunks created - excellent coverage!"
        }
    }

    # ===== DONE =====
    Info ""
    Success "=========================================="
    Success "CRAG CORPUS REBUILD COMPLETE!"
    Success "=========================================="
    Info ""
    Info "Next steps:"
    Info "  1. Revert retrieval changes to original settings:"
    Info "     python revert_retrieval_changes.py"
    Info ""
    Info "  2. Test with new corpus (n=50):"
    Info "     python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3"
    Info ""
    Info "Expected improvements:"
    Info "  - Hallucinations: 32% → 18-22%"
    Info "  - Abstain: 28% → 20-25%"
    Info "  - Composite: 46 → 53-55"
    Info ""
    Info "Old data backed up to: $backupDir"

} catch {
    Fail "Error: $_"
    Fail "Stack trace: $($_.ScriptStackTrace)"
    exit 1
}
