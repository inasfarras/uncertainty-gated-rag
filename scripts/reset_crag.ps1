param(
  [int]$N = 200,
  [int]$MaxPagesPerQ = 20,
  [string]$OutDir = "data\crag_corpus_html",
  [string]$QsFile = "data\crag_questions.jsonl",
  [string]$MetaFile = "data\crag_meta.jsonl",
  [string]$Artifacts = "artifacts\crag_faiss",
  [ValidateSet("openai","mock")][string]$Backend = "openai",
  [switch]$StaticOnly = $true,
  [switch]$FallbackSnippet = $true,
  [switch]$ForceDownload
)

$ErrorActionPreference = 'Stop'

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[ERR ] $msg" -ForegroundColor Red }

try {
  $ts = Get-Date -Format 'yyyyMMdd_HHmmss'

  # Ensure base folders
  New-Item -ItemType Directory -Force -Path 'data' | Out-Null
  New-Item -ItemType Directory -Force -Path 'artifacts' | Out-Null

  $dataBackup = Join-Path 'data' ("backup_" + $ts)
  $artBackup  = Join-Path 'artifacts' ("backup_" + $ts)
  New-Item -ItemType Directory -Force -Path $dataBackup | Out-Null
  New-Item -ItemType Directory -Force -Path $artBackup  | Out-Null

  Info "Backing up generated data → $dataBackup"
  foreach ($p in @('data\crag_questions.jsonl','data\crag_meta.jsonl','data\crag_corpus_html','data\crag_corpus')) {
    if (Test-Path $p) {
      Move-Item -Force -Path $p -Destination $dataBackup
    }
  }

  Info "Backing up artifacts → $artBackup"
  if (Test-Path $Artifacts) { Move-Item -Force -Path $Artifacts -Destination $artBackup }

  # Optional re-download of dataset
  $datasetBz2 = 'data\crag_task_1_and_2_dev_v4.jsonl.bz2'
  if ($ForceDownload -or -not (Test-Path $datasetBz2)) {
    Info "Downloading CRAG dataset (.bz2)"
    python scripts\crag_full_download.py
  } else {
    Info "Using existing dataset: $datasetBz2"
  }

  # Prepare corpus
  $prepArgs = @('scripts/prepare_crag_from_jsonl.py')
  if ($StaticOnly) { $prepArgs += '--static-only' }
  if ($FallbackSnippet) { $prepArgs += '--fallback-snippet' }
  $prepArgs += @('--n', $N.ToString(), '--max-pages-per-q', $MaxPagesPerQ.ToString(), `
                 '--out-dir', $OutDir, '--qs-file', $QsFile, '--meta-file', $MetaFile)

  Info "Preparing corpus → $OutDir"
  python @prepArgs

  # Ingest into FAISS
  Info "Ingesting corpus into FAISS → $Artifacts (backend=$Backend)"
  python -m agentic_rag.ingest.ingest --input $OutDir --out $Artifacts --backend $Backend

  # Summary
  $pages = (Get-ChildItem -Path $OutDir -Filter *.txt -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
  $qcount = (Get-Content $QsFile -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
  Info "Pages: $pages | Questions: $qcount"
  if (Test-Path (Join-Path $Artifacts 'index.faiss')) {
    $faiss = Get-Item (Join-Path $Artifacts 'index.faiss')
    $chunks = Join-Path $Artifacts 'chunks.parquet'
    $chunksOk = Test-Path $chunks
    $chunksLabel = if ($chunksOk) { 'yes' } else { 'no' }
    Info ("FAISS index: {0:N0} bytes | chunks.parquet: {1}" -f $faiss.Length, $chunksLabel)
  } else {
    Warn "FAISS index not found at $Artifacts"
  }

  Info "Done. You can now run the anchor evaluation."
  Write-Host "Example:" -ForegroundColor Green
  Write-Host "python -m agentic_rag.eval.runner --dataset $QsFile --n 30 --system anchor --backend $Backend --judge-policy gray_zone --override \"USE_HYBRID_SEARCH=True USE_RERANK=False MMR_LAMBDA=0.0 MAX_ROUNDS=2\"" -ForegroundColor Gray

} catch {
  Fail $_.Exception.Message
  exit 1
}
