param(
    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 500,

    [Parameter(Mandatory=$false)]
    [int]$TotalQuestions = 2706,

    [Parameter(Mandatory=$false)]
    [int]$StartFromBatch = 0  # Start from batch number (0-based)
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BAUG OFF Batched Experiment (N=2706)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set memory-safe batch sizes
$env:EMBED_BATCH_SIZE = "8"
$env:RERANK_BATCH_SIZE = "32"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  EMBED_BATCH_SIZE: 8"
Write-Host "  RERANK_BATCH_SIZE: 32"
Write-Host "  Batch size: $BatchSize questions per run"
Write-Host "  Total questions: $TotalQuestions"
Write-Host ""

# Create temp directory for batch files
$temp_dir = "data/temp_batches"
if (-not (Test-Path $temp_dir)) {
    New-Item -ItemType Directory -Path $temp_dir | Out-Null
}

# Load full dataset
Write-Host "Loading dataset..." -ForegroundColor Cyan
$all_questions = Get-Content "data/crag_questions_full.jsonl" | ForEach-Object { $_ }

# Calculate batches
$batches = @()
for ($i = 0; $i -lt $TotalQuestions; $i += $BatchSize) {
    $end = [Math]::Min($i + $BatchSize, $TotalQuestions)
    $batches += @{
        Start = $i
        End = $end
        Count = $end - $i
        Label = "$($i+1)-$end"
        BatchNum = [int]($i / $BatchSize)
        DatasetFile = "$temp_dir/batch_$($i)_$($end).jsonl"
    }
}

Write-Host "Batches to run: $($batches.Count)" -ForegroundColor Cyan
foreach ($batch in $batches) {
    $skip_marker = if ($batch.BatchNum -lt $StartFromBatch) { " (skip)" } else { "" }
    Write-Host "  - Batch $($batch.BatchNum): Questions $($batch.Label) ($($batch.Count) questions)$skip_marker"
}
Write-Host ""

# Create batch dataset files
Write-Host "Creating batch dataset files..." -ForegroundColor Cyan
foreach ($batch in $batches) {
    if ($batch.BatchNum -lt $StartFromBatch) {
        continue
    }

    $batch_questions = $all_questions[$batch.Start..($batch.End - 1)]
    # Use .NET to write file without BOM for compatibility with older PowerShell
    [System.IO.File]::WriteAllLines($batch.DatasetFile, $batch_questions)
    Write-Host "  Created: $($batch.DatasetFile) ($($batch.Count) questions)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Starting in 3 seconds... (Press Ctrl+C to cancel)" -ForegroundColor Yellow
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  STARTING BATCHED RUNS" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$completed_batches = @()
$failed_batches = @()
$log_files = @()

foreach ($batch in $batches) {
    if ($batch.BatchNum -lt $StartFromBatch) {
        Write-Host "Skipping batch $($batch.BatchNum) (already completed)" -ForegroundColor Gray
        continue
    }

    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  Batch $($batch.BatchNum): Questions $($batch.Label)" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Started: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray

    # Clear GPU memory
    Write-Host "Clearing GPU memory..." -ForegroundColor Yellow
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 5

    # Set environment variables
    $env:EMBED_BATCH_SIZE = "8"
    $env:RERANK_BATCH_SIZE = "32"

    try {
        Write-Host "Command: python scripts/run_eval_with_judge.py --dataset $($batch.DatasetFile) --system anchor -- --n $($batch.Count) --gate-off" -ForegroundColor Gray
        Write-Host ""

        python scripts/run_eval_with_judge.py `
            --dataset $batch.DatasetFile `
            --system anchor `
            -- `
            --n $batch.Count `
            --gate-off

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✓ Batch $($batch.BatchNum) completed successfully!" -ForegroundColor Green
            $completed_batches += $batch

            # Find the log file that was just created
            $latest_log = Get-ChildItem "logs/anchor/*_anchor.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latest_log) {
                $log_files += @{
                    Batch = $batch.BatchNum
                    File = $latest_log.FullName
                    Questions = $batch.Count
                }
                Write-Host "  Log file: $($latest_log.Name)" -ForegroundColor Gray
            }

            # Wait to avoid rate limits and GPU heat
            if ($batch.BatchNum -lt ($batches.Count - 1)) {
                Write-Host "Waiting 60 seconds before next batch..." -ForegroundColor Gray
                Start-Sleep -Seconds 60
            }
        } else {
            throw "Process exited with code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host ""
        Write-Host "✗ Batch $($batch.BatchNum) FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $failed_batches += $batch

        Write-Host ""
        $retry = Read-Host "Retry this batch? (y/n)"
        if ($retry -eq 'y') {
            # Re-run this batch
            continue
        }

        $continueRun = Read-Host "Continue with next batch? (y/n)"
        if ($continueRun -ne 'y') {
            Write-Host "Stopping..." -ForegroundColor Red
            break
        }
    }

    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BATCHED EXPERIMENT SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Completed batches: $($completed_batches.Count) / $($batches.Count - $StartFromBatch)" -ForegroundColor Green
foreach ($batch in $completed_batches) {
    Write-Host "  ✓ Batch $($batch.BatchNum): $($batch.Label)"
}

if ($failed_batches.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed batches: $($failed_batches.Count)" -ForegroundColor Red
    foreach ($batch in $failed_batches) {
        Write-Host "  ✗ Batch $($batch.BatchNum): $($batch.Label)"
    }
}

Write-Host ""
Write-Host "Log files created:" -ForegroundColor Yellow
foreach ($log in $log_files) {
    Write-Host "  Batch $($log.Batch): $($log.File)"
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. After all batches complete, merge logs:"
Write-Host "     .\merge_batch_logs.ps1"
Write-Host "  2. Generate complete report"
Write-Host ""

# Save batch info for merging later
$batch_info = @{
    Batches = $completed_batches
    LogFiles = $log_files
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}
$batch_info | ConvertTo-Json | Out-File "logs/anchor/batch_info.json" -Encoding UTF8
