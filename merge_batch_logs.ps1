param(
    [Parameter(Mandatory=$false)]
    [string[]]$LogFiles = @(),

    [Parameter(Mandatory=$false)]
    [string]$OutputName = "baug_off_merged"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Merging BAUG OFF Batch Logs" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# If no log files specified, try to find them from batch_info.json
if ($LogFiles.Count -eq 0) {
    Write-Host "Looking for batch info..." -ForegroundColor Yellow

    if (Test-Path "logs/anchor/batch_info.json") {
        $batch_info = Get-Content "logs/anchor/batch_info.json" | ConvertFrom-Json
        $LogFiles = $batch_info.LogFiles | ForEach-Object { $_.File }
        Write-Host "Found $($LogFiles.Count) log files from batch_info.json" -ForegroundColor Green
    }

    # If still no files, ask user to select them
    if ($LogFiles.Count -eq 0) {
        Write-Host "No batch info found. Please select batch log files:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Recent anchor log files:" -ForegroundColor Cyan
        $recent_logs = Get-ChildItem "logs/anchor/*_anchor.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 10
        for ($i = 0; $i -lt $recent_logs.Count; $i++) {
            $log = $recent_logs[$i]
            Write-Host "  [$i] $($log.Name) ($(($log.Length/1KB).ToString('F1')) KB, $($log.LastWriteTime.ToString('HH:mm:ss')))"
        }
        Write-Host ""
        $selection = Read-Host "Enter file numbers to merge (comma-separated, e.g., 0,1,2)"
        $indices = $selection.Split(',') | ForEach-Object { [int]$_.Trim() }
        $LogFiles = $indices | ForEach-Object { $recent_logs[$_].FullName }
    }
}

if ($LogFiles.Count -eq 0) {
    Write-Host "No log files to merge!" -ForegroundColor Red
    exit 1
}

Write-Host "Merging $($LogFiles.Count) log files:" -ForegroundColor Cyan
foreach ($log in $LogFiles) {
    $file_info = Get-Item $log
    Write-Host "  - $($file_info.Name) ($([Math]::Round($file_info.Length/1KB, 1)) KB)"
}
Write-Host ""

# Read and merge all logs
Write-Host "Reading log files..." -ForegroundColor Yellow
$all_records = @()
$total_lines = 0

foreach ($log_file in $LogFiles) {
    Write-Host "  Processing: $log_file" -ForegroundColor Gray
    $lines = Get-Content $log_file
    $total_lines += $lines.Count

    foreach ($line in $lines) {
        try {
            $record = $line | ConvertFrom-Json
            $all_records += $record
        } catch {
            Write-Host "    Warning: Failed to parse line (skipping)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Total records parsed: $($all_records.Count) from $total_lines lines" -ForegroundColor Cyan

# Check for duplicates
$unique_questions = ($all_records | Select-Object -Property question -Unique).Count
Write-Host "Unique questions: $unique_questions" -ForegroundColor Cyan

if ($unique_questions -ne $all_records.Count) {
    Write-Host ""
    Write-Host "⚠ Warning: Found duplicate questions!" -ForegroundColor Yellow
    Write-Host "  Total records: $($all_records.Count)"
    Write-Host "  Unique questions: $unique_questions"
    Write-Host "  Duplicates: $($all_records.Count - $unique_questions)"
    Write-Host ""
    $remove_dupes = Read-Host "Remove duplicates? (y/n)"

    if ($remove_dupes -eq 'y') {
        # Keep first occurrence of each question
        $seen = @{}
        $deduplicated = @()
        foreach ($record in $all_records) {
            if (-not $seen.ContainsKey($record.question)) {
                $deduplicated += $record
                $seen[$record.question] = $true
            }
        }
        $all_records = $deduplicated
        Write-Host "Removed duplicates. New count: $($all_records.Count)" -ForegroundColor Green
    }
}

# Write merged log
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$output_log = "logs/anchor/${timestamp}_anchor.jsonl"

Write-Host ""
Write-Host "Writing merged log..." -ForegroundColor Yellow
foreach ($record in $all_records) {
    ($record | ConvertTo-Json -Compress -Depth 10) | Add-Content -Path $output_log -Encoding UTF8
}

Write-Host "✓ Merged log saved: $output_log" -ForegroundColor Green
Write-Host "  Records: $($all_records.Count)" -ForegroundColor Gray
Write-Host ""

# Now run judge and benchmarks on the merged file
Write-Host "Running judge evaluation on merged log..." -ForegroundColor Cyan
Write-Host ""

try {
    python -m eval.cli_judge `
        -p $output_log `
        --gold data/crag_questions_full.jsonl `
        --require-citation false

    $judge_file = $output_log -replace '\.jsonl$', '_judge_gold.jsonl'

    if (Test-Path $judge_file) {
        Write-Host "✓ Judge evaluation complete" -ForegroundColor Green
        Write-Host ""

        Write-Host "Computing benchmarks..." -ForegroundColor Cyan
        python -m eval.compute_benchmarks `
            --input $judge_file `
            --out ($output_log -replace '\.jsonl$', '_benchmarks.jsonl')

        Write-Host "✓ Benchmarks complete" -ForegroundColor Green
        Write-Host ""

        # Generate summary
        Write-Host "Generating summary..." -ForegroundColor Cyan
        $benchmark_file = $output_log -replace '\.jsonl$', '_benchmarks.jsonl'

        if (Test-Path $benchmark_file) {
            $bench_data = Get-Content $benchmark_file | ForEach-Object {
                $line = $_ -replace ', "F1_short": [^,}]+', ''
                $line | ConvertFrom-Json
            }

            $abstain_rate = ($all_records | Where-Object { $_.abstain -eq 1 }).Count / $all_records.Count
            $avg_faithfulness = ($bench_data | Measure-Object -Property final_f -Average).Average
            $avg_overlap = ($bench_data | Measure-Object -Property overlap -Average).Average
            $avg_em = ($bench_data | Measure-Object -Property em -Average).Average
            $avg_f1 = ($bench_data | Measure-Object -Property f1 -Average).Average
            $avg_tokens = ($all_records | Measure-Object -Property total_tokens -Average).Average
            $avg_latency = ($all_records | Measure-Object -Property p50_latency_ms -Average).Average
            $avg_rounds = ($all_records | Measure-Object -Property rounds -Average).Average
            $source_batches = $LogFiles | ForEach-Object { "- $((Get-Item $_).Name)" } | Out-String

            $summary = "# BAUG OFF Merged Experiment Summary`n`n"
            $summary += "- Total Questions: $($all_records.Count)`n"
            $summary += "- Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"
            $summary += "- Log File: $output_log`n`n"
            $summary += "## Metrics`n`n"
            $summary += "| Metric | Value |`n"
            $summary += "|--------|-------|`n"
            $summary += "| Count | $($bench_data.Count) |`n"
            $summary += "| Avg Faithfulness | $($avg_faithfulness.ToString('F3')) |`n"
            $summary += "| Avg Overlap | $($avg_overlap.ToString('F3')) |`n"
            $summary += "| Avg EM | $($avg_em.ToString('F3')) |`n"
            $summary += "| Avg F1 | $($avg_f1.ToString('F3')) |`n"
            $summary += "| Abstain Rate | $($abstain_rate.ToString('F3')) |`n"
            $summary += "| Avg Tokens | $($avg_tokens.ToString('F0')) |`n"
            $summary += "| P50 Latency (ms) | $($avg_latency.ToString('F0')) |`n"
            $summary += "| Avg Rounds | $($avg_rounds.ToString('F1')) |`n`n"
            $summary += "## Source Batches`n`n"
            $summary += $source_batches

            $summary_file = $output_log -replace '\.jsonl$', '_summary.md'
            $summary | Out-File -FilePath $summary_file -Encoding UTF8

            Write-Host "✓ Summary saved: $summary_file" -ForegroundColor Green
            Write-Host ""
            Get-Content $summary_file
        }
    } else {
        Write-Host "⚠ Judge file not found" -ForegroundColor Yellow
    }

} catch {
    Write-Host "✗ Error during judge/benchmark: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  MERGE COMPLETE!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Merged log file: $output_log" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step: Generate full report with:" -ForegroundColor Yellow
Write-Host "  .\regenerate_full_report.ps1" -ForegroundColor Cyan
Write-Host ""
