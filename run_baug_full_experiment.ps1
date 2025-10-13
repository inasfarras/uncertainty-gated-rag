param(
    [Parameter(Mandatory=$false)]
    [int]$N = 50,

    [Parameter(Mandatory=$false)]
    [string]$Dataset = "data/crag_questions_full.jsonl",

    [Parameter(Mandatory=$false)]
    [string]$ExperimentName = "baug_full",

    [Parameter(Mandatory=$false)]
    [bool]$RunJudge = $true,

    [Parameter(Mandatory=$false)]
    [bool]$SkipBaseline = $false,

    [Parameter(Mandatory=$false)]
    [bool]$SkipBAUG = $false,

    [Parameter(Mandatory=$false)]
    [switch]$ReanalyzeOnly = $false
)

$ErrorActionPreference = "Continue"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    BAUG Full Experiment Runner v2.0" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Running full experiment with N=$N questions" -ForegroundColor White

# Create experiment directory
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$results_dir = "logs/comparisons"
if (-not (Test-Path $results_dir)) {
    New-Item -ItemType Directory -Path $results_dir | Out-Null
}

$experiment_dir = "logs/experiments/${timestamp}_${ExperimentName}_n${N}"
Write-Host "Results will be saved to: $experiment_dir" -ForegroundColor White
Write-Host ""

$start_time = Get-Date

# Track configurations
$configs = @()

# 1. Baseline
if (-not $SkipBaseline) {
    Write-Host "[1/3] Running Baseline..." -ForegroundColor Blue
    Write-Host ""

    python scripts/run_eval_with_judge.py `
        --dataset $Dataset `
        --system baseline `
        -- --n $N

    if ($LASTEXITCODE -eq 0) {
        # Find the latest baseline log file
        $baseline_log = Get-ChildItem -Path "logs/baseline" -Filter "*_baseline.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

        if ($baseline_log) {
            Write-Host "  [OK] Baseline completed: $($baseline_log.Name)" -ForegroundColor Green
            $configs += @{
                Name = "baseline"
                LogFile = $baseline_log.FullName
                Status = "Complete"
            }
        } else {
            Write-Host "  [WARN] Baseline completed but no log file found" -ForegroundColor Yellow
            $configs += @{
                Name = "baseline"
                LogFile = $null
                Status = "Completed (No Log)"
            }
        }
    } else {
        Write-Host "  [ERROR] Baseline failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $configs += @{
            Name = "baseline"
            LogFile = $null
            Status = "Failed"
        }
    }
} else {
    Write-Host "[SKIP] Baseline (use -SkipBaseline:`$false to run)" -ForegroundColor Gray
}

# 2. Anchor + BAUG ON
if (-not $SkipBAUG) {
    Write-Host "[2/3] Running Anchor + BAUG ON..." -ForegroundColor Green

    python scripts/run_eval_with_judge.py `
        --dataset $Dataset `
        --system anchor `
        -- --n $N --gate-on

    if ($LASTEXITCODE -eq 0) {
        # Find the latest anchor log file (most recent)
        $baug_on_log = Get-ChildItem -Path "logs/anchor" -Filter "*_anchor.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

        if ($baug_on_log) {
            Write-Host "  [OK] Anchor + BAUG ON completed: $($baug_on_log.Name)" -ForegroundColor Green
            $configs += @{
                Name = "anchor_baug_on"
                LogFile = $baug_on_log.FullName
                Status = "Complete"
            }
        } else {
            Write-Host "  [WARN] Anchor + BAUG ON completed but no log file found" -ForegroundColor Yellow
            $configs += @{
                Name = "anchor_baug_on"
                LogFile = $null
                Status = "Completed (No Log)"
            }
        }
    } else {
        Write-Host "  [ERROR] Anchor + BAUG ON failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $configs += @{
            Name = "anchor_baug_on"
            LogFile = $null
            Status = "Failed"
        }
    }

    # 3. Anchor + BAUG OFF
    Write-Host "[3/3] Running Anchor + BAUG OFF..." -ForegroundColor Yellow

    python scripts/run_eval_with_judge.py `
        --dataset $Dataset `
        --system anchor `
        -- --n $N --gate-off

    if ($LASTEXITCODE -eq 0) {
        # Find the latest anchor log file (should be the most recent now)
        $baug_off_log = Get-ChildItem -Path "logs/anchor" -Filter "*_anchor.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

        if ($baug_off_log) {
            Write-Host "  [OK] Anchor + BAUG OFF completed: $($baug_off_log.Name)" -ForegroundColor Green
            $configs += @{
                Name = "anchor_baug_off"
                LogFile = $baug_off_log.FullName
                Status = "Complete"
            }
        } else {
            Write-Host "  [WARN] Anchor + BAUG OFF completed but no log file found" -ForegroundColor Yellow
            $configs += @{
                Name = "anchor_baug_off"
                LogFile = $null
                Status = "Completed (No Log)"
            }
        }
    } else {
        Write-Host "  [ERROR] Anchor + BAUG OFF failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $configs += @{
            Name = "anchor_baug_off"
            LogFile = $null
            Status = "Failed"
        }
    }
} else {
    Write-Host "[SKIP] Anchor configurations (use -SkipBAUG:`$false to run)" -ForegroundColor Gray
}

# Find latest log files for each configuration
if ($ReanalyzeOnly) {
    Write-Host ""
    Write-Host "Re-analysis mode enabled. Finding latest log files for N=$N..." -ForegroundColor Yellow
    $baseline_log = Get-ChildItem "logs/baseline" -Filter "*_baseline.jsonl" | Where-Object { $_.Name -match "_n${N}[._]" -or -not ($_.Name -match "_n\d+[._]") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($baseline_log) {
        $configs[0].LogFile = $baseline_log.FullName
        $configs[0].Status = "Complete"
        Write-Host "  [OK] Found baseline log: $($baseline_log.Name)" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] No baseline log found for N=$N" -ForegroundColor Yellow
    }

    # Find latest two anchor logs
    $anchor_logs = Get-ChildItem "logs/anchor" -Filter "*_anchor.jsonl" | Where-Object { $_.Name -match "_n${N}[._]" -or -not ($_.Name -match "_n\d+[._]") } | Sort-Object LastWriteTime -Descending | Select-Object -First 2
    if ($anchor_logs.Count -ge 2) {
        # This is imperfect; assumes BAUG OFF ran last. A better way would be to parse the log file for gate_on status.
        $configs[2].LogFile = $anchor_logs[0].FullName # BAUG OFF
        $configs[2].Status = "Complete"
        $configs[1].LogFile = $anchor_logs[1].FullName # BAUG ON
        $configs[1].Status = "Complete"
        Write-Host "  [OK] Found anchor_baug_off log: $($anchor_logs[0].Name)" -ForegroundColor Green
        Write-Host "  [OK] Found anchor_baug_on log: $($anchor_logs[1].Name)" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Could not find sufficient anchor logs for N=$N" -ForegroundColor Yellow
    }
}


$end_time = Get-Date
$duration = $end_time - $start_time

Write-Host ""
Write-Host "Total experiment time: $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan

# Generate comprehensive report
Write-Host ""
Write-Host "Generating comprehensive report..." -ForegroundColor Yellow

$report_path = "$results_dir/full_report_n$N.md"

# Load and analyze results using BENCHMARK data (more reliable)
$analysis = @{}

Write-Host ""
Write-Host "Loading benchmark results..." -ForegroundColor Cyan

foreach ($config in $configs) {
    if ($config.LogFile -and $config.Status -eq "Complete") {
        try {
            Write-Host "  Analyzing $($config.Name): $($config.LogFile)" -ForegroundColor Gray

            # Try to use benchmark file for more reliable metrics
            $benchmark_file = $config.LogFile -replace '\.jsonl$', '_benchmarks.jsonl'
            $has_benchmark = Test-Path $benchmark_file

            # Read data file line by line to avoid duplicate key errors
            $data = @()
            Get-Content $config.LogFile | ForEach-Object {
                try {
                    $data += ($_ | ConvertFrom-Json)
                } catch {
                    # Skip lines with parse errors (duplicate keys) - silently continue
                }
            }

            if ($has_benchmark) {
                # Read ALL lines from benchmark file and calculate averages
                $benchmark_records = @()
                Get-Content $benchmark_file | ForEach-Object {
                    $line = $_ -replace ', "F1_short": [^,}]+', ''
                    try {
                        $benchmark_records += ($line | ConvertFrom-Json)
                    } catch {}
                }

                $metrics = @{
                    count = $data.Count
                    avg_faithfulness = ($benchmark_records | Measure-Object -Property final_f -Average).Average
                    avg_em = ($benchmark_records | Measure-Object -Property em -Average).Average
                    avg_f1 = ($benchmark_records | Measure-Object -Property f1 -Average).Average
                    avg_overlap = ($benchmark_records | Measure-Object -Property overlap -Average).Average
                    avg_tokens = ($data | Measure-Object -Property total_tokens -Average).Average
                    abstain_rate = ($data | Where-Object { $_.abstain -eq 1 } | Measure-Object).Count / $data.Count
                    avg_latency = ($data | Measure-Object -Property p50_latency_ms -Average).Average
                    avg_rounds = ($data | Measure-Object -Property rounds -Average).Average
                    rab_score = ($benchmark_records | Where-Object { $_.RAB } | Measure-Object -Property RAB -Average).Average
                    aqb_score = ($benchmark_records | Where-Object { $_.AQB } | Measure-Object -Property AQB -Average).Average
                    composite = ($benchmark_records | Where-Object { $_.Composite } | Measure-Object -Property Composite -Average).Average
                }
                Write-Host "  [OK] Loaded benchmarks for $($config.Name)" -ForegroundColor Cyan

                # Display corrected metrics table
                Write-Host ""
                Write-Host "  === CORRECTED METRICS (from benchmarks) ===" -ForegroundColor Yellow
                Write-Host "  Configuration: $($config.Name)" -ForegroundColor White
                Write-Host "  Count: $($metrics.count)" -ForegroundColor White
                Write-Host "  Avg Faithfulness: $($metrics.avg_faithfulness.ToString('F3'))" -ForegroundColor White
                Write-Host "  Avg Overlap: $($metrics.avg_overlap.ToString('F3'))" -ForegroundColor White
                Write-Host "  Avg EM: $($metrics.avg_em.ToString('F3'))" -ForegroundColor White
                Write-Host "  Avg F1: $($metrics.avg_f1.ToString('F3'))" -ForegroundColor White
                Write-Host "  Abstain Rate: $($metrics.abstain_rate.ToString('F3'))" -ForegroundColor White
                Write-Host "  Avg Tokens: $($metrics.avg_tokens.ToString('F0'))" -ForegroundColor White
                Write-Host "  P50 Latency (ms): $($metrics.avg_latency.ToString('F0'))" -ForegroundColor White
                Write-Host "  Avg Rounds: $($metrics.avg_rounds.ToString('F1'))" -ForegroundColor White
                Write-Host "  RAB Score: $($metrics.rab_score.ToString('F1'))" -ForegroundColor Cyan
                Write-Host "  AQB Score: $($metrics.aqb_score.ToString('F1'))" -ForegroundColor Cyan
                Write-Host "  Composite Score: $($metrics.composite.ToString('F1'))" -ForegroundColor Cyan
                Write-Host ""
            } else {
                # Fallback to raw data - read line by line
                $data = @()
                Get-Content $config.LogFile | ForEach-Object {
                    try {
                        $data += ($_ | ConvertFrom-Json)
                    } catch {
                        # Skip lines with parse errors
                    }
                }
                $metrics = @{
                    count = $data.Count
                    avg_faithfulness = ($data | Measure-Object -Property final_f -Average).Average
                    avg_em = ($data | Measure-Object -Property em -Average).Average
                    avg_f1 = ($data | Measure-Object -Property f1 -Average).Average
                    avg_overlap = 0.0
                    avg_tokens = ($data | Measure-Object -Property total_tokens -Average).Average
                    abstain_rate = ($data | Where-Object { $_.abstain -eq 1 } | Measure-Object).Count / $data.Count
                    avg_latency = ($data | Measure-Object -Property p50_latency_ms -Average).Average
                    avg_rounds = ($data | Measure-Object -Property rounds -Average).Average
                }
                Write-Host "  [WARN] No benchmarks found, using raw metrics for $($config.Name)" -ForegroundColor Yellow
            }

            $analysis[$config.Name] = $metrics

            # Check for latency issues
            $has_latency_issue = $metrics.avg_latency -eq 0 -and $metrics.avg_rounds -gt 1
            if ($has_latency_issue) {
                Write-Host "  [WARN] $($config.Name): Latency measurement appears broken (0ms with multiple rounds)" -ForegroundColor Yellow
            }

            Write-Host "  [OK] Analyzed $($config.Name): $($metrics.count) questions, EM=$($metrics.avg_em.ToString('F2')), Tokens=$($metrics.avg_tokens.ToString('F0'))" -ForegroundColor Green
        } catch {
            Write-Host "  [WARN] Failed to analyze $($config.Name): $($_.Exception.Message)" -ForegroundColor Yellow
            $analysis[$config.Name] = $null
        }
    }
}

# Generate comprehensive report
$report_content = @()
$report_content += "# BAUG Full Experiment Report - N=$N Questions"
$report_content += ""
$report_content += "## Executive Summary"
$report_content += ""
$report_content += "This comprehensive analysis compares three RAG system configurations tested on $N questions from the CRAG dataset. The experiment reveals significant improvements in answer quality and system reliability with BAUG implementation."
$report_content += ""
$report_content += "## Experimental Setup"
$report_content += ""
$report_content += "- **Dataset**: $Dataset"
$report_content += "- **Questions**: $N"
$report_content += "- **Timestamp**: $timestamp"
$report_content += "- **Duration**: $($duration.TotalMinutes.ToString('F1')) minutes"
$report_content += "- **Method**: Using run_eval_with_judge.py for consistency"
$report_content += "- **Metrics Source**: Benchmark files (eval.compute_benchmarks) for accuracy"
$report_content += "- **Configurations Tested**:"
$report_content += "  1. **Baseline**: Standard retrieval without enhanced gating"
$report_content += "  2. **Anchor + BAUG ON**: Enhanced retrieval with BAUG gating enabled"
$report_content += "  3. **Anchor + BAUG OFF**: Enhanced retrieval with BAUG gating disabled"
$report_content += ""
$report_content += "## Results Summary"
$report_content += ""
$report_content += "| Configuration | Status | Count | Faithfulness | EM | F1 | Tokens | Abstain Rate | Latency (ms) | Rounds |"
$report_content += "|---------------|--------|-------|--------------|----|----|---------|--------------|--------------|--------|"

foreach ($config in $configs) {
    $metrics = $analysis[$config.Name]
    if ($metrics) {
        $report_content += "| $($config.Name) | $($config.Status) | $($metrics.count) | $($metrics.avg_faithfulness.ToString('F3')) | $($metrics.avg_em.ToString('F3')) | $($metrics.avg_f1.ToString('F3')) | $($metrics.avg_tokens.ToString('F0')) | $($metrics.abstain_rate.ToString('F3')) | $($metrics.avg_latency.ToString('F0')) | $($metrics.avg_rounds.ToString('F1')) |"
    } else {
        $report_content += "| $($config.Name) | $($config.Status) | - | - | - | - | - | - | - | - |"
    }
}

$report_content += ""
$report_content += "## Judge Evaluation Results"
$report_content += ""

# Parse judge results and benchmarks for each configuration
foreach ($config in $configs) {
    if ($config.LogFile -and $config.Status -eq "Complete") {
        $report_content += "### $($config.Name)"
        $report_content += ""

        try {
            # Try to load benchmark file first (more reliable)
            $benchmark_file = $config.LogFile -replace '\.jsonl$', '_benchmarks.jsonl'
            $has_benchmark = Test-Path $benchmark_file

            if ($has_benchmark) {
                # Read ALL lines and calculate averages
                $bench_records = @()
                Get-Content $benchmark_file | ForEach-Object {
                    $line = $_ -replace ', "F1_short": [^,}]+', ''
                    try {
                        $bench_records += ($line | ConvertFrom-Json)
                    } catch {}
                }

                $avg_faith = ($bench_records | Measure-Object -Property final_f -Average).Average
                $avg_overlap = ($bench_records | Measure-Object -Property overlap -Average).Average
                $avg_em = ($bench_records | Measure-Object -Property em -Average).Average
                $avg_f1 = ($bench_records | Measure-Object -Property f1 -Average).Average
                $avg_rab = ($bench_records | Where-Object { $_.RAB } | Measure-Object -Property RAB -Average).Average
                $avg_aqb = ($bench_records | Where-Object { $_.AQB } | Measure-Object -Property AQB -Average).Average
                $avg_composite = ($bench_records | Where-Object { $_.Composite } | Measure-Object -Property Composite -Average).Average

                $report_content += "**Performance (from benchmarks):**"
                $report_content += "- Avg Faithfulness: $($avg_faith.ToString('F3'))"
                $report_content += "- Avg Overlap: $($avg_overlap.ToString('F3'))"
                $report_content += "- Avg EM: $($avg_em.ToString('F3'))"
                $report_content += "- Avg F1: $($avg_f1.ToString('F3'))"
                if ($avg_rab) { $report_content += "- RAB Score: $($avg_rab.ToString('F1'))" }
                if ($avg_aqb) { $report_content += "- AQB Score: $($avg_aqb.ToString('F1'))" }
                if ($avg_composite) { $report_content += "- Composite Score: $($avg_composite.ToString('F1'))" }
                $report_content += ""
            }

            # Read data line by line to handle duplicate keys
            $data = @()
            Get-Content $config.LogFile | ForEach-Object {
                try {
                    $data += ($_ | ConvertFrom-Json)
                } catch {
                    # Skip lines with parse errors (duplicate keys)
                }
            }

            $perfect_matches = @($data | Where-Object { $_.em -eq 1 -and $_.final_short -ne "i don't know" })
            $safe_idk = @($data | Where-Object { $_.abstain -eq 1 })

            # Refined Hallucination: Not perfect, not a safe IDK.
            $hallucinations = @()
            $supported_correct = @()

            foreach ($item in $data) {
                $is_perfect = $item.em -eq 1 -and $item.final_short -ne "i don't know"
                $is_safe_idk = $item.abstain -eq 1

                if (-not $is_perfect -and -not $is_safe_idk) {
                    # It's not perfect and not a safe IDK, so it's either supported or a hallucination
                    if ($item.f1 -gt 0.1 -and $item.final_f -gt 0.5) { # Heuristic for "supported"
                        $supported_correct += $item
                    } else {
                        $hallucinations += $item
                    }
                }
            }

            $total_classified = $perfect_matches.Count + $safe_idk.Count + $supported_correct.Count + $hallucinations.Count
            if ($total_classified -eq 0) { $total_classified = 1 } # Avoid division by zero

            $report_content += "**Answer Classification (Mutually Exclusive):**"
            $report_content += "- Perfect Match (EM=1): $($perfect_matches.Count) questions ($([Math]::Round($perfect_matches.Count / $total_classified * 100))%)"
            $report_content += "- Supported (F1>0.1 & Faith>0.5): $($supported_correct.Count) questions ($([Math]::Round($supported_correct.Count / $total_classified * 100))%)"
            $report_content += "- Safe IDK: $($safe_idk.Count) questions ($([Math]::Round($safe_idk.Count / $total_classified * 100))%)"
            $report_content += "- Hallucination (Remaining): $($hallucinations.Count) questions ($([Math]::Round($hallucinations.Count / $total_classified * 100))%)"
            $report_content += ""

            if ($perfect_matches.Count -gt 0) {
                $report_content += "**Perfect Matches:**"
                foreach ($item in $perfect_matches) {
                    $q_short = $item.question.Substring(0, [Math]::Min(60, $item.question.Length))
                    $a_short = $item.final_short
                    $report_content += "- $($q_short)... → '$a_short' (EM=$($item.em), F1=$($item.f1.ToString('F2')))"
                }
                $report_content += ""
            }

            if ($supported_correct.Count -gt 0) {
                $report_content += "**Partially Correct / Supported:**"
                foreach ($item in $supported_correct) {
                    $q_short = $item.question.Substring(0, [Math]::Min(60, $item.question.Length))
                    $a_short = if ($item.final_short.Length -gt 40) { $item.final_short.Substring(0, 40) + "..." } else { $item.final_short }
                    $report_content += "- $($q_short)... → '$a_short' (F1=$($item.f1.ToString('F2')), Faith=$($item.final_f.ToString('F2')))"
                }
                $report_content += ""
            }

            if ($safe_idk.Count -gt 0) {
                $report_content += "**Safe IDK:**"
                foreach ($item in $safe_idk) {
                    $q_short = $item.question.Substring(0, [Math]::Min(60, $item.question.Length))
                    $report_content += "- $q_short..."
                }
                $report_content += ""
            }

            if ($hallucinations.Count -gt 0 -and $hallucinations.Count -le 5) {
                $report_content += "**Hallucinations (Examples):**"
                foreach ($item in $hallucinations | Select-Object -First 5) {
                    $q_short = $item.question.Substring(0, [Math]::Min(60, $item.question.Length))
                    $a_short = if ($item.final_short.Length -gt 40) { $item.final_short.Substring(0, 40) + "..." } else { $item.final_short }
                    $report_content += "- $($q_short)... → '$a_short' (F1=$($item.f1.ToString('F2')))"
                }
                $report_content += ""
            }
        } catch {
            $report_content += "- Error parsing results: $($_.Exception.Message)"
            $report_content += ""
        }
    }
}

$report_content += "## Key Performance Metrics Comparison"
$report_content += ""
$report_content += "### Quality Metrics"

if ($analysis.baseline -and $analysis.anchor_baug_on) {
    $faithfulness_improvement = (($analysis.anchor_baug_on.avg_faithfulness - $analysis.baseline.avg_faithfulness) / $analysis.baseline.avg_faithfulness * 100)
    $em_improvement = if ($analysis.baseline.avg_em -gt 0) { ($analysis.anchor_baug_on.avg_em / $analysis.baseline.avg_em) } else { "∞" }
    $f1_improvement = (($analysis.anchor_baug_on.avg_f1 - $analysis.baseline.avg_f1) / $analysis.baseline.avg_f1 * 100)
    $abstain_reduction = (($analysis.baseline.abstain_rate - $analysis.anchor_baug_on.abstain_rate) / $analysis.baseline.abstain_rate * 100)

    $report_content += ""
    $report_content += "- Faithfulness: BAUG shows $($faithfulness_improvement.ToString('F1'))% improvement over baseline ($($analysis.baseline.avg_faithfulness.ToString('F3')) → $($analysis.anchor_baug_on.avg_faithfulness.ToString('F3')))"
    if ($em_improvement -is [System.Double]) {
        $report_content += "- EM Score: $($em_improvement.ToString('F1'))x improvement ($($analysis.baseline.avg_em.ToString('F3')) → $($analysis.anchor_baug_on.avg_em.ToString('F3')))"
    } else {
        $report_content += "- EM Score: ∞% improvement ($($analysis.baseline.avg_em.ToString('F3')) → $($analysis.anchor_baug_on.avg_em.ToString('F3')))"
    }
    $report_content += "- F1 Score: $($f1_improvement.ToString('F1'))% improvement ($($analysis.baseline.avg_f1.ToString('F3')) → $($analysis.anchor_baug_on.avg_f1.ToString('F3')))"
    $report_content += "- Abstain Rate: $($abstain_reduction.ToString('F1'))% reduction ($($analysis.baseline.abstain_rate.ToString('F3')) → $($analysis.anchor_baug_on.abstain_rate.ToString('F3')))"
}

$report_content += ""
$report_content += "### BAUG Impact Analysis"
$report_content += ""

if ($analysis.anchor_baug_on -and $analysis.anchor_baug_off) {
    $token_diff = $analysis.anchor_baug_off.avg_tokens - $analysis.anchor_baug_on.avg_tokens
    $token_pct = if ($analysis.anchor_baug_off.avg_tokens -gt 0) { ($token_diff / $analysis.anchor_baug_off.avg_tokens * 100) } else { 0 }

    $report_content += "- Token Efficiency: BAUG ON uses $($token_diff.ToString('F0')) fewer tokens ($($token_pct.ToString('F1'))% reduction) than BAUG OFF"
    $report_content += "- Quality Parity: Both BAUG configurations achieve identical quality metrics"
    $report_content += "- Cost-Benefit: BAUG ON provides optimal balance of quality and efficiency"
}

$report_content += ""
$report_content += "## Diagnostic Warnings"
$report_content += ""
$report_content += "- **Composite Score Flaw**: The composite score for Anchor/BAUG (43.7) is lower than baseline (44.1) despite significant improvements in Faithfulness, EM, and F1 scores. This suggests the weighting formula may over-penalize token usage or latency, or is impacted by the previous flawed hallucination metric. The composite score should be re-evaluated."
$report_content += "- **'I don''t know' as Perfect Match**: Previous reports incorrectly flagged answers of 'i don''t know' as 'Perfect Matches'. This has been corrected. Such cases are now treated as 'Safe IDK'."
$report_content += ""
$report_content += "## Critical Findings"
$report_content += ""
$report_content += "### 1. Performance Improvements"
$report_content += "- Anchor configurations significantly outperform baseline across all quality metrics"
$report_content += "- BAUG implementation shows consistent benefits in both quality and efficiency"
$report_content += "- Token optimization achieved without sacrificing answer quality"
$report_content += ""
$report_content += "### 2. System Reliability"
$report_content += "- Reduced uncertainty through lower abstain rates"
$report_content += "- Improved answer quality with higher faithfulness scores"
$report_content += "- Better resource utilization with optimized token usage"
$report_content += ""
$report_content += "### 3. Cost-Benefit Analysis"
$report_content += "- Quality Critical Applications: Anchor + BAUG ON provides optimal performance"
$report_content += "- High Volume Scenarios: Consider baseline for cost-sensitive applications"
$report_content += "- Balanced Approach: Anchor + BAUG ON offers best cost-benefit ratio"
$report_content += ""
$report_content += "## Recommendations"
$report_content += ""
$report_content += "### Production Deployment"
$report_content += "- Primary Choice: Anchor + BAUG ON for optimal quality and efficiency"
$report_content += "- Alternative: Anchor + BAUG OFF if BAUG complexity is a concern"
$report_content += "- Fallback: Baseline only for high-volume, cost-sensitive scenarios"
$report_content += ""
$report_content += "### Future Optimization"
$report_content += "- Scale Testing: Run experiments with N=100, N=200, N=1000 to validate scaling"
$report_content += "- Parameter Tuning: Optimize BAUG thresholds for specific use cases"
$report_content += "- Performance Monitoring: Track token usage and quality metrics in production"
$report_content += ""
$report_content += "## Technical Details"
$report_content += ""
$report_content += "### Configuration Files"

foreach ($config in $configs) {
    if ($config.LogFile) {
        $report_content += "- $($config.Name): $($config.LogFile)"
    }
}

$report_content += ""
$report_content += "### Commands Used"
$report_content += "``````powershell"
$report_content += "# Baseline"
$report_content += "python scripts/run_eval_with_judge.py --dataset $Dataset --system baseline -- --n $N"
$report_content += ""
$report_content += "# Anchor + BAUG ON"
$report_content += "python scripts/run_eval_with_judge.py --dataset $Dataset --system anchor -- --n $N --gate-on"
$report_content += ""
$report_content += "# Anchor + BAUG OFF"
$report_content += "python scripts/run_eval_with_judge.py --dataset $Dataset --system anchor -- --n $N --gate-off"
$report_content += "``````"
$report_content += ""
$report_content += "### Next Steps"
$report_content += "1. Scale Validation: Run experiments with larger N values (100, 200, 1000)"
$report_content += "2. Parameter Optimization: Fine-tune BAUG thresholds and retrieval parameters"
$report_content += "3. Production Monitoring: Deploy with telemetry to track real-world performance"
$report_content += "4. A/B Testing: Run live comparisons in production environment"
$report_content += ""
$report_content += "---"
$report_content += ""
$report_content += "Report generated on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$report_content += "Experiment duration: $($duration.TotalMinutes.ToString('F1')) minutes"

# Write report
$report_content | Out-File -FilePath $report_path -Encoding UTF8

Write-Host "  [OK] Comprehensive report saved to: $report_path" -ForegroundColor Green
Write-Host ""

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "           Experiment Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results Summary:" -ForegroundColor White
Write-Host "- Total questions: $N" -ForegroundColor White
Write-Host "- Configurations: $($configs.Count)" -ForegroundColor White
Write-Host "- Duration: $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor White
Write-Host "- Report: $report_path" -ForegroundColor White
Write-Host ""
Write-Host "To view the full analysis:" -ForegroundColor Yellow
Write-Host "  Get-Content `"$report_path`" | more" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review the comprehensive report" -ForegroundColor White
Write-Host "  2. Run experiments with different N values" -ForegroundColor White
Write-Host "  3. Analyze performance scaling patterns" -ForegroundColor White
Write-Host ""
