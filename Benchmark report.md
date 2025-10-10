Benchmark Report (n=100)

This report summarizes three CRAG evaluations run on 100 questions:
- Anchor with BAUG (gate on)
- Anchor without BAUG (gate off)
- Baseline

Numbers are taken from each run’s evaluation summary, judge summary, and enriched benchmark output.

Anchor + BAUG
- Evaluation summary
  - Count: 100
  - Avg Faithfulness (fallback): 0.727
  - Avg Overlap: 0.624
  - Avg EM: 0.060; Avg F1: 0.247
  - Abstain Rate: 0.220
  - Avg Total Tokens: 1549; P50 Latency (ms): 1656
- Judge summary
  - Mean overall: 44.35; Mean F1_short: 0.247; Mean support overlap: 0.083
  - Counts: partial_match(correct) 48, perfect EM 6, safe_idk 22, hallucination 34
- Benchmarks
  - Avg Faithfulness: 0.712; Avg Overlap: 0.610
  - Avg RAB: 66.10; Avg AQB: 61.56; Avg Composite: 49.82
- Artefacts
  - JSONL: logs\anchor\1759567172_anchor.jsonl
  - Summary: logs\anchor\1759567172_anchor_summary.md
  - CSV/QA: logs\anchor\1759567172_anchor_summary.csv, logs\anchor\1759567172_anchor_qa_pairs.csv

Anchor (Gate Off)
- Evaluation summary
  - Count: 100
  - Avg Faithfulness (fallback): 0.708
  - Avg Overlap: 0.591
  - Avg EM: 0.070; Avg F1: 0.263
  - Abstain Rate: 0.230
  - Avg Total Tokens: 2205; P50 Latency (ms): 1705
- Judge summary
  - Mean overall: 44.61; Mean F1_short: 0.263; Mean support overlap: 0.078
  - Counts: match 54, partial_match(correct) 47, safe_idk 23, hallucination 34
- Benchmarks
  - Avg Faithfulness: 0.694; Avg Overlap: 0.580
  - Avg RAB: 63.70; Avg AQB: 61.42; Avg Composite: 49.30
- Artefacts
  - JSONL: logs\anchor\1759569836_anchor.jsonl
  - Summary: logs\anchor\1759569836_anchor_summary.md
  - CSV/QA: logs\anchor\1759569836_anchor_summary.csv, logs\anchor\1759569836_anchor_qa_pairs.csv

Baseline
- Evaluation summary
  - Count: 100
  - Avg Faithfulness (fallback): 0.667
  - Avg Overlap: 0.603
  - Avg EM: 0.000; Avg F1: 0.150
  - Abstain Rate: 0.300
  - Avg Total Tokens: 1116; P50 Latency (ms): 1666
- Judge summary
  - Mean overall: 39.31; Mean F1_short: 0.150; Mean support overlap: 0.077
  - Counts: match 48, partial_match(correct) 48, safe_idk 30, hallucination 31, perfect EM 0
- Benchmarks
  - Avg Faithfulness: 0.656; Avg Overlap: 0.591
  - Avg RAB: 62.36; Avg AQB: 57.34; Avg Composite: 47.63
- Artefacts
  - JSONL: logs\baseline\1759563591_baseline.jsonl
  - Summary: logs\baseline\1759563591_baseline_summary.md
  - CSV/QA: logs\baseline\1759563591_baseline_summary.csv, logs\baseline\1759563591_baseline_qa_pairs.csv

Key Takeaways
- Anchor orchestration improves answer quality and evidence use over Baseline. BAUG (gate on) further nudges RAB/AQB/Composite upward with a slightly lower abstain rate.
- In this 100‑Q slice, Anchor + BAUG and Anchor (off) have comparable hallucination counts (both 34). BAUG yields 6 perfect EM; Baseline has none.
- Typical failures:
  - Award‑year ambiguity (ceremony year vs film/release year)
  - Numeric aggregation mismatch (e.g., daily vs weekly totals)
  - Incomplete lists when contexts don’t cover all requested items

Next Steps
- Add ceremony‑year disambiguation on BAUG retry for award queries.
- Tighten numeric time‑window checks to reduce aggregation mismatches.
- Introduce a list‑coverage guard (require ≥ N distinct items) before asserting in “top N” questions; otherwise prefer safe IDK.
