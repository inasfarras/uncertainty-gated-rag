#!/usr/bin/env python
"""Compare one or more agent summary CSV files.

This utility ingests the per-question summary CSVs emitted by
``python -m agentic_rag.eval.runner`` and prints an aggregated view so you can
contrast different runs quickly (e.g., gate on vs gate off).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Metrics we aggregate for each run (column name -> aggregation label)
AGG_METRICS: list[tuple[str, str]] = [
    ("final_f", "final_f_mean"),
    ("final_o", "final_o_mean"),
    ("em", "em_mean"),
    ("f1", "f1_mean"),
    ("abstain", "abstain_rate"),
    ("total_tokens", "total_tokens_mean"),
    ("latency_ms", "latency_ms_mean"),
]

PERCENTILE_METRICS: list[tuple[str, str]] = [
    ("total_tokens", "total_tokens_p50"),
    ("latency_ms", "latency_ms_p50"),
]


def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand shell-style patterns while preserving user-provided order."""
    paths: list[Path] = []
    for pattern in patterns:
        expanded = (
            list(Path().glob(pattern))
            if any(ch in pattern for ch in "*?[")
            else [Path(pattern)]
        )
        if not expanded:
            print(
                f"warning: pattern '{pattern}' did not match any files", file=sys.stderr
            )
        for path in expanded:
            if path.is_file():
                paths.append(path)
            else:
                print(f"warning: skipping '{path}' (not a file)", file=sys.stderr)
    # Deduplicate while keeping order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _summarise(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"run '{path}' has no rows")

    summary = {
        "run": path.stem,
        "path": str(path),
        "rows": int(len(df)),
        "system": (
            ",".join(sorted(str(s) for s in df["system"].dropna().unique()))
            if "system" in df
            else "unknown"
        ),
    }

    for source_col, target in AGG_METRICS:
        if source_col in df:
            summary[target] = float(df[source_col].mean())
    for source_col, target in PERCENTILE_METRICS:
        if source_col in df:
            summary[target] = float(df[source_col].median())

    if "idk_with_citation_count" in df:
        summary["idk_with_citation_total"] = int(df["idk_with_citation_count"].sum())

    if "rounds" in df:
        summary["rounds_mean"] = float(df["rounds"].mean())

    return pd.Series(summary)


def compare_runs(paths: Iterable[str], sort_by: str, ascending: bool) -> None:
    files = _expand_paths(paths)
    if not files:
        raise SystemExit("no input files matched the provided patterns")

    runs: list[pd.Series] = []
    for file_path in files:
        try:
            runs.append(_summarise(file_path))
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"error: failed to process '{file_path}': {exc}", file=sys.stderr)

    if not runs:
        raise SystemExit("no runs could be processed")

    df = pd.DataFrame(runs)
    numeric_cols = [col for col in df.columns if col not in {"run", "path", "system"}]

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    formatters = {
        col: (lambda v: f"{v:.3f}" if pd.notna(v) else "-") for col in numeric_cols
    }
    formatters.update(
        {
            "rows": lambda v: f"{int(v)}",
            "idk_with_citation_total": lambda v: f"{int(v)}",
        }
    )

    display_cols = [
        "run",
        "system",
        "rows",
        "final_f_mean",
        "final_o_mean",
        "em_mean",
        "f1_mean",
        "abstain_rate",
        "total_tokens_mean",
        "total_tokens_p50",
        "latency_ms_mean",
        "latency_ms_p50",
        "rounds_mean",
        "idk_with_citation_total",
    ]
    display_cols = [col for col in display_cols if col in df.columns]

    print(
        "Run comparison (sorted by {} {}):".format(
            sort_by, "asc" if ascending else "desc"
        )
    )
    print(df[display_cols].to_string(index=False, formatters=formatters))

    if len(df) > 1:
        baseline = df.iloc[0]
        diff_cols = [
            col
            for col in display_cols
            if col not in {"run", "system", "rows", "idk_with_citation_total"}
        ]
        diff_records = []
        for _, row in df.iterrows():
            record = {"run": row["run"]}
            for col in diff_cols:
                base_val = baseline[col]
                cur_val = row[col]
                if pd.isna(cur_val) or pd.isna(base_val):
                    delta = float("nan")
                else:
                    delta = cur_val - base_val
                record[col] = delta
            diff_records.append(record)
        diff_df = pd.DataFrame(diff_records)
        diff_formatters = {
            col: (lambda v: f"{v:+.3f}" if pd.notna(v) else "-") for col in diff_cols
        }
        print(f"\nDelta relative to baseline run '{baseline['run']}':")
        print(
            diff_df.to_string(index=False, formatters={"run": str, **diff_formatters})
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare agent/baseline summary CSV files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Summary CSV paths or glob patterns (e.g. logs/*_summary.csv)",
    )
    parser.add_argument(
        "--sort",
        default="final_f_mean",
        help="Column to sort by (default: final_f_mean)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default descending).",
    )

    args = parser.parse_args()

    compare_runs(args.paths, sort_by=args.sort, ascending=args.ascending)


if __name__ == "__main__":
    main()
