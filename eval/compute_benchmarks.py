from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List

from .idk_policy import classify_answer


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def compute_row(row: Dict[str, Any]) -> Dict[str, Any]:
    faith = _num(row.get("faithfulness", 0.0))
    overlap = _num(row.get("overlap", 0.0))
    rab = 100.0 * (0.5 * faith + 0.5 * overlap)

    llm_scores = row.get("llm_scores", {}) or {}
    scores = llm_scores.get("scores", {}) or {}
    aqb = _num(scores.get("overall", 0.0))

    answer = str(row.get("answer", ""))
    passages = row.get("passages", {}) or {}
    auto = row.get("auto_metrics", {}) or {}
    classification = classify_answer(
        answer,
        auto,
        row.get("gold"),
        passages,
        llm_scores.get("flags", {}),
        composite=row.get("composite_overall"),
    )
    safe_idk = classification["safe_idk"]
    bad_idk = classification["bad_idk"]
    hallucination = classification["hallucination"]
    gold_invalid = classification.get("gold_invalid", False)
    match = classification["match"]
    perfect_match = classification["perfect_match"]
    partial_correct = classification["partial_correct"]

    composite = 0.6 * rab + 0.4 * aqb
    if hallucination:
        composite = min(composite, 10.0)
    elif bad_idk:
        composite = min(composite, 20.0)
    elif safe_idk:
        composite = max(composite, 30.0)

    auto = row.get("auto_metrics", {}) or {}
    f1_short = _num(auto.get("f1_short", 0.0))
    support_overlap = _num(auto.get("support_overlap", 0.0))

    enriched = dict(row)
    enriched.update(
        {
            "RAB": round(rab, 2),
            "AQB": round(aqb, 2),
            "Composite": round(composite, 2),
            "F1_short": round(f1_short, 3),
            "SupportOverlap": round(support_overlap, 3),
            "GoldInvalid": gold_invalid,
            "SafeIDK": safe_idk,
            "BadIDK": bad_idk,
            "Hallucination": hallucination,
            "Match": match,
            "PerfectMatch": perfect_match,
            "PartialCorrect": partial_correct,
        }
    )
    return enriched


def _mean(rows: List[Dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    values = [_num(row.get(field)) for row in rows]
    return sum(values) / len(rows)


def summarize(rows: List[Dict[str, Any]]) -> str:
    count = len(rows)
    avg_faith = _mean(rows, "faithfulness")
    avg_overlap = _mean(rows, "overlap")
    avg_rab = _mean(rows, "RAB")
    avg_aqb = _mean(rows, "AQB")
    avg_comp = _mean(rows, "Composite")
    avg_f1 = _mean(rows, "F1_short")
    avg_support = _mean(rows, "SupportOverlap")

    match_count = sum(1 for r in rows if r.get("Match"))
    partial_correct = sum(1 for r in rows if r.get("PartialCorrect"))
    safe = sum(1 for r in rows if r.get("SafeIDK"))
    hallu = sum(1 for r in rows if r.get("Hallucination"))

    lines = [
        "Benchmark Summary",
        f"Count: {count}",
        f"Avg Faithfulness: {avg_faith:.3f}",
        f"Avg Overlap: {avg_overlap:.3f}",
        f"Avg RAB: {avg_rab:.2f}",
        f"Avg AQB: {avg_aqb:.2f}",
        f"Avg Composite: {avg_comp:.2f}",
        f"Avg F1_short: {avg_f1:.3f}",
        f"Avg SupportOverlap: {avg_support:.3f}",
        f"Counts - match: {match_count}, partial_match(correct): {partial_correct}, safe_idk: {safe}, hallucination: {hallu}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RAB/AQB/Composite benchmarks from judge JSONL."
    )
    parser.add_argument("--input", required=True, help="Path to judge_gold.jsonl")
    parser.add_argument(
        "--out", required=True, help="Path to write enriched benchmarks JSONL"
    )
    args = parser.parse_args()

    rows_in = [r for r in _read_jsonl(args.input)]
    rows_out = [compute_row(r) for r in rows_in]
    _write_jsonl(args.out, rows_out)

    print(summarize(rows_out))


if __name__ == "__main__":
    main()
