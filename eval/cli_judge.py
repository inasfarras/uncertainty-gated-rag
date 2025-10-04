"""CLI for re-judging agent predictions with gold-aware GPT judge."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .gpt_agent_judge import extract_citation_ids, judge_example
from .idk_policy import classify_answer


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_passages(passages: dict[Any, Any] | None) -> dict[str, str]:
    if not passages:
        return {}
    normalized: dict[str, str] = {}
    for key, value in passages.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Judge agent outputs against gold answers."
    )
    parser.add_argument(
        "-p",
        "--predictions",
        type=Path,
        required=True,
        help="Predictions JSONL file (usually logs/<run>/<timestamp>_*.jsonl)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/crag_questions.jsonl"),
        help="Gold dataset JSONL (default: data/crag_questions.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination JSONL (default: <predictions>_judge_gold.jsonl)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--temp", type=float, default=0.2, help="Generation temperature for the judge"
    )
    parser.add_argument("--parallel", type=int, default=4, help="Parallel worker count")
    parser.add_argument(
        "--require-citation",
        choices={"true", "false"},
        default="false",
        help="Require at least one citation in answers",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print progress as rows are judged",
    )
    args = parser.parse_args()

    predictions = _load_jsonl(args.predictions)
    gold_rows = _load_jsonl(args.gold)
    gold_lookup = {str(row.get("id")): row.get("gold", "") for row in gold_rows}

    require_citation = args.require_citation == "true"

    out_path = args.out
    if out_path is None:
        stem = args.predictions.stem + "_judge_gold.jsonl"
        out_path = args.predictions.with_name(stem)

    def _evaluate(row: dict[str, Any]) -> dict[str, Any]:
        qid = str(row.get("qid") or row.get("id") or "")
        gold_answer = gold_lookup.get(qid, "")
        # Prefer explicit passages; if missing, fall back to contexts (id->text)
        raw_passages = row.get("passages") or {}
        evidence_source = "passages"
        answer_text = row.get("answer") or row.get("final_answer") or ""
        if not raw_passages:
            ctx_list = row.get("contexts") or []
            # Prioritize contexts referenced by citations, then fill to a cap
            cap = 20
            cited_ids = [str(cid) for cid in extract_citation_ids(answer_text)]
            cited_set = set(cited_ids)
            ordered: list[tuple[str, str]] = []
            seen: set[str] = set()
            for c in ctx_list:
                if not c:
                    continue
                cid = str(c.get("id")) if c.get("id") is not None else None
                text = c.get("text")
                if not cid or not text:
                    continue
                if cid in cited_set and cid not in seen:
                    ordered.append((cid, str(text)))
                    seen.add(cid)
            for c in ctx_list:
                if len(ordered) >= cap:
                    break
                if not c:
                    continue
                cid = str(c.get("id")) if c.get("id") is not None else None
                text = c.get("text")
                if not cid or not text or cid in seen:
                    continue
                ordered.append((cid, str(text)))
                seen.add(cid)
            if ordered:
                raw_passages = {cid: txt for cid, txt in ordered}
                evidence_source = "contexts_fallback"

        normalized_passages = _normalize_passages(raw_passages)
        short_answer = row.get("final_short") or row.get("short_answer")
        judgement = judge_example(
            row.get("question", ""),
            gold_answer,
            answer_text,
            normalized_passages,
            model=args.model,
            temperature=args.temp,
            short_answer=short_answer,
        )
        auto = judgement["auto_metrics"]
        has_citation = bool(auto.get("has_citation"))
        if require_citation and not has_citation:
            auto["citation_required_violation"] = True
        classification = classify_answer(
            answer_text,
            auto,
            gold_answer,
            normalized_passages,
            judgement["llm_scores"].get("flags", {}),
            composite=judgement.get("composite_overall"),
        )
        auto.update(classification)
        record = dict(row)
        record.update(
            {
                "qid": qid,
                "question": row.get("question"),
                "answer": answer_text,
                # Persist the exact evidence we sent to the judge
                "passages": normalized_passages,
                "gold": gold_answer,
                **judgement,
            }
        )
        record.setdefault("debug", {})
        record["debug"]["evidence_source"] = evidence_source
        try:
            record["debug"]["cit_in_answer"] = [
                str(x) for x in extract_citation_ids(answer_text)
            ]
        except Exception:
            pass
        if "faithfulness" not in record and "final_f" in record:
            record["faithfulness"] = record.get("final_f")
        if "overlap" not in record and "final_o" in record:
            record["overlap"] = record.get("final_o")
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = [executor.submit(_evaluate, row) for row in predictions]
        results: list[dict[str, Any]] = []
        for idx, fut in enumerate(as_completed(futures), start=1):
            outcome = fut.result()
            results.append(outcome)
            if args.debug:
                print(f"[judge] processed {idx}/{len(futures)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    overall_scores = [r.get("composite_overall", 0.0) for r in results]
    f1_scores = [r.get("auto_metrics", {}).get("f1_short", 0.0) for r in results]
    overlap_scores = [
        r.get("auto_metrics", {}).get("support_overlap", 0.0) for r in results
    ]
    safe_idk_count = sum(
        1 for r in results if r.get("auto_metrics", {}).get("safe_idk")
    )
    hallucination_count = sum(
        1 for r in results if r.get("auto_metrics", {}).get("hallucination")
    )
    match_count = sum(1 for r in results if r.get("auto_metrics", {}).get("match"))
    partial_correct_count = sum(
        1 for r in results if r.get("auto_metrics", {}).get("partial_correct")
    )

    def _avg(values: Iterable[float]) -> float:
        values = list(values)
        return float(mean(values)) if values else 0.0

    print(f"Results written to {out_path}")
    print("Judge Summary")
    print(f"Mean overall: {_avg(overall_scores):.2f}")
    print(f"Mean F1_short: {_avg(f1_scores):.3f}")
    print(f"Mean support overlap: {_avg(overlap_scores):.3f}")
    print(
        f"Counts - match: {match_count}, partial_match(correct): {partial_correct_count}, "
        f"safe_idk: {safe_idk_count}, hallucination: {hallucination_count}"
    )


if __name__ == "__main__":  # pragma: no cover
    run_cli()
