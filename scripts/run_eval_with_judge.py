"""Run evaluation, then judge, benchmark, and summarize validator findings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _run(cmd: Sequence[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, env=env)


def _latest_prediction(system: str, before: set[str]) -> Path:
    log_dir = Path("logs") / system
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    pattern = f"*_{system}.jsonl"
    candidates = [p for p in log_dir.glob(pattern) if p.is_file()]
    after = {p.name for p in candidates}
    new_files = after - before

    if new_files:
        picks = [log_dir / name for name in new_files]
    else:
        if not candidates:
            raise FileNotFoundError(
                f"No prediction files matching {pattern} in {log_dir}"
            )
        picks = candidates

    return max(picks, key=lambda p: p.stat().st_mtime)


def _preview(title: str, items: list[dict[str, Any]], limit: int) -> None:
    print(f"\n{title}: {len(items)}")
    for hit in items[:limit]:
        print("- QID", hit.get("qid"))
        print("  Q :", hit.get("question"))
        print("  A :", hit.get("answer"))
        print("  G :", hit.get("gold"))
        if label := hit.get("validator_label"):
            print("  label:", label)
        if notes := hit.get("validator_notes"):
            print("  notes:", notes)
        print(
            "  em="
            + str(hit.get("em"))
            + ", f1="
            + str(hit.get("f1"))
            + ", overall="
            + str(hit.get("overall"))
            + (", safe_idk" if hit.get("safe_idk") else "")
            + (", hallucination" if hit.get("hallucination") else "")
        )


def _validator_report(
    judge_path: Path,
    *,
    limit: int,
    f1_threshold: float,
    overall_pass: float,
    overall_partial: float,
) -> None:
    try:
        rows = [
            json.loads(line)
            for line in judge_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        print(f"No judge log found at {judge_path}")
        return

    perfect: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []
    safe_idk_rows: list[dict[str, Any]] = []
    hallucinations: list[dict[str, Any]] = []

    for row in rows:
        auto = row.get("auto_metrics", {}) or {}
        scores = (row.get("llm_scores", {}) or {}).get("scores", {}) or {}
        em_raw = auto.get("em")
        f1_raw = auto.get("f1_short")
        overall_raw = scores.get("overall")

        try:
            em = float(em_raw) if em_raw is not None else None
        except (TypeError, ValueError):
            em = None

        try:
            f1 = float(f1_raw) if f1_raw is not None else None
        except (TypeError, ValueError):
            f1 = None

        try:
            overall = float(overall_raw) if overall_raw is not None else 0.0
        except (TypeError, ValueError):
            overall = 0.0

        entry = {
            "qid": row.get("qid"),
            "question": row.get("question"),
            "answer": row.get("final_answer") or row.get("answer"),
            "gold": row.get("gold"),
            "em": em,
            "f1": f1,
            "overall": overall,
            "safe_idk": auto.get("safe_idk"),
            "hallucination": auto.get("hallucination")
            or bool(
                (row.get("llm_scores", {}) or {})
                .get("flags", {})
                .get("contradiction_with_evidence")
            ),
        }

        notes: list[str] = []
        if f1 is not None and f1 >= f1_threshold:
            notes.append(f"f1 >= {f1_threshold}")
        if overall >= overall_pass:
            notes.append(f"overall >= {overall_pass}")
        elif overall >= overall_partial:
            notes.append(f"overall >= {overall_partial}")
        if notes:
            entry["validator_notes"] = "; ".join(notes)

        if entry["safe_idk"]:
            safe_idk_rows.append(entry)
            continue
        if entry["hallucination"]:
            hallucinations.append(entry)
            continue

        is_perfect = em is not None and em >= 1.0
        meets_semantic = overall >= overall_pass
        meets_partial = (
            f1 is not None and f1 >= f1_threshold
        ) or overall >= overall_partial

        if is_perfect:
            entry["validator_label"] = "Perfect"
            perfect.append(entry)
        elif meets_semantic or meets_partial:
            entry["validator_label"] = "Partial"
            partial.append(entry)
        else:
            entry["validator_label"] = "Needs review"
            needs_review.append(entry)

    _preview("Perfect (EM = 1)", perfect, limit)
    _preview(
        f"Partial match (overall >= {overall_partial} or F1 >= {f1_threshold})",
        partial,
        limit,
    )
    _preview("Needs review", needs_review, limit)
    _preview("Safe IDK", safe_idk_rows, limit)
    _preview("Hallucination flagged", hallucinations, limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval, judge, benchmark, and validator."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    parser.add_argument(
        "--system", required=True, help="System name (baseline|agent|anchor ...)"
    )
    parser.add_argument(
        "--judge-require-citation",
        choices={"true", "false"},
        default="false",
        help="Pass-through to eval.cli_judge",
    )
    parser.add_argument(
        "--judge-debug", action="store_true", help="Enable --debug for judge step"
    )
    parser.add_argument(
        "--validator-limit",
        type=int,
        default=5,
        help="How many rows to preview in validator report",
    )
    parser.add_argument(
        "--validator-f1-threshold",
        type=float,
        default=0.3,
        help="Minimum F1 to consider answer as partial match",
    )
    parser.add_argument(
        "--validator-overall-pass",
        type=float,
        default=70.0,
        help="LLM overall score to treat answer as semantically correct",
    )
    parser.add_argument(
        "--validator-overall-partial",
        type=float,
        default=50.0,
        help="LLM overall score to treat answer as partial",
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Extra args for agentic_rag.eval.runner (prefix with -- before these args)",
    )

    args = parser.parse_args()

    eval_extra = args.eval_args or []
    if eval_extra and eval_extra[0] == "--":
        eval_extra = eval_extra[1:]

    log_dir = Path("logs") / args.system
    before = {p.name for p in log_dir.glob(f"*_{args.system}.jsonl") if p.is_file()}

    eval_cmd = [
        sys.executable,
        "-m",
        "agentic_rag.eval.runner",
        "--dataset",
        args.dataset,
        "--system",
        args.system,
    ] + eval_extra
    _run(eval_cmd)

    pred_path = _latest_prediction(args.system, before)
    judge_cmd = [
        sys.executable,
        "-m",
        "eval.cli_judge",
        "-p",
        str(pred_path),
        "--require-citation",
        args.judge_require_citation,
    ]
    if args.judge_debug:
        judge_cmd.append("--debug")
    _run(judge_cmd)

    judge_path = pred_path.with_name(pred_path.stem + "_judge_gold.jsonl")
    bench_path = pred_path.with_name(pred_path.stem + "_benchmarks.jsonl")

    bench_cmd = [
        sys.executable,
        "-m",
        "eval.compute_benchmarks",
        "--input",
        str(judge_path),
        "--out",
        str(bench_path),
    ]
    _run(bench_cmd)

    _validator_report(
        judge_path,
        limit=args.validator_limit,
        f1_threshold=args.validator_f1_threshold,
        overall_pass=args.validator_overall_pass,
        overall_partial=args.validator_overall_partial,
    )

    print("\nDone.")
    print(f"Predictions : {pred_path}")
    print(f"Judge log   : {judge_path}")
    print(f"Benchmarks  : {bench_path}")


if __name__ == "__main__":
    main()
