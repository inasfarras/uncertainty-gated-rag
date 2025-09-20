import json
import random
import re
import string
import time
from pathlib import Path
from typing import Literal, Optional, cast

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from agentic_rag.agent.loop import Agent, Baseline
from agentic_rag.config import settings
from agentic_rag.eval.signals import (
    em_f1,
    faithfulness_fallback,
    sentence_support,
)

# Helper functions for runner
_IDK_PAT = re.compile(r"^i\s*do?n'?t\s*know\.?$")


def _strip_citations(s: str) -> str:
    return re.sub(r"\s*\[CIT:[A-Za-z0-9_\-]+\]\s*", " ", s or "").strip()


def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def _is_idk_no_citation(ans: str) -> bool:
    has_cit = bool(re.search(r"\[CIT:[A-Za-z0-9_\-]+\]", ans or ""))
    core = _canon(re.sub(r"\[CIT:[A-Za-z0-9_\-]+\]", " ", ans or ""))
    return (not has_cit) and (_IDK_PAT.match(core.replace(" ", "") + ".") is not None)


app = typer.Typer()
console = Console()


def override_settings(
    tau_f: Optional[float], tau_o: Optional[float], max_rounds: Optional[int]
):
    if tau_f is not None:
        settings.FAITHFULNESS_TAU = tau_f
    if tau_o is not None:
        settings.OVERLAP_TAU = tau_o
    if max_rounds is not None:
        settings.MAX_ROUNDS = max_rounds


_dataset_option = typer.Option(
    ..., "--dataset", help="Path to a JSONL dataset with {'id', 'question'}."
)
_n_option = typer.Option(0, "--n", help="Number of items to process. 0 for all.")
_system_option = typer.Option(
    "baseline", "--system", help="System to run: 'baseline' or 'agent'."
)
_backend_option = typer.Option(
    settings.EMBED_BACKEND, "--backend", help="Backend to use: 'openai' or 'mock'."
)
_gate_on_option = typer.Option(
    True, "--gate-on/--gate-off", help="Enable or disable the agentic gate."
)
_max_rounds_option = typer.Option(
    None, "--max-rounds", help="Max rounds for the agent."
)
_tau_f_option = typer.Option(None, "--tau-f", help="Faithfulness threshold.")
_tau_o_option = typer.Option(None, "--tau-o", help="Overlap threshold.")
_debug_prompts_option = typer.Option(
    False, "--debug-prompts", help="Enable debug logging for first 3 queries."
)
_judge_policy_option = typer.Option(
    "never",
    "--judge-policy",
    help="Judge policy for agent: never | gray_zone | always",
    case_sensitive=False,
)
# Gate kind removed - only UncertaintyGate available
_faith_report_option = typer.Option(
    "fallback",
    "--faith-report",
    help="Which faithfulness metrics to include in summary: fallback | ragas | both",
)
_ragas_subset_option = typer.Option(
    0,
    "--ragas-eval-subset",
    help="Evaluate RAGAS on a subset of final outputs (0 for all)",
)
_tau_lo_option = typer.Option(
    0.40, "--tau-lo", help="Gray-zone lower bound for overlap"
)
_tau_hi_option = typer.Option(
    0.60, "--tau-hi", help="Gray-zone upper bound for overlap"
)
_eps_overlap_option = typer.Option(
    0.02, "--epsilon-overlap", help="Stagnation threshold for overlap delta"
)

_use_final_short_option = typer.Option(
    False,
    "--use-final-short/--no-use-final-short",
    help="Score EM/F1 using best-of (final_short vs full answer)",
)


@app.command()
def run(
    dataset: Path = _dataset_option,
    n: int = _n_option,
    system: str = _system_option,
    backend: str = _backend_option,
    gate_on: bool = _gate_on_option,
    max_rounds: Optional[int] = _max_rounds_option,
    tau_f: Optional[float] = _tau_f_option,
    tau_o: Optional[float] = _tau_o_option,
    debug_prompts: bool = _debug_prompts_option,
    judge_policy: str = _judge_policy_option,
    faith_report: str = _faith_report_option,
    ragas_eval_subset: int = _ragas_subset_option,
    tau_lo: float = _tau_lo_option,
    tau_hi: float = _tau_hi_option,
    epsilon_overlap: float = _eps_overlap_option,
    use_final_short: bool = _use_final_short_option,
    # gate_kind removed - only UncertaintyGate available
):
    """Runs an evaluation of the Agentic RAG system."""

    # Update settings from CLI
    settings.EMBED_BACKEND = cast(Literal["openai", "st"], backend)
    override_settings(tau_f, tau_o, max_rounds)
    # Update additional settings
    settings.JUDGE_POLICY = cast(Literal["never", "gray_zone", "always"], judge_policy)
    # Gate kind setting removed - only UncertaintyGate available
    settings.TAU_LO = tau_lo
    settings.TAU_HI = tau_hi
    settings.EPSILON_OVERLAP = epsilon_overlap
    settings.USE_FINAL_SHORT_SCORING = bool(use_final_short)

    # Load dataset
    with open(dataset) as f:
        questions = [json.loads(line) for line in f]
    if n > 0:
        questions = questions[:n]

    # Select system
    model: Agent | Baseline
    if system == "agent":
        model = Agent(gate_on=gate_on, debug_mode=debug_prompts)
    elif system == "baseline":
        model = Baseline(debug_mode=debug_prompts)
    else:
        console.print(f"[bold red]Unknown system: {system}[/bold red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Running evaluation for system: '{system}' with gate {'ON' if gate_on and system == 'agent' else 'OFF'}[/bold green]"
    )

    # Run evaluation
    results = []
    for idx, item in enumerate(questions):
        console.print(f"Processing qid: {item['id']}")
        # Enable debug only for first 3 queries
        if debug_prompts and idx >= 3:
            model.debug_mode = False
        summary = model.answer(question=item["question"], qid=item["id"])

        console.print(
            f"  [bold blue]Agent Answer:[/bold blue] {summary.get('final_answer', '')}"
        )

        gold = item.get("gold", "")
        is_unans = gold.strip().lower() in {"invalid question", "n/a", "unknown", ""}

        pred_answer = summary.get("final_answer", "")
        said_idk = _is_idk_no_citation(pred_answer)
        abstain_correct = 1 if (is_unans and said_idk) else 0
        hallucinated_unans = 1 if (is_unans and not said_idk) else 0

        # Build ctx_map from exact contexts used
        ctx_list = summary.get("contexts", []) or []
        ctx_map = {c["id"]: c["text"] for c in ctx_list}
        sup = sentence_support(
            summary.get("final_answer", ""), ctx_map, tau_sim=settings.OVERLAP_SIM_TAU
        )
        final_o = (
            float(sup["overlap"]) if isinstance(sup.get("overlap"), float) else 0.0
        )
        # Choose best of final_short vs. full answer for scoring to avoid penalizing
        # when short extraction fails.
        full_pred = summary.get("final_answer", "")
        short_pred = summary.get("final_short") or ""
        ef_full = em_f1(full_pred, item.get("gold"))
        ef_short = (
            em_f1(short_pred, item.get("gold"))
            if short_pred
            else {"em": 0.0, "f1": 0.0}
        )
        # Choose scoring strategy
        if settings.USE_FINAL_SHORT_SCORING:
            # Prefer higher F1; tie-breaker on EM
            if ef_short.get("f1", 0.0) > ef_full.get("f1", 0.0) or (
                ef_short.get("f1", 0.0) == ef_full.get("f1", 0.0)
                and ef_short.get("em", 0.0) > ef_full.get("em", 0.0)
            ):
                pred_for_scoring = short_pred
                ef = ef_short
                summary["pred_source"] = "short"
            else:
                pred_for_scoring = full_pred
                ef = ef_full
                summary["pred_source"] = "full"
        else:
            pred_for_scoring = full_pred
            ef = ef_full
            summary["pred_source"] = "full"
        final_f = faithfulness_fallback(pred_for_scoring, item.get("gold"), final_o)
        abstain = 1 if said_idk else 0
        wrong_on_answerable = (
            1
            if ((not is_unans) and (not said_idk) and float(ef.get("f1", 0.0)) == 0.0)
            else 0
        )

        # Include question for readability in logs
        summary["question"] = item.get("question", "")

        # Update summary with hardened metrics
        summary["final_o"] = final_o
        summary["final_f"] = final_f
        summary["em"] = ef.get("em", 0.0)
        summary["f1"] = ef.get("f1", 0.0)
        summary["em_full"] = ef_full.get("em", 0.0)
        summary["f1_full"] = ef_full.get("f1", 0.0)
        summary["em_short"] = ef_short.get("em", 0.0)
        summary["f1_short"] = ef_short.get("f1", 0.0)
        summary["abstain"] = abstain
        summary["wrong_on_answerable"] = wrong_on_answerable
        summary["idk_with_citation_count"] = sup.get("idk_with_citation_count", 0)
        summary["is_unans"] = is_unans
        summary["abstain_correct"] = abstain_correct
        summary["hallucinated_unans"] = hallucinated_unans

        # Dump debug prompt and raw output for first 3 when requested
        if debug_prompts and idx < 3:
            debug_dir = Path("logs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            dp = summary.get("debug_prompt", "")
            with open(
                debug_dir / f"{item['id']}_prompt.txt", "w", encoding="utf-8"
            ) as f:
                f.write(dp)
            with open(
                debug_dir / f"{item['id']}_output.txt", "w", encoding="utf-8"
            ) as f:
                f.write(summary.get("final_answer", ""))
            # Extracted citations and retrieved ids
            from agentic_rag.eval.signals import extract_citations

            with open(
                debug_dir / f"{item['id']}_citations.txt", "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(extract_citations(summary.get("final_answer", ""))))
            with open(
                debug_dir / f"{item['id']}_retrieved_ids.txt", "w", encoding="utf-8"
            ) as f:
                f.write(
                    "\n".join([str(x) for x in (summary.get("retrieved_ids") or [])])
                )
        results.append(summary)

    # Logging and reporting
    timestamp = int(time.time())
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Write JSONL summary
    jsonl_path = log_dir / f"{timestamp}_{system}.jsonl"
    with open(jsonl_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    console.print(f"Full summary logs saved to [cyan]{jsonl_path}[/cyan]")

    # Write CSV summary
    csv_path = log_dir / f"{timestamp}_{system}_summary.csv"
    df = pd.DataFrame(results)
    df["system"] = system
    df["id"] = [q["id"] for q in questions]

    # Overall Accuracy and Score (calculated after all other metrics)
    df["overall_accuracy"] = df.apply(
        lambda row: (
            row["abstain_correct"] if row["is_unans"] else row["em"]
        ),  # Assuming EM for answerable questions, or a judge score if implemented
        axis=1,
    )
    df["score"] = df["overall_accuracy"] - df["hallucinated_unans"]

    # Rename for clarity in CSV
    df = df.rename(columns={"p50_latency_ms": "latency_ms"})

    # Optionally compute RAGAS post-hoc on final outputs
    ragas_scores: list[float] = []
    if faith_report in {"ragas", "both"}:
        # Sample subset if requested
        indices = list(range(len(df)))
        if ragas_eval_subset and ragas_eval_subset > 0:
            rng = random.Random(42)
            indices = rng.sample(indices, k=min(ragas_eval_subset, len(indices)))
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness

            dataset = Dataset.from_dict(
                {
                    "question": [questions[i]["question"] for i in indices],
                    "contexts": [
                        [c.get("text", "") for c in (results[i].get("contexts") or [])]
                        for i in indices
                    ],
                    "answer": [results[i].get("final_answer", "") for i in indices],
                }
            )
            ragas_res = evaluate(dataset, metrics=[faithfulness])
            ragas_scores = (
                list(ragas_res["faithfulness"])
                if "faithfulness" in ragas_res.column_names
                or hasattr(ragas_res, "__getitem__")
                else []
            )
        except Exception:
            ragas_scores = []
        # Inflate ragas column with NaNs where not evaluated
        ragas_map = (
            {indices[i]: ragas_scores[i] for i in range(len(ragas_scores))}
            if ragas_scores
            else {}
        )
        df["final_f_ragas"] = [ragas_map.get(i, None) for i in range(len(df))]

    csv_cols = [
        "system",
        "id",
        "final_f",
        "final_f_ragas",
        "final_o",
        "em",
        "f1",
        "em_full",
        "f1_full",
        "em_short",
        "f1_short",
        "pred_source",
        "abstain",
        "wrong_on_answerable",
        "idk_with_citation_count",
        "total_tokens",
        "latency_ms",
        "rounds",
        "n_ctx_blocks",
        "context_tokens",
        "retrieved_ids",
        "action",
        "is_unans",
        "abstain_correct",
        "hallucinated_unans",
        "overall_accuracy",
        "score",
    ]

    # Only include ragas if it was computed
    if "final_f_ragas" not in df.columns:
        csv_cols.remove("final_f_ragas")

    df[csv_cols].to_csv(csv_path, index=False)
    console.print(f"CSV summary saved to [cyan]{csv_path}[/cyan]")

    # Print console summary
    table = Table(title=f"Evaluation Summary: {system.capitalize()}")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="yellow")

    summary_metrics = {"Count": str(len(df))}

    answerable_df = df[~df["is_unans"]]

    # Faithfulness reporting options
    if faith_report in {"fallback", "both"}:
        summary_metrics["Avg Faithfulness (fallback)"] = (
            f"{answerable_df['final_f'].mean():.3f}"
        )
    if faith_report in {"ragas", "both"} and "final_f_ragas" in df.columns:
        mean_r = (
            answerable_df["final_f_ragas"].dropna().mean()
            if len(answerable_df["final_f_ragas"].dropna()) > 0
            else 0.0
        )
        summary_metrics["Avg Faithfulness (RAGAS)"] = f"{mean_r:.3f}"

    summary_metrics.update(
        {
            "Avg Overlap": f"{answerable_df['final_o'].mean():.3f}",
            "Avg EM": f"{df['em'].mean():.3f}",
            "Avg F1": f"{df['f1'].mean():.3f}",
            "Abstain Rate": f"{df['abstain'].mean():.3f}",
            "Wrong-on-Answerable Rate": f"{df['wrong_on_answerable'].mean():.3f}",
            "Abstain Accuracy": f"{df['abstain_correct'].mean():.3f}",
            "Overall Accuracy": f"{df['overall_accuracy'].mean():.3f}",
            "Hallucination Rate": f"{df['hallucinated_unans'].mean():.3f}",
            "Score": f"{df['score'].mean():.3f}",
            "Avg Total Tokens": f"{df['total_tokens'].mean():.0f}",
            "P50 Latency (ms)": f"{df['latency_ms'].median():.0f}",
            "IDK+Cit Count": str(int(df["idk_with_citation_count"].sum())),
        }
    )

    if system == "agent":
        # Judge invoked rate and tokens (if present)
        judge_rate = (
            float(df.get("used_judge", pd.Series([0] * len(df))).mean())
            if "used_judge" in df.columns
            else 0.0
        )
        summary_metrics["Judge%Invoked"] = f"{judge_rate:.3f}"
        # Tokens spent by judge if tracked; default 0
        judge_tokens = 0.0
        summary_metrics["Judge Tokens / q"] = f"{judge_tokens:.1f}"

    for name, value in summary_metrics.items():
        table.add_row(name, value)

    console.print(table)


if __name__ == "__main__":
    app()
