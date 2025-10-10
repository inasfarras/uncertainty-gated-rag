import json
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Literal, Optional, cast

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
from agentic_rag.supervisor.orchestrator import AnchorSystem

# Built-in lightweight profiles to shorten long commands
PROFILES: dict[str, dict[str, Any]] = {
    # Balanced quality/speed for AnchorSystem
    "anchor_balanced": {
        "system": "anchor",
        "gate_on": True,
        "max_rounds": 2,
        "judge_policy": "gray_zone",
        # Keep backend unchanged by default; user can pass --backend
        "overrides": (
            "USE_HYBRID_SEARCH=False "
            "RETRIEVAL_K=8 PROBE_FACTOR=1 RETRIEVAL_POOL_K=24 "
            "MAX_CONTEXT_TOKENS=1100 MAX_OUTPUT_TOKENS=90 "
            "MMR_LAMBDA=0.0 MAX_WORKERS=6"
        ),
    },
    # Baseline plain/vanilla (keeps things simple and deterministic)
    "baseline_plain": {
        "system": "baseline",
        "gate_on": False,  # N/A for baseline; kept for uniformity
        "max_rounds": 1,
        "judge_policy": "never",
        "overrides": (
            "USE_HYBRID_SEARCH=False USE_RERANK=False "
            "RETRIEVAL_K=8 PROBE_FACTOR=1 RETRIEVAL_POOL_K=8 "
            "MAX_CONTEXT_TOKENS=900 MAX_OUTPUT_TOKENS=60 "
            "MMR_LAMBDA=0.0 MAX_WORKERS=6"
        ),
    },
    # Balanced profile but with BAUG/Gate OFF (keeps overrides identical)
    "anchor_balanced_off": {
        "system": "anchor",
        "gate_on": False,
        "max_rounds": 2,
        "judge_policy": "gray_zone",
        "overrides": (
            "USE_HYBRID_SEARCH=False "
            "RETRIEVAL_K=8 PROBE_FACTOR=1 RETRIEVAL_POOL_K=24 "
            "MAX_CONTEXT_TOKENS=1100 MAX_OUTPUT_TOKENS=90 "
            "MMR_LAMBDA=0.0 MAX_WORKERS=6"
        ),
    },
    # Fastest single-round factoid-style pass
    "anchor_fast": {
        "system": "anchor",
        "gate_on": True,
        "max_rounds": 1,
        "judge_policy": "never",
        "overrides": (
            "USE_HYBRID_SEARCH=False "
            "RETRIEVAL_K=8 PROBE_FACTOR=1 RETRIEVAL_POOL_K=16 "
            "MAX_CONTEXT_TOKENS=900 MAX_OUTPUT_TOKENS=60 "
            "MMR_LAMBDA=0.0"
        ),
    },
    # Minimal hybrid (vector+BM25) at small pool sizes
    "anchor_hybrid_light": {
        "system": "anchor",
        "gate_on": True,
        "max_rounds": 2,
        "judge_policy": "gray_zone",
        "overrides": (
            "USE_HYBRID_SEARCH=True HYBRID_ALPHA=0.7 "
            "RETRIEVAL_K=8 PROBE_FACTOR=1 RETRIEVAL_POOL_K=24 "
            "MAX_CONTEXT_TOKENS=1100 MAX_OUTPUT_TOKENS=90 "
            "MMR_LAMBDA=0.0"
        ),
    },
}

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
    "baseline", "--system", help="System to run: 'baseline' | 'agent' | 'anchor'."
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
    True,
    "--use-final-short/--no-use-final-short",
    help="Score EM/F1 using best-of (final_short vs full answer)",
)

# Generic overrides: KEY=VAL pairs (e.g., MAX_ROUNDS=1 USE_HYDE=False)
_override_option = typer.Option(
    None,
    "--override",
    help="Override settings as KEY=VAL pairs (repeatable)",
    show_default=False,
)

# Optional file with KEY=VAL overrides, one per line (comments with #)
_override_file_option = typer.Option(
    None,
    "--override-file",
    help="Path to file with KEY=VAL overrides (one per line)",
    show_default=False,
)

# Named profile to shorten commands
_profile_option = typer.Option(
    None,
    "--profile",
    help=(
        "Named profile: anchor_fast | anchor_balanced | anchor_balanced_off | "
        "anchor_hybrid_light | baseline_plain"
    ),
    case_sensitive=False,
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
    override: Optional[list[str]] = _override_option,
    override_file: Optional[Path] = _override_file_option,
    profile: Optional[str] = _profile_option,
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

    # Apply profile (if any) to set defaults and append overrides
    if profile:
        prof = PROFILES.get(profile.lower())
        if not prof:
            console.print(
                f"[bold red]Unknown profile: {profile}. Choose from: {', '.join(PROFILES)}[/bold red]"
            )
            raise typer.Exit(1)
        # Enforce intended system for the profile
        if prof.get("system") and system != prof["system"]:
            console.print(
                f"[yellow]Profile '{profile}' is designed for system '{prof['system']}'. Overriding system.[/yellow]"
            )
            system = prof["system"]
        # Apply profile-level flags
        gate_on = bool(prof.get("gate_on", gate_on))
        max_rounds = int(prof.get("max_rounds", max_rounds or settings.MAX_ROUNDS))
        judge_policy = str(prof.get("judge_policy", judge_policy))
        # Append profile overrides
        if prof.get("overrides"):
            override = list(override or []) + [str(prof["overrides"])]

    # Apply generic overrides (CLI and/or file)
    def _coerce(v: str):
        vl = v.strip()
        if vl.lower() in {"true", "false"}:
            return vl.lower() == "true"
        try:
            if "." in vl:
                return float(vl)
            return int(vl)
        except Exception:
            return vl

    # Load override file if provided
    if override_file and override_file.exists():
        try:
            lines = []
            for raw in override_file.read_text(encoding="utf-8").splitlines():
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                lines.append(s)
            if lines:
                override = list(override or []) + [" ".join(lines)]
        except Exception:
            pass

    if override:
        # Support space-separated list or repeated flags
        for pair in override:
            for token in str(pair).split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                key = k.strip()
                val = _coerce(v)
                # Map common alias
                if key == "USE_RERANKER":
                    key = "USE_RERANK"
                if hasattr(settings, key):
                    setattr(settings, key, val)

    # Decide log directory by system and gate flag
    def _folder_for_run(sys_name: str, gate: bool) -> str:
        if sys_name == "baseline":
            return "logs/baseline"
        if sys_name == "agent":
            return "logs/agent_gate_on" if gate else "logs/agent_gate_off"
        if sys_name == "anchor":
            return "logs/anchor"
        return "logs/other"

    # Support new lowercase setting name; keep backward-compat
    settings.log_dir = _folder_for_run(system, gate_on)

    # Load dataset
    with open(dataset, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    if n > 0:
        questions = questions[:n]

    # Select system
    model: Any
    if system == "agent":
        model = Agent(gate_on=gate_on, debug_mode=debug_prompts)
    elif system == "baseline":
        model = Baseline(debug_mode=debug_prompts)
    elif system == "anchor":
        model = AnchorSystem(debug_mode=debug_prompts)
    else:
        console.print(f"[bold red]Unknown system: {system}[/bold red]")
        raise typer.Exit(1)

    # Agent uses UncertaintyGate; Anchor uses BAUG (or can be turned OFF via gate flag)
    if system == "agent":
        gate_note = "ON" if gate_on else "OFF"
    elif system == "anchor":
        # Allow gate flag to control Anchor gate, too
        from agentic_rag.config import settings as _settings

        _settings.ANCHOR_GATE_ON = bool(gate_on)
        gate_note = "BAUG" if _settings.ANCHOR_GATE_ON else "OFF"
    else:
        gate_note = "N/A"
    console.print(
        f"[bold green]Running evaluation for system: '{system}' with gate {gate_note}[/bold green]"
    )
    console.print(
        f"[bold]Logs directory:[/bold] {getattr(settings, 'log_dir', 'logs')}"
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

        # Normalize gold to string for robust checks (gold can be non-string, e.g., int)
        gold_raw = item.get("gold", "")
        gold_text = str(gold_raw) if gold_raw is not None else ""
        is_unans = gold_text.strip().lower() in {
            "invalid question",
            "n/a",
            "unknown",
            "",
        }

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
        ef_full = em_f1(full_pred, gold_text)
        ef_short = (
            em_f1(short_pred, gold_text) if short_pred else {"em": 0.0, "f1": 0.0}
        )
        # Choose scoring strategy: prefer final_short when available
        if short_pred:
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
        # Pass normalized gold_text to avoid attribute errors on non-strings
        final_f = faithfulness_fallback(pred_for_scoring, gold_text, final_o)
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
            debug_dir = Path(getattr(settings, "log_dir", "logs")) / "debug"
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
    log_dir = Path(getattr(settings, "log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL summary
    file_tag = (
        system
        if system in {"baseline", "anchor"}
        else ("agent_gate_on" if gate_on and system == "agent" else "agent_gate_off")
    )
    jsonl_path = log_dir / f"{timestamp}_{file_tag}.jsonl"
    with open(jsonl_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    console.print(f"Full summary logs saved to [cyan]{jsonl_path}[/cyan]")

    # Write CSV summary
    csv_path = log_dir / f"{timestamp}_{file_tag}_summary.csv"
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

    # Optional columns for anchor system
    if "anchor_coverage" in df.columns:
        csv_cols.append("anchor_coverage")
    if "conflict_risk" in df.columns:
        csv_cols.append("conflict_risk")
    if "baug_reasons" in df.columns:
        csv_cols.append("baug_reasons")
    if "used_judge" in df.columns and "used_judge" not in csv_cols:
        csv_cols.append("used_judge")

    df[csv_cols].to_csv(csv_path, index=False)
    console.print(f"CSV summary saved to [cyan]{csv_path}[/cyan]")

    # Write a QA pairs log (qid, question, agent answers, gold, alt_ans if present)
    try:
        qa_path = log_dir / f"{timestamp}_{file_tag}_qa_pairs.csv"
        # Build a lookup from dataset for gold/alt answers
        gold_map: dict[str, dict] = {}
        for it in questions:
            qid = str(it.get("id"))
            gold_map[qid] = {
                "question": it.get("question", ""),
                # prefer 'answer' if present, else 'gold'
                "gold": it.get("answer", it.get("gold", "")),
                "alt_ans": it.get("alt_ans", []),
            }
        # Optional: load metadata to resolve URLs of cited docs
        from agentic_rag.data.meta import load_meta

        meta_map = load_meta()
        import csv as _csv

        from agentic_rag.eval.signals import extract_citations

        with open(qa_path, "w", newline="", encoding="utf-8") as fqa:
            writer = _csv.writer(fqa)
            writer.writerow(
                [
                    "id",
                    "question",
                    "agent_final_answer",
                    "agent_final_short",
                    "gold",
                    "alt_ans",
                    "cited_doc_ids",
                    "cited_urls",
                    "cited_titles",
                ]
            )
            for res in results:
                qid = str(res.get("qid"))
                gm = gold_map.get(qid, {"question": "", "gold": "", "alt_ans": []})
                # Resolve citations
                cids = extract_citations(res.get("final_answer", "") or "")
                urls, titles = [], []
                for cid in cids:
                    m = meta_map.get(cid, {})
                    if m.get("url"):
                        urls.append(str(m.get("url")))
                    if m.get("title"):
                        titles.append(str(m.get("title")))
                writer.writerow(
                    [
                        qid,
                        gm.get("question", ""),
                        res.get("final_answer", ""),
                        res.get("final_short", ""),
                        gm.get("gold", ""),
                        "; ".join(
                            gm.get("alt_ans", [])
                            if isinstance(gm.get("alt_ans"), list)
                            else []
                        ),
                        "; ".join(cids),
                        "; ".join(urls),
                        "; ".join(titles),
                    ]
                )
        console.print(f"QA pairs saved to [cyan]{qa_path}[/cyan]")
    except Exception:
        pass

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

    detailed_metrics = {
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
    omit_from_overview = {
        "Wrong-on-Answerable Rate",
        "Abstain Accuracy",
        "Overall Accuracy",
        "Hallucination Rate",
        "Score",
    }
    summary_metrics.update(
        {k: v for k, v in detailed_metrics.items() if k not in omit_from_overview}
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

    # Write a Markdown summary alongside CSV/JSONL for convenient reporting
    try:
        md_path = log_dir / f"{timestamp}_{file_tag}_summary.md"
        # Minimal markdown table from summary_metrics
        summary_lines = [
            f"# Evaluation Summary: {system}",
            "",
            f"- Dataset: `{dataset}`",
            f"- Count: {len(df)}",
            f"- Backend: `{backend}`",
            f"- Judge Policy: `{judge_policy}`",
            f"- Gate: {gate_note}",
            f"- JSONL: `{jsonl_path}`",
            f"- CSV: `{csv_path}`",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for name, value in summary_metrics.items():
            summary_lines.append(f"| {name} | {value} |")

        # Add override details if provided
        if override:
            summary_lines.extend(
                [
                    "",
                    "## Overrides",
                    "",
                    "```",
                    " ".join(override),
                    "```",
                ]
            )

        md_content = "\n".join(summary_lines) + "\n"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        console.print(f"Markdown summary saved to [cyan]{md_path}[/cyan]")
    except Exception:
        # Never fail the run because of markdown reporting
        pass


if __name__ == "__main__":
    app()
