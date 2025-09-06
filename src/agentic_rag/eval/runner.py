import json
import time
from pathlib import Path
from typing import Literal, Optional, cast

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from agentic_rag.agent.loop import Agent, Baseline
from agentic_rag.config import settings

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
):
    """Runs an evaluation of the Agentic RAG system."""

    # Update settings from CLI
    settings.EMBED_BACKEND = cast(Literal["openai", "st"], backend)
    override_settings(tau_f, tau_o, max_rounds)

    # Load dataset
    with open(dataset) as f:
        questions = [json.loads(line) for line in f]
    if n > 0:
        questions = questions[:n]

    # Select system
    model: Agent | Baseline
    if system == "agent":
        model = Agent(gate_on=gate_on)
    elif system == "baseline":
        model = Baseline()
    else:
        console.print(f"[bold red]Unknown system: {system}[/bold red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Running evaluation for system: '{system}' with gate {'ON' if gate_on and system=='agent' else 'OFF'}[/bold green]"
    )

    # Run evaluation
    results = []
    for item in questions:
        console.print(f"Processing qid: {item['id']}")
        summary = model.answer(question=item["question"], qid=item["id"])
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

    # Rename for clarity in CSV
    df = df.rename(columns={"p50_latency_ms": "latency_ms"})

    csv_cols = [
        "system",
        "id",
        "final_f",
        "final_o",
        "total_tokens",
        "latency_ms",
        "rounds",
    ]
    df[csv_cols].to_csv(csv_path, index=False)
    console.print(f"CSV summary saved to [cyan]{csv_path}[/cyan]")

    # Print console summary
    table = Table(title=f"Evaluation Summary: {system.capitalize()}")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="yellow")

    table.add_row("Count", str(len(df)))
    table.add_row("Avg Faithfulness", f"{df['final_f'].mean():.3f}")
    table.add_row("Avg Overlap", f"{df['final_o'].mean():.3f}")
    table.add_row("Avg Total Tokens", f"{df['total_tokens'].mean():.0f}")
    table.add_row("P50 Latency (ms)", f"{df['latency_ms'].median():.0f}")

    console.print(table)


if __name__ == "__main__":
    app()
