import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Assuming `settings` is available globally or passed in context
# from agentic_rag.config import settings # Will import if needed later


def log_baug_round(
    log_dir: str,
    q_id: str,
    round_idx: int,
    action: str,
    reasons: List[str],
    signals: Dict[str, Any],
    final_decision: Dict[str, Any],
):
    """
    Logs the BAUG decision and relevant signals for a given round.

    Args:
        log_dir: The base directory for logs.
        q_id: The ID of the current question.
        round_idx: The current round number.
        action: The action taken by BAUG (e.g., "GO", "SKIP", "STOP", "ABSTAIN").
        reasons: A list of reasons for the BAUG decision.
        signals: A dictionary of relevant signals (overlap_est, faith_est, etc.).
        final_decision: The final decision dict from the gate, including thresholds.
    """
    # Create a unique filename for this question's telemetry, or append to an existing one
    log_file = Path(log_dir) / "baug_telemetry" / f"{q_id}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "q_id": q_id,
        "round_idx": round_idx,
        "action": action,
        "reasons": reasons,
        "signals": signals,
        "final_decision": final_decision,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def aggregate_telemetry_results(telemetry_dir: str) -> Dict[str, Any]:
    """
    Aggregates telemetry logs to compute overall statistics.

    Args:
        telemetry_dir: The directory containing individual question telemetry logs.

    Returns:
        A dictionary of aggregated statistics (e.g., STOP@R1 rate, round histogram).
    """
    all_logs = []
    telemetry_path = Path(telemetry_dir) / "baug_telemetry"
    if not telemetry_path.exists():
        return {"error": "Telemetry directory not found"}

    for log_file in telemetry_path.glob("*.jsonl"):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                all_logs.append(json.loads(line))

    # Initialize aggregation metrics
    total_questions = len(set(log["q_id"] for log in all_logs))
    stop_at_r1_count = 0
    round_counts: Counter[int] = Counter()
    total_tokens_per_round: Counter[int] = Counter()
    total_latency_per_round: Counter[int] = Counter()
    question_final_rounds: dict[str, int] = {}
    question_tokens: Counter[str] = Counter()
    question_latency: Counter[str] = Counter()

    for log in all_logs:
        q_id = log["q_id"]
        round_idx = log["round_idx"]
        signals = log["signals"]

        round_counts[round_idx] += 1
        total_tokens_per_round[round_idx] += (
            signals.get("tokens_last_round", 0) if "tokens_last_round" in signals else 0
        )
        total_latency_per_round[round_idx] += (
            signals.get("latency_ms", 0) if "latency_ms" in signals else 0
        )

        # Track the final round for each question
        if q_id not in question_final_rounds or round_idx > question_final_rounds[q_id]:
            question_final_rounds[q_id] = round_idx

        # Track total tokens and latency per question
        question_tokens[q_id] += (
            signals.get("tokens_last_round", 0) if "tokens_last_round" in signals else 0
        )
        question_latency[q_id] += (
            signals.get("latency_ms", 0) if "latency_ms" in signals else 0
        )

    # Calculate STOP@R1 rate
    for q_id, final_round in question_final_rounds.items():
        if final_round == 1 and any(
            log["action"] == "STOP" and log["q_id"] == q_id for log in all_logs
        ):
            stop_at_r1_count += 1

    stop_at_r1_rate = (
        (stop_at_r1_count / total_questions) * 100 if total_questions > 0 else 0
    )

    # Histogram of rounds
    round_histogram = {f"Round {r}": count for r, count in sorted(round_counts.items())}

    # Average tokens and latency per round
    avg_tokens_per_round = {
        f"Round {r}": (
            total_tokens_per_round[r] / round_counts[r] if round_counts[r] > 0 else 0
        )
        for r in sorted(round_counts.keys())
    }
    avg_latency_per_round = {
        f"Round {r}": (
            total_latency_per_round[r] / round_counts[r] if round_counts[r] > 0 else 0
        )
        for r in sorted(round_counts.keys())
    }

    return {
        "total_questions_logged": total_questions,
        "stop_at_r1_rate": f"{stop_at_r1_rate:.1f}%",
        "round_distribution": round_histogram,
        "avg_tokens_per_round": avg_tokens_per_round,
        "avg_latency_per_round": avg_latency_per_round,
        "total_tokens_sum": sum(question_tokens.values()),
        "total_latency_sum_ms": sum(question_latency.values()),
        # Add more aggregations as needed
    }
