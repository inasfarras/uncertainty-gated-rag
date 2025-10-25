import json

import numpy as np
import pandas as pd


def classify_question(question_text):
    """Classifies a question into Factoid, List, Numeric, or Temporal."""
    q = question_text.lower()

    # Temporal questions
    temporal_keywords = [
        "when did",
        "what year",
        "on which date",
        "in 20",
        "what was the date",
        "release date",
    ]
    if any(kw in q for kw in temporal_keywords):
        return "Temporal"

    # Numeric questions
    numeric_keywords = [
        "how many",
        "what percentage",
        "how much",
        "what is the population",
        "what is the ratio",
        "how long is",
        "how old",
        "how big is",
    ]
    if any(kw in q for kw in numeric_keywords):
        return "Numeric"

    # List questions
    list_keywords = [
        "list the",
        "what are the",
        "name all",
        "who were",
        "who are",
        "give me",
        "show me",
        "what teams",
        "what movies",
    ]
    if any(kw in q for kw in list_keywords):
        return "List"

    # Default to Factoid
    return "Factoid"


def analyze_logs_by_type():
    """Loads logs, classifies questions, and aggregates metrics by type."""
    log_files = {
        "Baseline RAG": "logs/baseline/1760553674_baseline.jsonl",
        "Anchor (BAUG ON)": "logs/anchor/1760582178_anchor.jsonl",
        "Anchor (BAUG OFF)": "logs/anchor/1760621536_anchor.jsonl",
    }

    all_data = []
    for config_name, log_path in log_files.items():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    record["config"] = config_name
                    all_data.append(record)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(all_data)

    # Add question type
    df["question_type"] = df["question"].apply(classify_question)

    # Handle abstain rate (abstain is 1 if final_answer is "I don't know", else 0)
    df["abstain"] = np.where(df["final_answer"].str.lower() == "i don't know", 1, 0)

    # Ensure numeric types
    metrics = ["f1", "final_f", "total_tokens", "n_rounds", "abstain"]
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename columns for clarity in the final table
    df.rename(
        columns={
            "final_f": "Faithfulness",
            "f1": "F1 Score",
            "total_tokens": "Avg. Tokens",
            "n_rounds": "Avg. Rounds",
            "abstain": "Abstain Rate",
        },
        inplace=True,
    )

    # Group and aggregate
    summary = (
        df.groupby(["question_type", "config"])
        .agg(
            {
                "F1 Score": "mean",
                "Faithfulness": "mean",
                "Abstain Rate": "mean",
                "Avg. Tokens": "mean",
                "Avg. Rounds": "mean",
            }
        )
        .reset_index()
    )

    # Reorder for presentation
    summary = summary.set_index(["question_type", "config"]).sort_index()

    # Format for display
    summary["F1 Score"] = summary["F1 Score"].map("{:.3f}".format)
    summary["Faithfulness"] = summary["Faithfulness"].map("{:.3f}".format)
    summary["Abstain Rate"] = (summary["Abstain Rate"] * 100).map("{:.1f}%".format)
    summary["Avg. Tokens"] = summary["Avg. Tokens"].map("{:,.0f}".format)
    summary["Avg. Rounds"] = summary["Avg. Rounds"].map("{:.2f}".format)

    print("--- Per-Question-Type Analysis Results ---")
    print(summary)


if __name__ == "__main__":
    analyze_logs_by_type()
