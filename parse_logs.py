import json
import os


def parse_agent_logs():
    """Parses agent log files to extract the last run's summary and per-round data."""

    qids = [
        "7bb29eb4-12f9-45f9-bf8a-66832b3c8962",
        "a2486535-98e7-4876-9880-80374eac2fa8",
        "161a89f3-7a70-4e12-a1a2-7832e098b0a7",
        "30395ec2-4247-4a23-8003-37dcfdf1531c",
        "8163a6f0-3238-4a69-ba60-a4a06090bc6f",
        "06511362-0742-4608-8fdf-d192d0d53743",
        "99150c8b-556e-4800-b9e6-d33e6b5adfa5",
        "74ec83d9-0ebc-4f5e-b738-7cffbf262e74",
        "9864fff2-af82-495c-ab18-a6c589635d29",
        "aa838149-4446-4917-be80-945afe5287f8",
    ]

    question_map = {
        "7bb29eb4-12f9-45f9-bf8a-66832b3c8962": "how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?",
        "a2486535-98e7-4876-9880-80374eac2fa8": "what is a movie to feature a person who can create and control a device that can manipulate the laws of physics?",
        "161a89f3-7a70-4e12-a1a2-7832e098b0a7": "where did the ceo of salesforce previously work?",
        "30395ec2-4247-4a23-8003-37dcfdf1531c": "which movie won the oscar best visual effects in 2021?",
        "8163a6f0-3238-4a69-ba60-a4a06090bc6f": "in 2004, which animated film was recognized with the best animated feature film oscar?",
        "06511362-0742-4608-8fdf-d192d0d53743": "what is the average gross for the top 3 pixar movies?",
        "99150c8b-556e-4800-b9e6-d33e6b5adfa5": "what are the countries that are located in southern africa.",
        "74ec83d9-0ebc-4f5e-b738-7cffbf262e74": "what's the cooling source of the koeberg nuclear power station?",
        "9864fff2-af82-495c-ab18-a6c589635d29": "when did hamburg become the biggest city of germany?",
        "aa838149-4446-4917-be80-945afe5287f8": "how much did voyager therapeutics's stock rise in value over the past month?",
    }

    summary_log_file = "logs/1758190522_agent.jsonl"
    main_summaries = {}
    with open(summary_log_file) as f:
        for line in f:
            log = json.loads(line)
            main_summaries[log["qid"]] = log

    log_dir = "logs"

    final_table_data = []

    for qid in qids:
        log_file = os.path.join(log_dir, f"agent_{qid}.jsonl")

        question_info = {
            "qid": qid,
            "question": question_map.get(qid, "Unknown Question"),
            "summary": None,
            "rounds": [],
        }

        if not os.path.exists(log_file):
            print(f"Log file not found for qid: {qid}")
            final_table_data.append(question_info)
            continue

        all_entries = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        summary = None
        summary_index = -1
        for i in range(len(all_entries) - 1, -1, -1):
            if "final_answer" in all_entries[i] and "round" not in all_entries[i]:
                summary = all_entries[i]
                summary_index = i
                break

        if not summary:
            print(f"No summary found for qid: {qid}")
            final_table_data.append(question_info)
            continue

        question_info["summary"] = summary

        n_rounds = summary.get("n_rounds", 0)

        # The round logs are the n_rounds entries before the summary
        start_index = max(0, summary_index - n_rounds)
        rounds_data = all_entries[start_index:summary_index]
        question_info["rounds"] = rounds_data

        final_table_data.append(question_info)

    # Now print the formatted table
    for item in final_table_data:
        print(f"--- QID: {item['qid']} ---")
        print(f"Question: {item['question']}")

        summary = item["summary"]
        if not summary:
            print("No summary data found.")
            print("-" * (len(item["qid"]) + 12))
            continue

        final_action = summary.get("final_action", "N/A")
        n_rounds = summary.get("n_rounds", "N/A")
        total_tokens = summary.get("total_tokens", "N/A")
        latency_ms = summary.get("p50_latency_ms", "N/A")
        idk_violation = main_summaries.get(item["qid"], {}).get(
            "idk_with_citation_count", 0
        )

        latency_ms_str = f"{latency_ms:.2f}" if isinstance(latency_ms, float) else "N/A"
        print(
            f"final_action: {final_action}, n_rounds: {n_rounds}, total_tokens: {total_tokens}, latency_ms: {latency_ms_str}"
        )
        print(f"idk_with_citation_violation: {idk_violation}")

        for r_data in item["rounds"]:
            round_num = r_data.get("round")
            action = r_data.get("action", "N/A")
            reason = r_data.get("reason", "N/A")
            new_hits = len(r_data.get("new_hits", []))
            new_hits_ratio = r_data.get("new_hits_ratio", 0.0)
            overlap_est = r_data.get("overlap_est", 0.0)
            faith_est = r_data.get("f", 0.0)
            context_tokens = r_data.get("context_tokens", "N/A")
            gen_tokens = r_data.get("gen_tokens", "N/A")

            print(f"  Round {round_num}:")
            print(
                f"    action: {action}, reason: {reason}, new_hits: {new_hits}, new_hits_ratio: {new_hits_ratio:.2f}, overlap_est: {overlap_est:.2f}, faith_est: {faith_est:.2f}, context_tokens: {context_tokens}, gen_tokens: {gen_tokens}"
            )
        print("-" * (len(item["qid"]) + 12))


if __name__ == "__main__":
    parse_agent_logs()
