import json

file_path = "logs/anchor/1758613608_anchor.jsonl"
target_qid = "7e250528-339f-4acd-b264-2317213def00"

completion_tokens_for_final_answer = None

with open(file_path, encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("qid") == target_qid:
            # Look for a step_log entry that contains a 'draft' and 'usage' with completion_tokens
            if (
                "draft" in entry
                and entry.get("usage")
                and entry["usage"].get("completion_tokens") is not None
            ):
                # Store this, as it's the latest generation step we've seen
                completion_tokens_for_final_answer = entry["usage"].get(
                    "completion_tokens"
                )

if completion_tokens_for_final_answer is not None:
    print(f"QID: {target_qid}")
    print(f"Completion tokens for final answer: {completion_tokens_for_final_answer}")
else:
    print(f"Could not find completion tokens for final answer for QID: {target_qid}")
