import json

# Load the set of QIDs we want to keep
with open("data/anchor_qids.jsonl") as f:
    qids_to_keep = {json.loads(line)["qid"] for line in f}

print(f"Loaded {len(qids_to_keep)} QIDs to filter the dataset with.")

# Filter the main question file
count = 0
with open("data/crag_questions_full.jsonl", encoding="utf-8") as f_in, open(
    "data/baseline_fix_questions.jsonl", "w", encoding="utf-8"
) as f_out:
    for line in f_in:
        item = json.loads(line)
        if item.get("id") in qids_to_keep:
            f_out.write(line)
            count += 1
            if count % 100 == 0:
                print(f"Found {count} matching questions...", end="\\r")

print(f"\\nFinished. Wrote {count} questions to data/baseline_fix_questions.jsonl")
