"""Count total questions in CRAG dataset."""

import bz2

print("Counting questions in full CRAG dataset...")
count = 0
with bz2.open("data/crag_task_1_and_2_dev_v4.jsonl.bz2", "rt", encoding="utf-8") as f:
    for _ in f:
        count += 1
        if count % 100 == 0:
            print(f"  Counted {count}...")

print(f"\n✅ Total questions in dataset: {count}")
print("\n📊 Potential corpus size (5 results/question):")
print(f"   Total search results: ~{count * 5}")
print(f"   Expected documents: ~{count * 4.8:.0f} (assuming 95% success rate)")
print(f"   Expected chunks: ~{count * 4.8 * 25:.0f} (assuming 25 chunks/doc)")
