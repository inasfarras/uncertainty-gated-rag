import json
from pathlib import Path

path = Path("data/crag_questions_full.jsonl")
with path.open("r", encoding="utf-8") as handle:
    for i, line in enumerate(handle, 1):
        s = line.strip()
        if not s:
            continue
        try:
            json.loads(s)
        except Exception:
            print("fail line", i, "repr", repr(s[:20]))
            raise
print("all ok")
