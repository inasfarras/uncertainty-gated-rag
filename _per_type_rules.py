import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from agentic_rag.intent.rules import interpret_with_rules

paths = {
    "baseline": Path(r"logs/baseline/1760553674_baseline.jsonl"),
    "anchor_baug_on": Path(r"logs/anchor/1760582178_anchor.jsonl"),
    "anchor_baug_off": Path(r"logs/anchor/1760621536_anchor.jsonl"),
}

qs = {}
with open(r"data/crag_questions_full.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            qs[row["id"]] = row["question"]

# Precompute type map using rules
label = {}
for qid, q in qs.items():
    intent = interpret_with_rules(q)
    if intent.task_type == "list":
        typ = "List"
    elif "unit" in intent.slots:
        typ = "Numeric"
    elif ("year" in intent.slots) or ("time_window" in intent.slots):
        typ = "Temporal"
    else:
        typ = "Factoid"
    label[qid] = typ

cols = ("f1", "final_f", "abstain", "total_tokens", "rounds")
for name, p in paths.items():
    agg: dict[str, dict[str, list[float]]] = {
        t: {k: [] for k in cols} for t in ("Factoid", "List", "Numeric", "Temporal")
    }
    cnt = {t: 0 for t in agg}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid")
            t = label.get(qid, "Factoid")
            cnt[t] += 1
            for k in cols:
                v = row.get(k)
                if v is not None:
                    try:
                        agg[t][k].append(float(v))
                    except Exception:
                        pass
    print(f"CONFIG: {name}")
    for t in ("Factoid", "List", "Numeric", "Temporal"):
        n = cnt[t]
        if n == 0:
            continue
        print(
            f"{t}: n={n}, F1={sum(agg[t]['f1'])/max(1,len(agg[t]['f1'])):.3f}, Faith={sum(agg[t]['final_f'])/max(1,len(agg[t]['final_f'])):.3f}, Abstain={100*sum(agg[t]['abstain'])/max(1,len(agg[t]['abstain'])):.1f}%, Tokens={sum(agg[t]['total_tokens'])/max(1,len(agg[t]['total_tokens'])):.0f}, Rounds={sum(agg[t]['rounds'])/max(1,len(agg[t]['rounds'])):.2f}"
        )
    print()
