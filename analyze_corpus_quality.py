"""Analyze CRAG corpus quality - file sizes, content quality, etc."""

import json
from collections import Counter
from pathlib import Path


def analyze_corpus():
    corpus_dir = Path("data/crag_corpus_html")
    meta_file = Path("data/crag_meta.jsonl")

    # Load metadata
    meta_map = {}
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            doc_id = entry["doc_id"]
            meta_map[doc_id] = entry

    files = list(corpus_dir.glob("*.txt"))
    print(f"📁 Total files: {len(files)}")
    print(f"📋 Metadata entries: {len(meta_map)}")
    print()

    # Analyze file sizes
    sizes = []
    tiny_files = []  # < 500 chars
    small_files = []  # 500-2000 chars
    medium_files = []  # 2k-10k
    large_files = []  # > 10k

    for f in files:
        size = f.stat().st_size
        sizes.append(size)

        doc_id = f.stem
        if size < 500:
            tiny_files.append((doc_id, size))
        elif size < 2000:
            small_files.append((doc_id, size))
        elif size < 10000:
            medium_files.append((doc_id, size))
        else:
            large_files.append((doc_id, size))

    print("📊 FILE SIZE DISTRIBUTION:")
    print(
        f"  Tiny (<500B):       {len(tiny_files):4d} ({len(tiny_files)/len(files)*100:.1f}%) - likely junk"
    )
    print(
        f"  Small (500B-2KB):   {len(small_files):4d} ({len(small_files)/len(files)*100:.1f}%) - low quality"
    )
    print(
        f"  Medium (2KB-10KB):  {len(medium_files):4d} ({len(medium_files)/len(files)*100:.1f}%) - okay"
    )
    print(
        f"  Large (>10KB):      {len(large_files):4d} ({len(large_files)/len(files)*100:.1f}%) - good"
    )
    print()

    avg_size = sum(sizes) / len(sizes)
    median_size = sorted(sizes)[len(sizes) // 2]
    print(f"  Average file size: {avg_size/1024:.1f} KB")
    print(f"  Median file size:  {median_size/1024:.1f} KB")
    print()

    # Sample tiny files
    print("🔍 SAMPLE TINY FILES (likely junk):")
    for doc_id, size in tiny_files[:5]:
        meta = meta_map.get(doc_id, {})
        url = meta.get("url", "unknown")
        print(f"  {doc_id}: {size}B")
        print(f"    URL: {url}")
        # Read first 150 chars
        content = Path(corpus_dir / f"{doc_id}.txt").read_text(encoding="utf-8")[:150]
        print(f"    Content: {content[:100]}...")
        print()

    # Analyze by source domain
    print("🌐 DOCUMENTS BY SOURCE DOMAIN:")
    domains = Counter()
    for _, entry in meta_map.items():
        url = entry.get("url", "")
        if url:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc
            domains[domain] += 1

    for domain, count in domains.most_common(20):
        print(f"  {domain:40s} {count:3d} docs")
    print()

    # Check questions coverage
    questions_file = Path("data/crag_questions.jsonl")
    qids = set()
    with open(questions_file, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qids.add(q["id"])

    docs_per_q = Counter()
    for entry in meta_map.values():
        qid = entry["qid"]
        docs_per_q[qid] += 1

    print("📈 DOCUMENTS PER QUESTION:")
    print(f"  Total questions: {len(qids)}")
    print(f"  Questions with docs: {len(docs_per_q)}")
    print(f"  Average docs/question: {sum(docs_per_q.values()) / len(docs_per_q):.1f}")
    print(f"  Min docs/question: {min(docs_per_q.values())}")
    print(f"  Max docs/question: {max(docs_per_q.values())}")
    print()

    # Questions with few docs
    sparse_qs = [(qid, count) for qid, count in docs_per_q.items() if count < 3]
    print(f"⚠️  Questions with <3 docs: {len(sparse_qs)}")

    # Sample sparse questions
    if sparse_qs:
        print("\n🔍 SAMPLE SPARSE QUESTIONS:")
        with open(questions_file, encoding="utf-8") as f:
            q_map = {json.loads(line)["id"]: json.loads(line) for line in f}

        for qid, count in sparse_qs[:5]:
            q_data = q_map.get(qid, {})
            print(f"  QID: {qid}")
            print(f"  Question: {q_data.get('question', 'unknown')[:100]}...")
            print(f"  Gold: {q_data.get('gold', 'unknown')}")
            print(f"  Docs: {count}")
            print()


if __name__ == "__main__":
    analyze_corpus()
