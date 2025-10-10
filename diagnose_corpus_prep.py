"""Diagnose why corpus preparation creates so few documents."""

import bz2
import json
import re
from collections import Counter
from typing import List, TypedDict

from bs4 import BeautifulSoup


class Stats(TypedDict):
    total_questions: int
    questions_processed: int
    questions_skipped_dynamic: int
    total_search_results: int
    docs_written: int
    skipped_too_short: int
    skipped_no_html: int
    skipped_max_pages: int
    html_lengths: List[int]
    text_lengths: List[int]


def html2text(html: str) -> str:
    """Same function as in prepare_crag_from_jsonl.py"""
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    txt = soup.get_text("\n")
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


print("=" * 80)
print("CORPUS PREPARATION DIAGNOSTIC")
print("=" * 80)

# Parameters from your script
MIN_CHARS = 200
MAX_PAGES_PER_Q = 80
STATIC_ONLY = False  # You ran without --static-only last time
N = 200

# Track statistics
stats: Stats = {
    "total_questions": 0,
    "questions_processed": 0,
    "questions_skipped_dynamic": 0,
    "total_search_results": 0,
    "docs_written": 0,
    "skipped_too_short": 0,
    "skipped_no_html": 0,
    "skipped_max_pages": 0,
    "html_lengths": [],
    "text_lengths": [],
}

docs_per_question: List[int] = []

# Process dataset
with bz2.open("data/crag_task_1_and_2_dev_v4.jsonl.bz2", "rt", encoding="utf-8") as fin:
    for _, line in enumerate(fin):
        if stats["questions_processed"] >= N:
            break

        ex = json.loads(line)
        stats["total_questions"] += 1

        tag = (ex.get("static_or_dynamic") or "").lower()
        if STATIC_ONLY and tag in {"real-time", "fast-changing"}:
            stats["questions_skipped_dynamic"] += 1
            continue

        stats["questions_processed"] += 1

        results = ex.get("search_results") or []
        stats["total_search_results"] += len(results)

        pages_this_q = 0

        for _, page in enumerate(results):
            if pages_this_q >= MAX_PAGES_PER_Q:
                stats["skipped_max_pages"] += 1
                continue

            page = page or {}
            html = page.get("page_result") or ""

            if not html:
                stats["skipped_no_html"] += 1
                continue

            stats["html_lengths"].append(len(html))

            text = html2text(html)
            stats["text_lengths"].append(len(text))

            if len(text) < MIN_CHARS:
                stats["skipped_too_short"] += 1
                continue

            # This would be written
            stats["docs_written"] += 1
            pages_this_q += 1

        docs_per_question.append(pages_this_q)

print("\n📊 PROCESSING STATISTICS:")
print(f"  Questions in dataset:      {stats['total_questions']}")
print(f"  Questions processed:       {stats['questions_processed']}")
print(f"  Questions skipped (dynamic): {stats['questions_skipped_dynamic']}")
print(f"  Total search results:      {stats['total_search_results']}")
print(
    f"  Avg results per question:  {stats['total_search_results'] / stats['questions_processed']:.1f}"
)

print("\n📝 DOCUMENT CREATION:")
print(f"  Documents that WOULD be written: {stats['docs_written']}")
print(
    f"  Avg docs per question:           {stats['docs_written'] / stats['questions_processed']:.1f}"
)
print(f"  Max pages per question setting:  {MAX_PAGES_PER_Q}")

print("\n❌ DOCUMENTS FILTERED OUT:")
print(
    f"  No HTML:                   {stats['skipped_no_html']:5d} ({stats['skipped_no_html']/stats['total_search_results']*100:.1f}%)"
)
print(
    f"  Too short (<{MIN_CHARS} chars):  {stats['skipped_too_short']:5d} ({stats['skipped_too_short']/stats['total_search_results']*100:.1f}%)"
)
print(
    f"  Max pages limit reached:   {stats['skipped_max_pages']:5d} ({stats['skipped_max_pages']/stats['total_search_results']*100:.1f}%)"
)

print("\n📏 HTML vs TEXT SIZE:")
if stats["html_lengths"]:
    avg_html = sum(stats["html_lengths"]) / len(stats["html_lengths"])
    avg_text = sum(stats["text_lengths"]) / len(stats["text_lengths"])
    print(f"  Average HTML size:  {avg_html:,.0f} chars")
    print(f"  Average TEXT size:  {avg_text:,.0f} chars")
    print(f"  Conversion ratio:   {avg_text/avg_html*100:.1f}% (text/html)")

# Distribution of docs per question
print("\n📈 DOCS PER QUESTION DISTRIBUTION:")
dist = Counter(docs_per_question)
for count in sorted(dist.keys()):
    num_questions = dist[count]
    print(
        f"  {count:2d} docs: {num_questions:3d} questions ({num_questions/stats['questions_processed']*100:.1f}%)"
    )

# Find questions with 0 docs
zero_doc_questions = sum(1 for x in docs_per_question if x == 0)
print(f"\n⚠️  Questions with ZERO documents: {zero_doc_questions}")

# Sample analysis - show WHY documents are filtered
print("\n🔍 SAMPLE FILTERING ANALYSIS (first 3 questions):")
with bz2.open("data/crag_task_1_and_2_dev_v4.jsonl.bz2", "rt", encoding="utf-8") as fin:
    for idx, line in enumerate(fin):
        if idx >= 3:
            break

        ex = json.loads(line)
        q = ex.get("query", "")[:80]
        print(f"\n  Q{idx+1}: {q}...")

        results = ex.get("search_results") or []
        print(f"    Total results: {len(results)}")

        for rank, page in enumerate(results[:5]):  # First 5 results
            html = page.get("page_result") or ""
            text = html2text(html) if html else ""

            status = "✓ KEPT" if html and len(text) >= MIN_CHARS else "✗ FILTERED"
            reason = ""
            if not html:
                reason = "no HTML"
            elif len(text) < MIN_CHARS:
                reason = f"too short ({len(text)} < {MIN_CHARS})"

            print(
                f"      [{rank}] HTML: {len(html):7,d} → TEXT: {len(text):6,d} | {status} {reason}"
            )

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if stats["skipped_no_html"] > stats["total_search_results"] * 0.5:
    print("❌ MAJOR ISSUE: >50% of results have no HTML")
    print("   → The dataset file may be incomplete or different from expected")
elif stats["skipped_too_short"] > stats["total_search_results"] * 0.5:
    print("❌ MAJOR ISSUE: >50% filtered due to min-chars threshold")
    print(f"   → Current threshold: {MIN_CHARS} chars")
    print("   → Recommendation: Lower to 100 or 50 chars")
    print("   → OR: Improve HTML→text conversion to preserve more content")
elif stats["docs_written"] < 5000:
    print(f"⚠️  WARNING: Only {stats['docs_written']} docs created (expected ~10,000+)")
    print(f"   → Main issue: {stats['skipped_too_short']:,d} filtered as too short")
    print(
        f"   → HTML→text conversion is losing {100 - avg_text/avg_html*100:.0f}% of content"
    )
    print(
        "   → Navigation, menus, and boilerplate are counted as content but filtered out"
    )
else:
    print(f"✅ GOOD: {stats['docs_written']} documents would be created")

print("\n💡 RECOMMENDATIONS:")
if stats["skipped_too_short"] > 1000:
    print(f"  1. Lower MIN_CHARS from {MIN_CHARS} to 100 or 50")
    print("  2. Improve HTML cleaning to extract main content only")
    print("  3. Use --fallback-snippet for pages that fail HTML conversion")
