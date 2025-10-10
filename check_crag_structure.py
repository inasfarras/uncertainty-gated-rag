"""Check the CRAG dataset structure and HTML availability."""

import bz2
import json

# Read first few examples
with bz2.open("data/crag_task_1_and_2_dev_v4.jsonl.bz2", "rt", encoding="utf-8") as f:
    examples = [json.loads(line) for i, line in enumerate(f) if i < 5]

print("=" * 80)
print("CRAG DATASET STRUCTURE ANALYSIS")
print("=" * 80)

ex = examples[0]
print("\n📋 Available fields per example:")
for key in ex.keys():
    print(f"  - {key}")

print("\n🔍 Search Results Analysis:")
print(f"  Questions analyzed: {len(examples)}")

html_stats = {
    "has_html": 0,
    "has_snippet": 0,
    "has_both": 0,
    "has_neither": 0,
    "html_only": 0,
    "snippet_only": 0,
}

total_results = 0

for ex in examples:
    sr = ex.get("search_results", [])
    total_results += len(sr)

    for result in sr:
        has_html = bool(result.get("page_result"))
        has_snippet = bool(result.get("page_snippet"))

        if has_html and has_snippet:
            html_stats["has_both"] += 1
        elif has_html:
            html_stats["html_only"] += 1
        elif has_snippet:
            html_stats["snippet_only"] += 1
        else:
            html_stats["has_neither"] += 1

        if has_html:
            html_stats["has_html"] += 1
        if has_snippet:
            html_stats["has_snippet"] += 1

print(f"\n  Total search results examined: {total_results}")
print("\n  HTML (page_result) availability:")
print(
    f"    Has HTML:           {html_stats['has_html']:4d} ({html_stats['has_html']/total_results*100:.1f}%)"
)
print(
    f"    HTML only:          {html_stats['html_only']:4d} ({html_stats['html_only']/total_results*100:.1f}%)"
)
print(
    f"    No HTML:            {total_results - html_stats['has_html']:4d} ({(total_results-html_stats['has_html'])/total_results*100:.1f}%)"
)
print("\n  Snippet (page_snippet) availability:")
print(
    f"    Has snippet:        {html_stats['has_snippet']:4d} ({html_stats['has_snippet']/total_results*100:.1f}%)"
)
print(
    f"    Snippet only:       {html_stats['snippet_only']:4d} ({html_stats['snippet_only']/total_results*100:.1f}%)"
)
print(
    f"    No snippet:         {total_results - html_stats['has_snippet']:4d} ({(total_results-html_stats['has_snippet'])/total_results*100:.1f}%)"
)
print("\n  Combined:")
print(
    f"    Has both:           {html_stats['has_both']:4d} ({html_stats['has_both']/total_results*100:.1f}%)"
)
print(
    f"    Has neither:        {html_stats['has_neither']:4d} ({html_stats['has_neither']/total_results*100:.1f}%)"
)

# Sample sizes
print("\n📏 Sample HTML/Snippet Sizes:")
for i, ex in enumerate(examples[:2]):
    print(f"\n  Question {i+1}: {ex['query'][:80]}...")
    sr = ex.get("search_results", [])[:3]
    for j, result in enumerate(sr):
        html = result.get("page_result", "")
        snippet = result.get("page_snippet", "")
        url = result.get("page_url", "")
        print(f"    Result {j+1}:")
        print(f"      URL: {url[:60]}...")
        print(f"      HTML length:    {len(html):6d} chars {'✓' if html else '✗'}")
        print(
            f"      Snippet length: {len(snippet):6d} chars {'✓' if snippet else '✗'}"
        )

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

html_pct = html_stats["has_html"] / total_results * 100
if html_pct > 80:
    print(f"✅ Good: {html_pct:.0f}% of results have full HTML")
elif html_pct > 50:
    print(f"⚠️  Partial: {html_pct:.0f}% of results have full HTML")
else:
    print(f"❌ Poor: Only {html_pct:.0f}% of results have full HTML")
    print("   → This explains your sparse corpus (~980 docs instead of ~16,000)")
    print(f"   → {html_stats['snippet_only']} results have snippets only (short text)")
    print(f"   → {html_stats['has_neither']} results have NO content at all")
