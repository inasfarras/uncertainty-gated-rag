import argparse
import bz2
import io
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def open_text_maybe_bz2(path: Path):
    """Opens a file, transparently handling .bz2 compression."""
    if str(path).endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8")


def html2text(html: str) -> str:
    """Converts HTML to cleaned plain text."""
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    txt = soup.get_text("\n")
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


def main():
    """Main function to prepare the CRAG dataset from JSONL."""
    ap = argparse.ArgumentParser(
        description="Prepare CRAG full HTML → TXT + questions/meta"
    )
    ap.add_argument("--src", default="data/crag_task_1_and_2_dev_v4.jsonl.bz2")
    ap.add_argument("--out-dir", default="data/crag_corpus_html")
    ap.add_argument("--qs-file", default="data/crag_questions.jsonl")
    ap.add_argument("--meta-file", default="data/crag_meta.jsonl")
    ap.add_argument("--n", type=int)
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-pages-per-q", type=int, default=20)
    ap.add_argument("--fallback-snippet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.qs_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_file).parent.mkdir(parents=True, exist_ok=True)

    docs_written = 0
    questions_exported = 0
    skipped_short = 0
    skipped_dynamic = 0

    with open_text_maybe_bz2(Path(args.src)) as fin, open(
        args.qs_file, "w", encoding="utf-8"
    ) as qf, open(args.meta_file, "w", encoding="utf-8") as mf:
        for idx, line in enumerate(fin):
            if args.n and questions_exported >= args.n:
                break
            ex = json.loads(line)
            qid = str(ex.get("interaction_id") or ex.get("id") or f"ex_{idx}")
            q = ex.get("query") or ex.get("question") or ""
            tag = (ex.get("static_or_dynamic") or "").lower()
            if args.static_only and tag in {"real-time", "fast-changing"}:
                skipped_dynamic += 1
                continue
            qf.write(
                json.dumps(
                    {"id": qid, "question": q, "gold": ex.get("answer")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            questions_exported += 1

            results = ex.get("search_results") or []
            pages = 0
            for rank, page in enumerate(results):
                if pages >= args.max_pages_per_q:
                    break
                page = page or {}
                html = page.get("page_result") or ""
                if not html and args.fallback_snippet:
                    html = page.get("page_snippet") or ""
                if not html:
                    continue
                text = html2text(html)
                if len(text) < args.min_chars:
                    skipped_short += 1
                    continue
                doc_id = f"{qid}_{rank}"
                (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
                mf.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "qid": qid,
                            "url": page.get("page_url"),
                            "title": page.get("page_name"),
                            "rank": rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                docs_written += 1
                pages += 1

    print("\n--- Summary ---")
    print("Documents written:       ", docs_written)
    print("Questions exported:      ", questions_exported)
    print("Pages skipped (too short):", skipped_short)
    print("Questions skipped (dyn): ", skipped_dynamic)
    print("Corpus dir:              ", args.out_dir)
    print("Questions JSONL:         ", args.qs_file)
    print("Meta JSONL:              ", args.meta_file)


if __name__ == "__main__":
    main()
