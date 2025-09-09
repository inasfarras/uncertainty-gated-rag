# scripts/prepare_crag.py
# Usage (public, no login):   python scripts/prepare_crag.py --split train --static-only --n 200
# Usage (with token):         set HF_TOKEN=hf_xxx  # PowerShell: $env:HF_TOKEN="hf_xxx"
#                             python scripts/prepare_crag.py --split train --static-only --n 200
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm


def html_to_text(html: str, prefer: str = "lxml") -> str:
    """Convert HTML to plain text, removing script/style and collapsing blanks."""
    if not html:
        return ""
    parser = prefer if prefer in {"lxml", "html.parser"} else "html.parser"
    try:
        soup = BeautifulSoup(html, parser)
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare CRAG dataset (HTML -> TXT + questions JSONL)."
    )
    p.add_argument(
        "--out-dir", default="data/crag_corpus", help="Directory for .txt pages."
    )
    p.add_argument(
        "--qs-file",
        default="data/crag_questions.jsonl",
        help="Output JSONL with questions.",
    )
    p.add_argument(
        "--meta-file",
        default="data/crag_meta.jsonl",
        help="Output JSONL mapping doc->meta (qid,url,title,rank).",
    )
    p.add_argument(
        "--config-name",
        default="crag_task_1_and_2",
        help="HF config (keep default if unsure).",
    )
    p.add_argument(
        "--split", default="train", help="Split to use (train/validation/test)."
    )
    p.add_argument("--n", type=int, help="Cap number of questions exported.")
    p.add_argument(
        "--static-only",
        action="store_true",
        help="Skip real-time/fast-changing questions.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Skip pages shorter than this text length.",
    )
    p.add_argument(
        "--max-pages-per-q", type=int, default=10, help="Cap pages saved per question."
    )
    p.add_argument(
        "--dedupe-by-url",
        action="store_true",
        help="Skip duplicate page URLs within a question.",
    )
    p.add_argument(
        "--seed", type=int, help="Optional seed to shuffle before taking --n."
    )
    p.add_argument(
        "--min-chars-snippet",
        type=int,
        default=200,
        help="Min length when using snippet.",
    )
    p.add_argument(
        "--fallback-question-doc",
        action="store_true",
        help="If a question yields no pages, write a tiny doc from question(+gold).",
    )
    # Live fetch arguments
    p.add_argument(
        "--fetch-live",
        action="store_true",
        help="If page_result is empty, fetch from page_url.",
    )
    p.add_argument(
        "--user-agent", default="agentic-rag/1.0", help="User-Agent for live fetching."
    )
    p.add_argument(
        "--timeout", type=int, default=10, help="HTTP request timeout in seconds."
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between live requests.",
    )
    p.add_argument(
        "--max-live-per-q",
        type=int,
        default=3,
        help="Max live pages to fetch per question.",
    )
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.qs_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_file).parent.mkdir(parents=True, exist_ok=True)

    # Use token from env (avoids Windows Git Credential Manager pop-up)
    hf_token = os.getenv(
        "HF_TOKEN"
    )  # set this only if needed; public datasets work without it
    print(
        f"Loading dataset: Quivr/CRAG (config={args.config_name}, split={args.split})"
    )
    ds = load_dataset(
        "Quivr/CRAG", name=args.config_name, split=args.split, token=hf_token
    )

    # Optional shuffle for a more representative small subset
    if args.seed is not None:
        ds = ds.shuffle(seed=args.seed)

    docs_written = 0
    questions_exported = 0
    skipped_short = 0
    skipped_dynamic = 0
    live_fetch_ok = 0
    live_fetch_blocked = 0
    live_fetch_fail = 0
    docs_from = {"html": 0, "snippet": 0, "live": 0, "fallback": 0}
    robot_parsers: Dict[str, RobotFileParser] = {}

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    meta_out = open(args.meta_file, "w", encoding="utf-8")
    qs_out = open(args.qs_file, "w", encoding="utf-8")

    try:
        for ex_idx, ex in enumerate(tqdm(ds, desc="Processing")):
            if args.n and questions_exported >= args.n:
                print(f"\nReached cap of {args.n} questions.")
                break

            qid = str(
                ex.get("interaction_id")
                or ex.get("id")
                or ex.get("qid")
                or f"ex_{ex_idx}"
            )
            question = ex.get("query") or ex.get("question") or ""
            gold = ex.get("answer")
            tag = (ex.get("static_or_dynamic") or "").lower()

            if args.static_only and tag in {"real-time", "fast-changing"}:
                skipped_dynamic += 1
                continue

            results = ex.get("search_results") or []
            if not isinstance(results, list):
                results = []

            # Within-question URL dedupe
            seen_urls = set()
            pages_saved_this_q = 0
            live_fetches_this_q = 0

            for rank, page in enumerate(results):
                if pages_saved_this_q >= args.max_pages_per_q:
                    break
                page = page or {}
                url = (page.get("page_url") or "").strip()
                title = (page.get("page_name") or "").strip()
                html = page.get("page_result") or ""
                snippet = page.get("page_snippet") or ""

                if (
                    not html
                    and args.fetch_live
                    and url
                    and live_fetches_this_q < args.max_live_per_q
                ):
                    live_fetches_this_q += 1
                    content_source = None
                    try:
                        parsed_url = urlparse(url)
                        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                        robots_url = f"{domain}/robots.txt"
                        if domain not in robot_parsers:
                            rp = RobotFileParser()
                            rp.set_url(robots_url)
                            try:
                                rp.read()
                            except Exception:
                                rp = RobotFileParser()
                                rp.parse("")
                            robot_parsers[domain] = rp

                        if robot_parsers[domain].can_fetch(args.user_agent, url):
                            time.sleep(args.sleep)
                            resp = session.get(url, timeout=args.timeout)
                            resp.raise_for_status()
                            if "text/html" in (resp.headers.get("Content-Type") or ""):
                                html = resp.text
                                live_fetch_ok += 1
                                content_source = "live"
                            else:
                                live_fetch_fail += 1
                        else:
                            live_fetch_blocked += 1
                    except Exception:
                        live_fetch_fail += 1

                content_source_val = None
                if html:
                    raw = html
                    content_source_val = content_source or "html"
                else:
                    raw = snippet
                    content_source_val = content_source or (
                        "snippet" if snippet else None
                    )

                if args.dedupe_by_url and url:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                text = html_to_text(raw, prefer="lxml")
                min_chars = (
                    args.min_chars
                    if content_source_val in {"html", "live"}
                    else args.min_chars_snippet
                )
                if len(text) < min_chars:
                    skipped_short += 1
                    continue

                doc_id = f"{qid}_{rank}"
                (Path(args.out_dir) / f"{doc_id}.txt").write_text(
                    text, encoding="utf-8"
                )
                meta_out.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "qid": qid,
                            "url": url,
                            "title": title,
                            "rank": rank,
                            "source": content_source_val,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if content_source_val:
                    docs_from[content_source_val] += 1
                docs_written += 1
                pages_saved_this_q += 1

            if pages_saved_this_q == 0 and args.fallback_question_doc:
                tiny = question or ""
                if gold:
                    tiny += f"\n\nGOLD: {gold}"
                if tiny.strip():
                    doc_id = f"{qid}_q"
                    (Path(args.out_dir) / f"{doc_id}.txt").write_text(
                        tiny, encoding="utf-8"
                    )
                    meta_out.write(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "qid": qid,
                                "url": "",
                                "title": "question_fallback",
                                "rank": -1,
                                "source": "fallback",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    docs_from["fallback"] += 1
                    docs_written += 1
                    pages_saved_this_q += 1

            qs_out.write(
                json.dumps(
                    {"id": qid, "question": question, "gold": gold}, ensure_ascii=False
                )
                + "\n"
            )
            questions_exported += 1
    finally:
        meta_out.close()
        qs_out.close()

    print("\n--- Summary ---")
    print(
        f"Documents written:        {docs_written} (html: {docs_from['html']}, snippet: {docs_from['snippet']}, live: {docs_from['live']}, fallback: {docs_from['fallback']})"
    )
    print(f"Questions exported:       {questions_exported}")
    print(f"Pages skipped (too short):{skipped_short}")
    print(f"Questions skipped (dyn):  {skipped_dynamic}")
    if args.fetch_live:
        print(f"Live fetches OK:          {live_fetch_ok}")
        print(f"Live fetches blocked:     {live_fetch_blocked}")
        print(f"Live fetches failed:      {live_fetch_fail}")
    print(f"Corpus dir:               {args.out_dir}")
    print(f"Questions JSONL:          {args.qs_file}")
    print(f"Meta JSONL:               {args.meta_file}")


if __name__ == "__main__":
    main()
