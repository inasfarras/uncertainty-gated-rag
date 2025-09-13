# scripts/crag_full_download.py
import argparse
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

RAW_URL = "https://raw.githubusercontent.com/facebookresearch/CRAG/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"


def is_valid_bz2(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(3)
        return head == b"BZh"
    except Exception:
        return False


def download(
    url: str, out: Path, force: bool = False, attempts: int = 3, timeout: int = 60
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        print(f"File exists, skip: {out}")
        return out

    tmp = out.with_suffix(out.suffix + ".part")
    for i in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"Downloading ({i}/{attempts}) → {out}")
            req = Request(url, headers={"User-Agent": "agentic-rag/1.0"})
            with urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)  # 1MB chunks

            tmp.rename(out)

            size_mb = out.stat().st_size / 1024 / 1024
            print(f"Done. Size: {size_mb:.1f} MB")

            # Basic sanity checks
            if out.stat().st_size < 100 * 1024:  # <100KB is suspicious for this file
                raise ValueError("Downloaded file is unexpectedly small.")
            if not is_valid_bz2(out):
                raise ValueError("Downloaded file is not a valid .bz2 (missing 'BZh').")

            return out
        except Exception as e:
            print("Error:", e)
            # clean up partial
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if i == attempts:
                raise
            time.sleep(2**i)  # exponential backoff

    return out


def main():
    ap = argparse.ArgumentParser(description="Download CRAG full HTML dataset (.bz2).")
    ap.add_argument("--url", default=RAW_URL, help="Source URL")
    ap.add_argument(
        "--out",
        default="data/crag_task_1_and_2_dev_v4.jsonl.bz2",
        help="Output file path",
    )
    ap.add_argument(
        "--force", action="store_true", help="Force re-download if file exists"
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = ap.parse_args()

    out_path = Path(args.out)
    download(args.url, out_path, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    main()
