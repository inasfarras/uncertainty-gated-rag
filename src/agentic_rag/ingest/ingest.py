import glob
import os
import re
from typing import Literal, cast

import numpy as np
import pandas as pd
import typer
from rich import print
from tqdm import tqdm

from agentic_rag.config import settings
from agentic_rag.embed.encoder import chunk_text, embed_texts
from agentic_rag.store.faiss_store import build_index

app = typer.Typer()


def _sanitize_doc_id(file_stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", file_stem)


def load_directory(input_dir: str) -> list[dict]:
    out = []
    for path in glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True):
        with open(path, encoding="utf-8", errors="ignore") as f:
            stem = os.path.splitext(os.path.basename(path))[0]
            doc_id = _sanitize_doc_id(stem)
            out.append({"doc_id": doc_id, "text": f.read()})
    return out


@app.command()
def main(
    input: str = "data/corpus",
    out: str = "artifacts/faiss",
    backend: str = typer.Option(
        settings.EMBED_BACKEND, "--backend", help="Backend to use: 'openai' or 'mock'."
    ),
):
    """Ingests text files, chunks them, embeds them, and builds a FAISS index."""
    settings.EMBED_BACKEND = cast(Literal["openai", "st"], backend)
    docs = load_directory(input)
    records = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks):
            # Store chunk ids as {doc_id}__{i}; doc_id is sanitized [A-Za-z0-9_-]+
            records.append({"id": f"{d['doc_id']}__{i}", "text": ch})
    if not records:
        print("[red]No .txt files found.[/red]")
        raise SystemExit(1)
    df = pd.DataFrame(records)

    # --- Batch processing for embeddings ---
    batch_size = 200  # A reasonable batch size to avoid API limits
    all_embs = []

    print(f"Embedding {len(df)} chunks in batches of {batch_size}...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df["text"].iloc[i : i + batch_size].tolist()
        batch_embs = embed_texts(batch_texts)
        all_embs.append(batch_embs)

    embs = np.concatenate(all_embs, axis=0)
    # --- End batch processing ---

    build_index(embs, df["id"].tolist(), out)
    df.to_parquet(os.path.join(out, "chunks.parquet"))
    print(f"[green]Built FAISS at {out} with {len(df)} chunks[/green]")


if __name__ == "__main__":
    app()
