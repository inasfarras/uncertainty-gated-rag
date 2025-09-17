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


class Ingestor:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def load_directory(self) -> list[dict]:
        out = []
        glob_path = os.path.join(self.input_dir, "**/*.txt")
        for path in glob.glob(glob_path, recursive=True):
            with open(path, encoding="utf-8", errors="ignore") as f:
                stem = os.path.splitext(os.path.basename(path))[0]
                doc_id = _sanitize_doc_id(stem)
                out.append({"doc_id": doc_id, "text": f.read()})
        return out

    def ingest(self, output_dir: str):
        docs = self.load_directory()
        records = []
        for d in docs:
            chunks = chunk_text(d["text"])
            for i, ch in enumerate(chunks):
                records.append({"id": f"{d['doc_id']}__{i}", "text": ch})
        if not records:
            print("[red]No .txt files found.[/red]")
            return

        df = pd.DataFrame(records)
        batch_size = 200
        all_embs = []

        print(f"Embedding {len(df)} chunks in batches of {batch_size}...")
        for i in tqdm(range(0, len(df), batch_size)):
            batch_texts = df["text"].iloc[i : i + batch_size].tolist()
            batch_embs = embed_texts(batch_texts)
            all_embs.append(batch_embs)

        embs = np.concatenate(all_embs, axis=0)
        build_index(embs, df["id"].tolist(), output_dir)
        df.to_parquet(os.path.join(output_dir, "chunks.parquet"))
        print(f"[green]Built FAISS at {output_dir} with {len(df)} chunks[/green]")


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
    ingestor = Ingestor(input)
    ingestor.ingest(output_dir=out)


if __name__ == "__main__":
    app()
