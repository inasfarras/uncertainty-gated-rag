import os
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd


class FaissStore:
    def __init__(self, index: faiss.Index, meta: pd.DataFrame):
        self.index = index
        self.meta = meta.set_index("id")

    def search(self, qvec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        distances, indices = self.index.search(
            qvec.reshape(1, -1).astype(np.float32), k
        )
        out = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            out.append((self.meta.iloc[idx].name, float(score)))
        return out


def build_index(embeddings: np.ndarray, ids: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # embeddings should already be normalized
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    pd.DataFrame({"id": ids}).to_parquet(os.path.join(out_dir, "meta.parquet"))


def load_index(out_dir: str) -> FaissStore:
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    meta = pd.read_parquet(os.path.join(out_dir, "meta.parquet"))
    return FaissStore(index, meta)
