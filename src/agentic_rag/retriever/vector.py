import os
import re
from typing import Dict, List, Tuple, TypedDict

import pandas as pd
import tiktoken

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.prompting import ContextBlock, pack_context
from agentic_rag.rerank.bge import create_reranker
from agentic_rag.retriever.hyde import hyde_query, should_use_hyde
from agentic_rag.retriever.mmr import mmr_select
from agentic_rag.store.faiss_store import load_index


class ContextChunk(TypedDict):
    id: str
    text: str
    score: float


class VectorRetriever:
    def __init__(self, faiss_dir: str = "artifacts/faiss"):
        self.store = load_index(faiss_dir)
        self.chunks = pd.read_parquet(
            os.path.join(faiss_dir, "chunks.parquet")
        ).set_index("id")
        self.reranker = None
        if settings.USE_RERANK:
            self.reranker = create_reranker(settings.RERANKER_MODEL)

    def _encoding(self):
        try:
            return tiktoken.encoding_for_model(settings.LLM_MODEL)
        except Exception:
            return tiktoken.get_encoding("o200k_base")

    def _token_count(self, text: str) -> int:
        try:
            enc = self._encoding()
            return len(enc.encode(text or ""))
        except Exception:
            return max(1, len(text or "") // 4)

    def retrieve_pack(
        self,
        query: str,
        k: int,
        *,
        probe_factor: int = 4,
        exclude_doc_ids: set[str] | None = None,
        round_idx: int = 0,
        llm_client=None,
    ) -> Tuple[List[dict], Dict[str, object]]:
        """Enhanced retrieve with MMR, reranking, and HyDE support.

        Returns:
            contexts: list of {id, text}
            stats: {context_tokens, n_ctx_blocks, retrieved_ids}
        """
        # Step 1: HyDE query rewriting (only for RETRIEVE_MORE rounds)
        search_query = query
        if should_use_hyde(round_idx, settings.USE_HYDE) and llm_client:
            print("   🔮 Generating HyDE query...")
            hyde_result = hyde_query(query, llm_client)
            if hyde_result:
                search_query = hyde_result
                print(f"   🔮 HyDE query: {hyde_result[:100]}...")
            else:
                print("   🔮 HyDE generation failed, using original query")

        # Step 2: Deep retrieval with larger pool
        print("   🔍 Embedding query and searching...")
        qvec = embed_texts([search_query])[0]
        pool_k = (
            settings.RETRIEVAL_POOL_K
            if (settings.USE_RERANK or settings.MMR_LAMBDA > 0)
            else max(k, probe_factor * k)
        )
        print(f"   🔍 Searching for {pool_k} candidates...")
        hits = self.store.search(qvec, pool_k)

        # Step 3: Map chunk hits to best chunk per doc_id and prepare candidates
        best_by_doc: Dict[str, Tuple[str, float]] = {}
        for chunk_id, score in hits:
            raw_doc_id = chunk_id.split("__")[0]
            doc_id = re.sub(r"[^A-Za-z0-9_\-]", "_", raw_doc_id)
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue
            if (doc_id not in best_by_doc) or (score > best_by_doc[doc_id][1]):
                best_by_doc[doc_id] = (chunk_id, score)

        # Prepare candidates for reranking/MMR
        candidates = []
        for doc_id, (chunk_id, faiss_score) in best_by_doc.items():
            text = self.chunks.loc[chunk_id, "text"]
            # Get embedding for MMR (reuse query embedding approach)
            emb = embed_texts([text])[0]
            candidates.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "faiss_score": faiss_score,
                    "emb": emb,
                    "score": faiss_score,  # Default score
                }
            )

        # Step 4: Optional reranking
        if self.reranker and candidates:
            try:
                rerank_k = min(len(candidates), k * 2)  # Keep 2x target for MMR
                print(f"   🏆 Reranking {len(candidates)} → {rerank_k} candidates...")
                candidates = self.reranker.rerank(query, candidates, rerank_k)
                # Update scores to rerank scores
                for c in candidates:
                    c["score"] = c.get("rerank_score", c["faiss_score"])
                print("   🏆 Reranking completed")
            except Exception as e:
                print(f"   ❌ Reranking failed, falling back to FAISS scores: {e}")

        # Step 5: MMR diversification
        if settings.MMR_LAMBDA > 0 and len(candidates) > k:
            try:
                print(
                    f"   🎯 Applying MMR diversification (λ={settings.MMR_LAMBDA}) {len(candidates)} → {k}..."
                )
                candidates = mmr_select(qvec, candidates, k, settings.MMR_LAMBDA)
                print("   🎯 MMR selection completed")
            except Exception as e:
                print(f"   ❌ MMR selection failed, using top-k: {e}")
                candidates = candidates[:k]
        else:
            # Just take top-k by score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:k]
            print(f"   📋 Selected top-{k} by score")

        # Step 6: Prepare blocks for packing
        blocks: List[ContextBlock] = []
        retrieved_ids = []
        for candidate in candidates:
            blocks.append(
                {
                    "id": candidate["doc_id"],
                    "text": candidate["text"],
                    "score": float(candidate["score"]),
                }
            )
            retrieved_ids.append(candidate["doc_id"])

        # Step 7: Pack under token cap
        cap = settings.MAX_CONTEXT_TOKENS or settings.CONTEXT_TOKEN_CAP
        packed, total_tokens, n_blocks = pack_context(blocks, max_tokens_cap=cap)
        contexts: List[dict] = [{"id": c["id"], "text": c["text"]} for c in packed]

        stats = {
            "context_tokens": total_tokens,
            "n_ctx_blocks": n_blocks,
            "retrieved_ids": retrieved_ids,
            "used_hyde": should_use_hyde(round_idx, settings.USE_HYDE),
            "used_rerank": self.reranker is not None,
            "used_mmr": settings.MMR_LAMBDA > 0,
        }
        return contexts, stats

    def retrieve(self, query: str, k: int) -> List[ContextChunk]:
        """Backwards-compatible retrieve that returns raw top-k chunks (no packing)."""
        qvec = embed_texts([query])[0]
        hits = self.store.search(qvec, k)
        out: List[ContextChunk] = []
        for chunk_id, score in hits:
            text = self.chunks.loc[chunk_id, "text"]
            out.append({"id": chunk_id, "text": text, "score": score})
        return out
