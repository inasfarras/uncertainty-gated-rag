import os
import re
from typing import Any, Dict, List, Tuple, TypedDict

import pandas as pd
import tiktoken

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.prompting import ContextBlock, pack_context
from agentic_rag.rerank.bge import create_reranker
from agentic_rag.retriever.bm25 import BM25Retriever
from agentic_rag.retriever.hyde import hyde_query, should_use_hyde
from agentic_rag.retriever.mmr import mmr_select
from agentic_rag.store.faiss_store import load_index


class ContextChunk(TypedDict):
    id: str
    text: str
    score: float


class Candidate(TypedDict):
    doc_id: str
    chunk_id: str
    text: str
    faiss_score: float
    emb: object
    score: float
    rerank_score: float


class VectorRetriever:
    def __init__(self, faiss_dir: str = "artifacts/faiss"):
        self.store = load_index(faiss_dir)
        self.chunks = pd.read_parquet(
            os.path.join(faiss_dir, "chunks.parquet")
        ).set_index("id")
        self.reranker = None
        if settings.USE_RERANK:
            self.reranker = create_reranker(settings.RERANKER_MODEL)

        # Initialize BM25 for hybrid search if enabled
        self.bm25_retriever: BM25Retriever | None = None
        self.use_hybrid = getattr(settings, "USE_HYBRID_SEARCH", False)
        if self.use_hybrid:
            self._init_bm25_retriever(faiss_dir)

    def _init_bm25_retriever(self, faiss_dir: str) -> None:
        """Initialize BM25 retriever for hybrid search."""
        try:
            from agentic_rag.retriever.bm25 import create_bm25_index

            # Try to load existing BM25 index
            bm25_index_path = os.path.join(faiss_dir, "bm25_index.pkl")
            if os.path.exists(bm25_index_path):
                from agentic_rag.retriever.bm25 import load_bm25_index

                self.bm25_retriever = load_bm25_index(bm25_index_path)
                print("   📚 Loaded existing BM25 index for hybrid search")
            else:
                # Create new BM25 index
                print("   📚 Creating BM25 index for hybrid search...")
                self.bm25_retriever = create_bm25_index(faiss_dir, bm25_index_path)
                print("   📚 BM25 index created and saved")

        except Exception as e:
            print(f"   ❌ Failed to initialize BM25 retriever: {e}")
            self.use_hybrid = False
            self.bm25_retriever = None

    def _hybrid_search(
        self, query: str, k: int, probe_factor: int
    ) -> List[Tuple[str, float, str]]:
        """Perform hybrid search combining vector and BM25 results."""
        # Determine pool sizes
        pool_k = (
            settings.RETRIEVAL_POOL_K
            if (settings.USE_RERANK or settings.MMR_LAMBDA > 0)
            else max(k, probe_factor * k)
        )

        # Get vector results
        qvec = embed_texts([query])[0]
        vector_hits = self.store.search(qvec, pool_k)

        # Get BM25 results (search on chunk level, then map to doc level)
        bm25_hits = []
        if self.bm25_retriever:
            bm25_hits = self.bm25_retriever.search(query, pool_k)

        # Combine and normalize scores
        alpha = getattr(settings, "HYBRID_ALPHA", 0.7)
        combined_results = self._combine_retrieval_results(
            vector_hits, bm25_hits, alpha
        )

        print(
            f"   🔍 Hybrid search: {len(vector_hits)} vector + {len(bm25_hits)} BM25 → {len(combined_results)} combined"
        )

        return combined_results[:pool_k]

    def _combine_retrieval_results(
        self,
        vector_hits: List[Tuple[str, float]],
        bm25_hits: List[Tuple[str, float]],
        alpha: float = 0.7,
    ) -> List[Tuple[str, float, str]]:
        """Combine vector and BM25 results with score normalization."""
        # Normalize vector scores
        if vector_hits:
            vector_scores = [score for _, score in vector_hits]
            min_v, max_v = min(vector_scores), max(vector_scores)
            if max_v > min_v:
                vector_normalized = [
                    (
                        chunk_id,
                        (score - min_v) / (max_v - min_v),
                        self.chunks.loc[chunk_id, "text"],
                    )
                    for chunk_id, score in vector_hits
                ]
            else:
                vector_normalized = [
                    (chunk_id, 1.0, self.chunks.loc[chunk_id, "text"])
                    for chunk_id, _ in vector_hits
                ]
        else:
            vector_normalized = []

        # Normalize BM25 scores
        if bm25_hits:
            bm25_scores = [score for _, score in bm25_hits]
            min_b, max_b = min(bm25_scores), max(bm25_scores)
            if max_b > min_b:
                bm25_normalized = [
                    (
                        doc_id,
                        (score - min_b) / (max_b - min_b),
                        (
                            self.chunks[
                                self.chunks.index.str.startswith(doc_id + "__")
                            ]["text"].iloc[0]
                            if not self.chunks[
                                self.chunks.index.str.startswith(doc_id + "__")
                            ].empty
                            else f"Content for {doc_id} not available."
                        ),
                    )
                    for doc_id, score in bm25_hits
                ]
            else:
                bm25_normalized = [
                    (
                        doc_id,
                        1.0,
                        (
                            self.chunks[
                                self.chunks.index.str.startswith(doc_id + "__")
                            ]["text"].iloc[0]
                            if not self.chunks[
                                self.chunks.index.str.startswith(doc_id + "__")
                            ].empty
                            else f"Content for {doc_id} not available."
                        ),
                    )
                    for doc_id, _ in bm25_hits
                ]
        else:
            bm25_normalized = []

        # Combine scores
        combined_scores: Dict[str, Dict[str, Any]] = {}

        # Add vector results
        for chunk_id, norm_score, text in vector_normalized:
            combined_scores[chunk_id] = {"score": alpha * norm_score, "text": text}

        # Add BM25 results (need to map doc_id back to chunk_id)
        for doc_id, norm_score, text in bm25_normalized:
            found_existing = False
            for chunk_id, existing_data in combined_scores.items():
                if chunk_id.startswith(doc_id + "__"):
                    existing_data["score"] += (1 - alpha) * norm_score
                    found_existing = True
                    break

            if not found_existing:
                synthetic_chunk_id = f"{doc_id}__bm25_synthetic"
                combined_scores[synthetic_chunk_id] = {
                    "score": (1 - alpha) * norm_score,
                    "text": text,  # Use the text directly from bm25_normalized
                }

        final_results = []
        for chunk_id, data in combined_scores.items():
            final_results.append((chunk_id, data["score"], data["text"]))

        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results

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

        # Step 2: Retrieval with hybrid search if enabled
        qvec = embed_texts([search_query])[0]  # Always embed query once
        if self.use_hybrid and self.bm25_retriever:
            print("   🔍 Performing hybrid search (Vector + BM25)...")
            # For hybrid search, hits already contain the text, so pass that along
            raw_hits = self._hybrid_search(search_query, k, probe_factor)
            # Convert raw_hits to (chunk_id, score) for compatibility with downstream logic
            hits: List[Tuple[str, float, str]] = raw_hits
        else:
            print("   🔍 Embedding query and searching...")
            pool_k = (
                settings.RETRIEVAL_POOL_K
                if (settings.USE_RERANK or settings.MMR_LAMBDA > 0)
                else max(k, probe_factor * k)
            )
            print(f"   🔍 Searching for {pool_k} candidates...")
            raw_hits_list = self.store.search(qvec, pool_k)
            hits = [
                (chunk_id, score, self.chunks.loc[chunk_id, "text"])
                for chunk_id, score in raw_hits_list
            ]

        # Step 3: Map chunk hits to best chunk per doc_id and prepare candidates
        best_by_doc: Dict[str, Tuple[str, float, str]] = {}
        for chunk_id, score, text in hits:
            raw_doc_id = chunk_id.split("__")[0]
            doc_id = re.sub(r"[^A-Za-z0-9_\-]", "_", raw_doc_id)
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue

            current_score = score
            current_text = text

            if (doc_id not in best_by_doc) or (current_score > best_by_doc[doc_id][1]):
                best_by_doc[doc_id] = (chunk_id, current_score, current_text)

        # Prepare candidates for reranking/MMR
        candidates: list[Candidate] = []
        if best_by_doc:
            chunk_items = list(best_by_doc.items())
            # Directly use the text that's already stored in best_by_doc
            texts = [item[1][2] for item in chunk_items]

            # BATCH EMBEDDING FIX: Embed all texts in one go
            all_texts_to_embed = texts
            if self.reranker:
                all_texts_to_embed.append(query)

            embeddings = embed_texts(all_texts_to_embed)
            text_embeddings = embeddings[: len(texts)]

            for (doc_id, (chunk_id, faiss_score, text)), emb in zip(
                chunk_items, text_embeddings
            ):
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": text,
                        "faiss_score": faiss_score,
                        "emb": emb,
                        "score": faiss_score,  # Default score
                        "rerank_score": 0.0,
                    }
                )

        # Step 4: Optional reranking
        if self.reranker and candidates:
            try:
                rerank_k = min(len(candidates), k * 2)  # Keep 2x target for MMR
                print(f"   🏆 Reranking {len(candidates)} → {rerank_k} candidates...")
                reranked_results = self.reranker.rerank(
                    query, [dict(c) for c in candidates], rerank_k
                )

                # Create a score map for efficient lookup
                score_map = {
                    item["chunk_id"]: item.get("rerank_score", 0.0)
                    for item in reranked_results
                }

                # Update candidates with scores and filter
                updated_candidates: list[Candidate] = []
                for c in candidates:
                    if c["chunk_id"] in score_map:
                        c["rerank_score"] = score_map[c["chunk_id"]]
                        c["score"] = c["rerank_score"]
                        updated_candidates.append(c)

                # Sort by new score and take top_k
                updated_candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = updated_candidates[:rerank_k]

                print("   🏆 Reranking completed")
            except Exception as e:
                print(f"   ❌ Reranking failed, falling back to FAISS scores: {e}")

        # Step 5: MMR diversification
        if settings.MMR_LAMBDA > 0 and len(candidates) > k:
            try:
                print(
                    f"   🎯 Applying MMR diversification (λ={settings.MMR_LAMBDA}) {len(candidates)} → {k}..."
                )
                selected_candidates = mmr_select(
                    qvec, [dict(c) for c in candidates], k, settings.MMR_LAMBDA
                )
                # Filter candidates to keep only those selected by MMR
                selected_ids = {c["chunk_id"] for c in selected_candidates}
                candidates = [c for c in candidates if c["chunk_id"] in selected_ids]
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
                    "score": candidate["score"],
                }
            )
            retrieved_ids.append(candidate["doc_id"])

        # Minimal MMR packer
        if settings.MMR_LAMBDA > 0:
            from agentic_rag.retriever.mmr import mmr_pack_context

            packed, total_tokens, n_blocks = mmr_pack_context(
                blocks,
                max_tokens_cap=(
                    settings.MAX_CONTEXT_TOKENS or settings.CONTEXT_TOKEN_CAP
                ),
                mmr_lambda=settings.MMR_LAMBDA,
            )
            contexts_list: list[dict] = packed
        else:
            # Step 7: Pack under token cap
            cap = settings.MAX_CONTEXT_TOKENS or settings.CONTEXT_TOKEN_CAP
            packed, total_tokens, n_blocks = pack_context(blocks, max_tokens_cap=cap)
            contexts_list = packed

        stats = {
            "context_tokens": total_tokens,
            "n_ctx_blocks": n_blocks,
            "retrieved_ids": retrieved_ids,
            "used_hyde": should_use_hyde(round_idx, settings.USE_HYDE),
            "used_rerank": self.reranker is not None,
            "used_mmr": settings.MMR_LAMBDA > 0,
            "used_hybrid": self.use_hybrid and self.bm25_retriever is not None,
        }
        return contexts_list, stats

    def retrieve(self, query: str, k: int) -> List[ContextChunk]:
        """Backwards-compatible retrieve that returns raw top-k chunks (no packing)."""
        qvec = embed_texts([query])[0]
        hits = self.store.search(qvec, k)
        out: List[ContextChunk] = []
        for chunk_id, score in hits:
            # Ensure that the text is always available in the ContextChunk
            if chunk_id in self.chunks.index:
                text = self.chunks.loc[chunk_id, "text"]
                out.append({"id": chunk_id, "text": text, "score": score})
            else:
                print(
                    f"Warning: Chunk ID {chunk_id} not found in index during retrieve."
                )
        return out
