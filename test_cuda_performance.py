"""Test CUDA performance for embeddings and reranking."""

import time

import torch
from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.rerank.bge import create_reranker

print("=" * 60)
print("CUDA Performance Test")
print("=" * 60)
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Test embedding performance
print("Testing Embedding Encoder...")
test_texts = ["This is a test document about machine learning."] * 20
start = time.time()
embeds = embed_texts(test_texts)
duration = time.time() - start
print(
    f"✓ Embedded {len(test_texts)} texts in {duration:.3f}s ({len(test_texts)/duration:.1f} texts/sec)"
)
print(f"  Embedding shape: {embeds.shape}")
print()

# Test reranking performance
print("Testing Reranker...")
if settings.USE_RERANK:
    reranker = create_reranker()
    if reranker:
        query = "What is machine learning?"
        candidates = [
            {
                "chunk_id": f"doc_{i}",
                "text": f"Document {i} about ML and AI technology.",
            }
            for i in range(50)
        ]
        start = time.time()
        results = reranker.rerank(query, candidates, top_k=10)
        duration = time.time() - start
        print(
            f"✓ Reranked {len(candidates)} candidates in {duration:.3f}s ({len(candidates)/duration:.1f} docs/sec)"
        )
        print(f"  Top score: {results[0].get('rerank_score', 0):.4f}")
    else:
        print("⚠️  Reranker not available")
else:
    print("⚠️  Reranking disabled in config")

print()
print("=" * 60)
print("Configuration:")
print(f"  EMBED_BACKEND: {settings.EMBED_BACKEND}")
print(f"  EMBED_BATCH_SIZE: {getattr(settings, 'EMBED_BATCH_SIZE', 32)}")
print(f"  USE_RERANK: {settings.USE_RERANK}")
print(f"  RERANK_FP16: {settings.RERANK_FP16}")
print(f"  RERANK_BATCH_SIZE: {getattr(settings, 'RERANK_BATCH_SIZE', 64)}")
print("=" * 60)
