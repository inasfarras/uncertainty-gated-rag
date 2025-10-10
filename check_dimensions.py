"""Check embedding dimensions."""

import os

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.store.faiss_store import load_index

print(f"Config EMBED_BACKEND: {settings.EMBED_BACKEND}")
print(f"Env EMBED_BACKEND: {os.getenv('EMBED_BACKEND', 'not set')}")

# Test actual embedding
test_emb = embed_texts(["test"])[0]
print(f"\nQuery embedding shape: {test_emb.shape}")

# Check FAISS index
index_path = settings.FAISS_INDEX_PATH
print(f"\nLoading index from: {index_path}")
store = load_index(index_path)
print(f"FAISS index dimension: {store.index.d}")
print(f"FAISS index size: {store.index.ntotal}")

if test_emb.shape[0] != store.index.d:
    print(f"\n❌ MISMATCH: query={test_emb.shape[0]} vs index={store.index.d}")
else:
    print("\n✓ Dimensions match!")
