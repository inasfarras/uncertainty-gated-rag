import os

import faiss
import pandas as pd
from agentic_rag.config import settings

# Check chunks
print("=" * 80)
print("CRAG EMBEDDINGS DIAGNOSTIC")
print("=" * 80)

# Check chunks.parquet
chunks_path = "artifacts/crag_faiss/chunks.parquet"
if os.path.exists(chunks_path):
    df = pd.read_parquet(chunks_path)
    print("\n1. CHUNKS INFO:")
    print(f"   Total chunks: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Sample chunk IDs: {df['id'].head(3).tolist()}")
    print(f"   Sample text (first 100 chars): {df['text'].iloc[0][:100]}...")
    print(f"   Avg text length: {df['text'].str.len().mean():.0f} chars")
else:
    print(f"❌ Chunks file not found: {chunks_path}")

# Check FAISS index
index_path = "artifacts/crag_faiss/index.faiss"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print("\n2. FAISS INDEX INFO:")
    print(f"   Index type: {type(index)}")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Vector dimension: {index.d}")
    print(f"   Is trained: {index.is_trained}")

    # Check if dimensions match expected
    expected_dim_openai = 1536  # text-embedding-3-small
    if index.d != expected_dim_openai:
        print("   ⚠️  WARNING: Dimension mismatch!")
        print(f"   Expected (OpenAI text-embedding-3-small): {expected_dim_openai}")
        print(f"   Actual: {index.d}")
else:
    print(f"❌ FAISS index not found: {index_path}")

# Check meta.parquet
meta_path = "artifacts/crag_faiss/meta.parquet"
if os.path.exists(meta_path):
    meta = pd.read_parquet(meta_path)
    print("\n3. META INFO:")
    print(f"   Total entries: {len(meta)}")
    print(f"   Columns: {meta.columns.tolist()}")
    if len(meta) > 0:
        print(f"   Sample entry: {meta.iloc[0].to_dict()}")
else:
    print(f"❌ Meta file not found: {meta_path}")

# Check current config
# from agentic_rag.config import settings

print("\n4. CURRENT CONFIG:")
print(f"   EMBED_BACKEND: {settings.EMBED_BACKEND}")
print(f"   EMBED_MODEL: {settings.EMBED_MODEL}")
print(f"   FAISS_INDEX_PATH: {settings.FAISS_INDEX_PATH}")

# Check BM25 index
bm25_path = "artifacts/crag_faiss/bm25_index.pkl"
if os.path.exists(bm25_path):
    import pickle

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print("\n5. BM25 INDEX INFO:")
    if isinstance(bm25, dict):
        print("   Type: dict (indexed structure)")
        print(f"   Keys: {list(bm25.keys())[:5]}")
        print(f"   Total documents: {len(bm25)}")
    else:
        print(f"   Corpus size: {bm25.corpus_size}")
        print(f"   Avg doc length: {bm25.avg_doc_length:.1f}")
        print(f"   Total documents: {len(bm25.doc_ids)}")
else:
    print("\n5. BM25 INDEX: Not found")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

# Diagnose issues
issues = []
if os.path.exists(chunks_path) and os.path.exists(index_path):
    df = pd.read_parquet(chunks_path)
    index = faiss.read_index(index_path)

    if len(df) != index.ntotal:
        issues.append(f"⚠️  Chunks count ({len(df)}) != Index vectors ({index.ntotal})")

    if index.d != 1536:
        issues.append(
            f"⚠️  Wrong embedding dimension: {index.d} (expected 1536 for text-embedding-3-small)"
        )

    if len(df) < 5000:
        issues.append(
            f"⚠️  Very few chunks ({len(df)}). CRAG typically has 100k+ chunks for good coverage"
        )

if not issues:
    print("✅ No obvious issues detected with embeddings structure")
else:
    for issue in issues:
        print(issue)

print("\nRECOMMENDATIONS:")
if len(df) < 10000:
    print("• CRAG dataset seems incomplete. You may need to re-download and re-index.")
    print("• Expected: ~100k chunks for full CRAG task")
    print(f"• Current: {len(df)} chunks")
