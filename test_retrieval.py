"""Quick test to check if retrieval is working."""

from agentic_rag.config import settings
from agentic_rag.retriever.vector import VectorRetriever

print(f"Testing retrieval from: {settings.FAISS_INDEX_PATH}")
retriever = VectorRetriever(settings.FAISS_INDEX_PATH)

# Simple test query
test_query = "What is the capital of France?"
print(f"\nTest query: {test_query}")

try:
    contexts, stats = retriever.retrieve_pack(test_query, k=5)
    print(f"\n✓ Retrieved {len(contexts)} contexts")
    print(f"Stats: {stats}")
    if contexts:
        print("\nFirst context:")
        print(f"  ID: {contexts[0].get('id')}")
        print(f"  Text: {contexts[0].get('text', '')[:200]}...")
    else:
        print("\n❌ NO CONTEXTS RETURNED!")
        print("\nChecking index details:")
        print(f"  Store object: {type(retriever.store)}")
        print(f"  Chunks shape: {retriever.chunks.shape}")
        print(f"  BM25 retriever: {retriever.bm25_retriever}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
