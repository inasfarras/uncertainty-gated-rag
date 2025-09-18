#!/usr/bin/env python3
"""
Quick test script to validate the new agentic features.

This script tests the key components of the enhanced agentic RAG system:
- Judge module context assessment
- Query transformation
- Hybrid search (if BM25 index exists)
- Enhanced uncertainty gate with Judge integration

Usage:
    python scripts/test_agentic_features.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_rag.agent.loop import Agent
from agentic_rag.config import settings


def test_judge_module():
    """Test the Judge module functionality."""
    print("🧠 Testing Judge Module...")

    agent = Agent(gate_on=True, debug_mode=False)

    # Test questions that should trigger different Judge responses
    test_questions = [
        "What are cats?",  # Simple question - should work well
        "What is the exact population of Mars in 2024?",  # Impossible question
        "How many 3-point attempts did Steve Nash average in his 50-40-90 seasons?",  # Complex multi-hop
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question} ---")
        try:
            # Run just one step to see Judge in action
            result = agent.answer(question)

            # Extract key metrics
            final_answer = result.get("final_answer", "")
            judge_used = result.get("used_judge", False)
            rounds = result.get("rounds", 0)

            print(f"Judge Used: {judge_used}")
            print(f"Rounds: {rounds}")
            print(
                f"Answer: {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}"
            )

        except Exception as e:
            print(f"Error testing question: {e}")
            continue

    print("\n🧠 Judge Module test completed!")


def test_hybrid_search():
    """Test hybrid search functionality."""
    print("\n🔍 Testing Hybrid Search...")

    from agentic_rag.retriever.vector import VectorRetriever

    try:
        retriever = VectorRetriever(settings.FAISS_INDEX_PATH)

        if retriever.use_hybrid and retriever.bm25_retriever:
            print("✅ Hybrid search is enabled and BM25 index loaded")

            # Test a query that should benefit from keyword matching
            test_query = "Steve Nash basketball statistics"
            contexts, stats = retriever.retrieve_pack(test_query, k=5)

            print(f"Retrieved {len(contexts)} contexts")
            # No need to inspect individual context items further in this test

            print(f"Used hybrid: {stats.get('used_hybrid', False)}")
            print(f"Used rerank: {stats.get('used_rerank', False)}")
            print(f"Used MMR: {stats.get('used_mmr', False)}")

        else:
            print("⚠️  Hybrid search not available (BM25 index not found or disabled)")

    except Exception as e:
        print(f"Error testing hybrid search: {e}")

    print("🔍 Hybrid search test completed!")


def test_configuration():
    """Test that new configuration settings are properly loaded."""
    print("\n⚙️  Testing Configuration...")

    print(f"JUDGE_POLICY: {settings.JUDGE_POLICY}")
    print(f"USE_HYBRID_SEARCH: {getattr(settings, 'USE_HYBRID_SEARCH', False)}")
    print(f"HYBRID_ALPHA: {getattr(settings, 'HYBRID_ALPHA', 0.7)}")
    print(f"USE_RERANK: {settings.USE_RERANK}")
    print(f"USE_HYDE: {settings.USE_HYDE}")
    print(f"MMR_LAMBDA: {settings.MMR_LAMBDA}")

    print("⚙️  Configuration test completed!")


def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Agentic RAG System")
    print("=" * 50)

    # Test configuration first
    test_configuration()

    # Test hybrid search capability
    test_hybrid_search()

    # Test Judge module (this will be the main test)
    test_judge_module()

    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nNext steps:")
    print(
        "1. Run full evaluation: python -m agentic_rag.eval.runner run --dataset data/crag_task_1_and_2_dev_v4.jsonl --n 50 --system agent"
    )
    print("2. Compare results with baseline run ")
    print("3. Monitor Judge invocation rate and query transformation usage")


if __name__ == "__main__":
    main()
