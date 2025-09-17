"""Performance optimizations for uncertainty gate and retrieval."""

import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, Optional


class LRUCache:
    """Simple LRU cache implementation for gate decisions."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, total),
            "size": len(self.cache),
            "maxsize": self.maxsize,
        }


class PerformanceProfiler:
    """Simple performance profiler for gate operations."""

    def __init__(self):
        self.timings: Dict[str, list[float]] = {}
        self.call_counts: Dict[str, int] = {}

    def time_function(self, func_name: str):
        """Decorator to time function calls."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    if func_name not in self.timings:
                        self.timings[func_name] = []
                        self.call_counts[func_name] = 0

                    self.timings[func_name].append(duration)
                    self.call_counts[func_name] += 1

            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    "count": self.call_counts[func_name],
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
        return stats

    def reset(self) -> None:
        """Reset all statistics."""
        self.timings.clear()
        self.call_counts.clear()


# Global instances
gate_cache = LRUCache(maxsize=256)
profiler = PerformanceProfiler()


def cached_gate_decision(cache_key: str, compute_func: Callable[[], str]) -> str:
    """Cache gate decisions for identical inputs."""
    cached_result = gate_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    result = compute_func()
    gate_cache.put(cache_key, result)
    return result


def create_cache_key(signals) -> str:
    """Create a cache key from gate signals."""
    # Create a hash-like key from key signal values
    key_components = [
        f"f:{signals.faith:.3f}",
        f"o:{signals.overlap:.3f}",
        f"lex:{signals.lexical_uncertainty:.3f}",
        f"comp:{signals.completeness:.3f}",
        f"sem:{getattr(signals, 'semantic_coherence', 1.0):.3f}",
        f"r:{signals.round_idx}",
        f"nov:{signals.novelty_ratio:.3f}",
        f"budget:{signals.budget_left_tokens // 100}00",  # Round to nearest 100
    ]
    return "|".join(key_components)


class BatchProcessor:
    """Batch process multiple uncertainty assessments for efficiency."""

    @staticmethod
    def batch_assess_lexical_uncertainty(responses: list[str]) -> list[float]:
        """Process multiple responses in batch for better performance."""
        if not responses:
            return []

        # Pre-compile patterns once for all responses
        import re

        uncertainty_pattern = re.compile(
            r"\b(might|maybe|perhaps|possibly|likely|probably|seems|appears|"
            r"suggests|indicates|unclear|uncertain|not sure|don\'t know|"
            r"can\'t say|difficult to determine|unsure|ambiguous)\b",
            re.IGNORECASE,
        )

        results = []
        for response in responses:
            if not response or len(response.strip()) < 3:
                results.append(1.0)
                continue

            matches = uncertainty_pattern.findall(response.lower())
            uncertainty_score = min(1.0, len(matches) * 0.25)
            results.append(uncertainty_score)

        return results


def optimize_retrieval_pipeline():
    """Optimize the retrieval pipeline for better performance."""
    # This would contain optimizations for the retrieval process
    # such as parallel processing, caching, etc.
    pass
