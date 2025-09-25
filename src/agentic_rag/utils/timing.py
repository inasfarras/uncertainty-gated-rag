"""Timing utilities for measuring latency and performance."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, Optional
try:
    # Python 3.11+
    from typing import Self  # type: ignore
except Exception:  # Python <3.11
    from typing_extensions import Self  # type: ignore


class Timer:
    """Timer for measuring execution time of code blocks."""

    def __init__(self) -> None:
        """Initialize timer."""
        self._start_times: Dict[str, float] = {}
        self._times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """
        Start timing a named operation.

        Args:
            name: Name of the operation to time
        """
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Stop timing a named operation.

        Args:
            name: Name of the operation to stop timing

        Returns:
            Elapsed time in seconds

        Raises:
            KeyError: If the named operation was not started
        """
        if name not in self._start_times:
            raise KeyError(f"Timer '{name}' was not started")

        elapsed = time.perf_counter() - self._start_times[name]
        self._times[name] = elapsed
        del self._start_times[name]
        return elapsed

    def get_time(self, name: str) -> Optional[float]:
        """
        Get the recorded time for a named operation.

        Args:
            name: Name of the operation

        Returns:
            Elapsed time in seconds, or None if not recorded
        """
        return self._times.get(name)

    def get_times(self) -> Dict[str, float]:
        """
        Get all recorded times.

        Returns:
            Dictionary mapping operation names to elapsed times
        """
        return self._times.copy()

    def reset(self) -> None:
        """Reset all timers and recorded times."""
        self._start_times.clear()
        self._times.clear()

    @contextmanager
    def time_block(self, name: str) -> Generator[Self, None, None]:
        """
        Context manager for timing a code block.

        Args:
            name: Name of the operation to time

        Yields:
            Timer instance
        """
        self.start(name)
        try:
            yield self
        finally:
            self.stop(name)


def measure_time(func: Any) -> Any:
    """
    Decorator to measure execution time of a function.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that prints execution time
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


async def measure_async_time(func: Any) -> Any:
    """
    Decorator to measure execution time of an async function.

    Args:
        func: Async function to decorate

    Returns:
        Decorated async function that prints execution time
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


class LatencyTracker:
    """Track latency statistics over multiple measurements."""

    def __init__(self, name: str) -> None:
        """
        Initialize latency tracker.

        Args:
            name: Name of the operation being tracked
        """
        self.name = name
        self._measurements: list[float] = []

    def record(self, latency: float) -> None:
        """
        Record a latency measurement.

        Args:
            latency: Latency in seconds
        """
        self._measurements.append(latency)

    def get_stats(self) -> Dict[str, float]:
        """
        Get latency statistics.

        Returns:
            Dictionary with min, max, mean, median latency
        """
        if not self._measurements:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_measurements = sorted(self._measurements)
        n = len(sorted_measurements)

        return {
            "count": n,
            "min": sorted_measurements[0],
            "max": sorted_measurements[-1],
            "mean": sum(sorted_measurements) / n,
            "median": sorted_measurements[n // 2],
            "p95": sorted_measurements[int(0.95 * n)],
            "p99": sorted_measurements[int(0.99 * n)],
        }

    def reset(self) -> None:
        """Reset all measurements."""
        self._measurements.clear()

    @contextmanager
    def measure(self) -> Generator[Self, None, None]:
        """
        Context manager to measure and record latency.

        Yields:
            LatencyTracker instance
        """
        start_time = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - start_time
            self.record(elapsed)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_seconds = seconds % 3600
        minutes = int(remaining_seconds // 60)
        remaining_seconds = remaining_seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.0f}s"


class Stopwatch:
    """Simple stopwatch for manual timing."""

    def __init__(self) -> None:
        """Initialize stopwatch."""
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        """Start the stopwatch."""
        if not self._running:
            self._start_time = time.perf_counter()
            self._running = True

    def stop(self) -> float:
        """
        Stop the stopwatch.

        Returns:
            Total elapsed time in seconds
        """
        if self._running and self._start_time is not None:
            self._elapsed += time.perf_counter() - self._start_time
            self._running = False
        return self._elapsed

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._start_time = None
        self._elapsed = 0.0
        self._running = False

    def lap(self) -> float:
        """
        Get lap time without stopping the stopwatch.

        Returns:
            Current elapsed time in seconds
        """
        current_elapsed = self._elapsed
        if self._running and self._start_time is not None:
            current_elapsed += time.perf_counter() - self._start_time
        return current_elapsed

    @property
    def elapsed(self) -> float:
        """Get current elapsed time."""
        return self.lap()

    @property
    def is_running(self) -> bool:
        """Check if stopwatch is running."""
        return self._running


@contextmanager
def timer():
    """A context manager to measure execution time."""
    start_time = time.perf_counter()
    # Yield a function that returns the elapsed time in milliseconds
    yield lambda: (time.perf_counter() - start_time) * 1000
