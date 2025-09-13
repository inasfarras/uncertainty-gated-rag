"""JSONL utilities for reading and writing JSON Lines format using orjson."""

from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Union, overload

import orjson


class JSONLReader:
    """Reader for JSONL (JSON Lines) files."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize JSONL reader.

        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = Path(file_path)

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all lines from the JSONL file.

        Returns:
            List of parsed JSON objects
        """
        data = []
        with open(self.file_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(orjson.loads(line))
        return data

    @overload
    def iterate(self, batch_size: None = None) -> Iterator[Dict[str, Any]]: ...

    @overload
    def iterate(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]: ...

    def iterate(
        self,
        batch_size: Optional[int] = None,
    ) -> Iterator[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Iterate over lines in the JSONL file.

        Args:
            batch_size: If specified, yield batches of this size

        Yields:
            Individual JSON objects or batches of objects
        """
        if batch_size is None:
            with open(self.file_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield orjson.loads(line)
        else:
            batch: List[Dict[str, Any]] = []
            with open(self.file_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        batch.append(orjson.loads(line))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

                # Yield remaining items
                if batch:
                    yield batch

    def count_lines(self) -> int:
        """
        Count the number of lines in the JSONL file.

        Returns:
            Number of lines
        """
        count = 0
        with open(self.file_path, "rb") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class JSONLWriter:
    """Writer for JSONL (JSON Lines) files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "w",
        ensure_ascii: bool = False,
    ) -> None:
        """
        Initialize JSONL writer.

        Args:
            file_path: Path to the output JSONL file
            mode: File open mode ("w" for write, "a" for append)
            ensure_ascii: Whether to ensure ASCII-only output
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.ensure_ascii = ensure_ascii
        self._file: Optional[IO[Any]] = None

        # Create parent directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "JSONLWriter":
        """Context manager entry."""
        self._file = open(self.file_path, self.mode + "b")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        if self._file:
            self._file.close()
            self._file = None

    def write(self, data: Dict[str, Any]) -> None:
        """
        Write a single JSON object to the file.

        Args:
            data: Dictionary to write as JSON
        """
        # No need for this check, as __enter__ ensures _file is not None
        # if self._file is None:
        #     raise RuntimeError(
        #         "Writer not opened. Use as context manager or call open()."
        #     )

        json_bytes = orjson.dumps(
            data,
            option=orjson.OPT_APPEND_NEWLINE
            | (orjson.OPT_NON_STR_KEYS if not self.ensure_ascii else 0),
        )
        if self._file:
            self._file.write(json_bytes)

    def write_batch(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Write multiple JSON objects to the file.

        Args:
            data_list: List of dictionaries to write
        """
        for data in data_list:
            self.write(data)

    def open(self) -> None:
        """Open the file for writing."""
        if self._file is None:
            self._file = open(self.file_path, self.mode + "b")

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Convenience function to read all data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    reader = JSONLReader(file_path)
    return reader.read_all()


def write_jsonl(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    mode: str = "w",
) -> None:
    """
    Convenience function to write data to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output file
        mode: File open mode
    """
    with JSONLWriter(file_path, mode) as writer:
        writer.write_batch(data)


def append_jsonl(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Convenience function to append a single record to a JSONL file.

    Args:
        data: Dictionary to append
        file_path: Path to the JSONL file
    """
    with JSONLWriter(file_path, "a") as writer:
        writer.write(data)


def merge_jsonl_files(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path],
) -> int:
    """
    Merge multiple JSONL files into a single file.

    Args:
        input_files: List of input JSONL file paths
        output_file: Path to the output merged file

    Returns:
        Total number of records merged
    """
    total_records = 0

    with JSONLWriter(output_file, "w") as writer:
        for input_file in input_files:
            reader = JSONLReader(input_file)
            for record in reader.iterate():
                writer.write(record)
                total_records += 1

    return total_records


def filter_jsonl(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    filter_func: Callable[[Dict[str, Any]], bool],
) -> int:
    """
    Filter a JSONL file based on a predicate function.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output filtered file
        filter_func: Function that takes a record and returns True to keep it

    Returns:
        Number of records kept after filtering
    """
    kept_records = 0

    reader = JSONLReader(input_file)
    with JSONLWriter(output_file, "w") as writer:
        for record in reader.iterate():
            if filter_func(record):
                writer.write(record)
                kept_records += 1

    return kept_records


def sample_jsonl(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    n_samples: int,
    random_seed: int = 42,
) -> int:
    """
    Sample n records from a JSONL file.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output sampled file
        n_samples: Number of samples to extract
        random_seed: Random seed for reproducible sampling

    Returns:
        Number of records sampled
    """
    import random

    # Read all records
    reader = JSONLReader(input_file)
    all_records = reader.read_all()

    # Sample records
    random.seed(random_seed)
    if len(all_records) <= n_samples:
        sampled_records = all_records
    else:
        sampled_records = random.sample(all_records, n_samples)

    # Write sampled records
    write_jsonl(sampled_records, output_file)

    return len(sampled_records)
