from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agentic_rag.config import settings


def _ensure_log_dir() -> Path:
    d = Path(
        getattr(settings, "log_dir", None)
        or getattr(settings, "LOG_DIR", None)
        or "logs"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_round(qid: str, round_idx: int, payload: dict[str, Any]) -> None:
    d = _ensure_log_dir()
    path = d / f"anchor_{qid}.jsonl"
    data = payload.copy()
    data.setdefault("qid", qid)
    data.setdefault("round", round_idx)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_jsonl_line(data))


def log_summary(qid: str, payload: dict[str, Any]) -> None:
    d = _ensure_log_dir()
    path = d / f"anchor_{qid}.jsonl"
    data = payload.copy()
    data.setdefault("qid", qid)
    data.setdefault("kind", "summary")
    with open(path, "a", encoding="utf-8") as f:
        f.write(_jsonl_line(data))


def _jsonl_line(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, ensure_ascii=False) + "\n"
