from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .classifier import Classifier, ParsedLabel
from .config import EvalConfig
from .data import Example


@dataclass
class RowResult:
    id: int
    snippet: str
    expected: str
    predicted: ParsedLabel
    raw_output: str
    correct: bool
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None


def _load_cached(jsonl_path: Path) -> dict[int, RowResult]:
    """Resume support: skip rows already in the JSONL.

    Why: re-runs while iterating on the report should not re-pay the API.
    """
    cache: dict[int, RowResult] = {}
    if not jsonl_path.exists():
        return cache
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                cache[int(d["id"])] = RowResult(**d)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return cache


async def _run_one(
    clf: Classifier,
    ex: Example,
    sem: asyncio.Semaphore,
) -> RowResult:
    async with sem:
        t0 = time.perf_counter()
        try:
            pred = await clf.classify(ex.snippet)
            return RowResult(
                id=ex.id,
                snippet=ex.snippet,
                expected=ex.label,
                predicted=pred.label,
                raw_output=pred.raw,
                correct=(pred.label == ex.label),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                provider=pred.provider,
                model=pred.model,
            )
        except Exception as e:
            return RowResult(
                id=ex.id,
                snippet=ex.snippet,
                expected=ex.label,
                predicted="unparseable",
                raw_output="",
                correct=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
                provider=None,
            )


async def run_eval(
    examples: list[Example],
    cfg: EvalConfig,
    jsonl_path: Path,
    resume: bool = True,
) -> list[RowResult]:
    """Classify every example with bounded concurrency, streaming to JSONL.

    Bounded by Semaphore so we don't blow rate limits at higher dataset sizes;
    JSONL append is the resume mechanism.
    """
    cache = _load_cached(jsonl_path) if resume else {}
    pending = [ex for ex in examples if ex.id not in cache]

    clf = Classifier(cfg)
    sem = asyncio.Semaphore(cfg.max_concurrency)

    results: list[RowResult] = list(cache.values())

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with jsonl_path.open("a", encoding="utf-8") as fh:
            tasks = [asyncio.create_task(_run_one(clf, ex, sem)) for ex in pending]
            for fut in asyncio.as_completed(tasks):
                r = await fut
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                fh.flush()
                results.append(r)
    finally:
        await clf.aclose()

    results.sort(key=lambda r: r.id)
    return results
