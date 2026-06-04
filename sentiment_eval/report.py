from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import EvalConfig, LABELS
from .data import DataIssues
from .metrics import EvalMetrics, baseline_majority_accuracy
from .runner import RowResult


def _disagreements(results: list[RowResult]) -> list[dict]:
    return [
        {
            "id": r.id,
            "snippet": r.snippet,
            "expected": r.expected,
            "predicted": r.predicted,
            "raw_output": r.raw_output,
            "error": r.error,
        }
        for r in results
        if not r.correct
    ]


def _adjusted_accuracy(
    results: list[RowResult], issues: DataIssues
) -> tuple[float, list[int]]:
    """Accuracy after excluding rows where the ground truth itself is broken.

    We exclude all ids involved in any duplicate-with-conflicting-label group,
    since at most one of them can be 'right' under any labeling scheme.
    """
    suspect_ids: set[int] = set()
    for a, b, _ in issues.exact_duplicates_conflicting:
        suspect_ids.update({a, b})
    for a, b, _, _ in issues.near_duplicates_conflicting:
        suspect_ids.update({a, b})

    kept = [r for r in results if r.id not in suspect_ids]
    if not kept:
        return 0.0, sorted(suspect_ids)
    acc = sum(1 for r in kept if r.correct) / len(kept)
    return round(acc, 4), sorted(suspect_ids)


def write_reports(
    results: list[RowResult],
    metrics: EvalMetrics,
    issues: DataIssues,
    cfg: EvalConfig,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    adjusted_acc, excluded_ids = _adjusted_accuracy(results, issues)
    baseline = baseline_majority_accuracy(results)

    summary = {
        "config": cfg.to_dict(),
        "metrics": {
            "raw_accuracy": metrics.accuracy,
            "adjusted_accuracy_excluding_suspect_labels": adjusted_acc,
            "majority_baseline_accuracy": baseline,
            "total": metrics.total,
            "correct": metrics.correct,
            "unparseable": metrics.unparseable,
            "errored": metrics.errored,
            "per_class": [asdict(c) for c in metrics.per_class],
            "confusion_matrix": metrics.confusion_matrix,
            "provider_usage": dict(Counter(r.provider for r in results)),
        },
        "data_quality": {
            "exact_duplicates_with_conflicting_labels": [
                {"id_a": a, "id_b": b, "snippet": s}
                for a, b, s in issues.exact_duplicates_conflicting
            ],
            "near_duplicates_with_conflicting_labels": [
                {"id_a": a, "id_b": b, "snippet_a": sa, "snippet_b": sb}
                for a, b, sa, sb in issues.near_duplicates_conflicting
            ],
            "excluded_from_adjusted_accuracy": excluded_ids,
        },
        "disagreements": _disagreements(results),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "id", "snippet", "expected", "predicted",
                "correct", "raw_output", "error", "latency_ms",
                "provider", "model",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def print_console_summary(
    metrics: EvalMetrics,
    issues: DataIssues,
    results: list[RowResult],
    cfg: EvalConfig,
) -> None:
    adjusted_acc, _ = _adjusted_accuracy(results, issues)
    baseline = baseline_majority_accuracy(results)

    print()
    print(f"Model:    {cfg.model}  (temperature={cfg.temperature})")
    print(f"Total:    {metrics.total}")
    print(f"Correct:  {metrics.correct}")
    print(f"Errored:  {metrics.errored}   Unparseable: {metrics.unparseable}")
    usage = Counter(r.provider for r in results)
    print(f"Provider: {', '.join(f'{k}={v}' for k, v in usage.items())}")
    print()
    print(f"Raw accuracy:                 {metrics.accuracy:.2%}")
    print(f"Adjusted (excl. bad labels):  {adjusted_acc:.2%}")
    print(f"Majority-class baseline:      {baseline:.2%}")
    print()
    print("Per-class:")
    print(f"  {'label':<10} {'support':>8} {'precision':>10} {'recall':>8} {'f1':>6}")
    for c in metrics.per_class:
        print(f"  {c.label:<10} {c.support:>8} {c.precision:>10.3f} {c.recall:>8.3f} {c.f1:>6.3f}")

    print()
    print("Confusion matrix (rows=expected, cols=predicted):")
    cols = (*LABELS, "unparseable")
    header = " " * 12 + "".join(f"{c[:9]:>11}" for c in cols)
    print(header)
    for e in LABELS:
        row = f"  {e:<10}" + "".join(f"{metrics.confusion_matrix[e][p]:>11d}" for p in cols)
        print(row)

    if issues.exact_duplicates_conflicting or issues.near_duplicates_conflicting:
        print()
        print("Data-quality flags (suspect ground truth):")
        for a, b, s in issues.exact_duplicates_conflicting:
            print(f"  ids {a},{b}: exact duplicate with conflicting labels - {s!r}")
        for a, b, sa, sb in issues.near_duplicates_conflicting:
            print(f"  ids {a},{b}: near-duplicate with conflicting labels - {sa!r} / {sb!r}")

    misses = [r for r in results if not r.correct]
    if misses:
        print()
        print(f"Disagreements ({len(misses)}):")
        for r in misses:
            tag = f"[err: {r.error}]" if r.error else ""
            print(f"  id {r.id:>3}  expected={r.expected:<8} predicted={r.predicted:<12} {r.snippet!r} {tag}")
    print()
