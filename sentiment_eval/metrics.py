from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import LABELS
from .runner import RowResult


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalMetrics:
    total: int
    correct: int
    unparseable: int
    errored: int
    accuracy: float
    per_class: list[ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]  # expected -> predicted -> count


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(results: list[RowResult]) -> EvalMetrics:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    unparseable = sum(1 for r in results if r.predicted == "unparseable")
    errored = sum(1 for r in results if r.error)

    confusion: dict[str, dict[str, int]] = {
        e: {p: 0 for p in (*LABELS, "unparseable")} for e in LABELS
    }
    for r in results:
        if r.expected in confusion and r.predicted in confusion[r.expected]:
            confusion[r.expected][r.predicted] += 1

    per_class: list[ClassMetrics] = []
    pred_totals = Counter(r.predicted for r in results)
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label][p] for p in confusion[label] if p != label)
        fp = pred_totals[label] - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            ClassMetrics(
                label=label,
                support=tp + fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
            )
        )

    return EvalMetrics(
        total=total,
        correct=correct,
        unparseable=unparseable,
        errored=errored,
        accuracy=round(_safe_div(correct, total), 4),
        per_class=per_class,
        confusion_matrix=confusion,
    )


def baseline_majority_accuracy(results: list[RowResult]) -> float:
    """Sanity check: a 'predict-the-most-common-label' baseline.

    If our accuracy isn't well above this, something's wrong with the harness
    OR the model is no better than guessing.
    """
    if not results:
        return 0.0
    counts = Counter(r.expected for r in results)
    majority_label, majority_count = counts.most_common(1)[0]
    return round(majority_count / len(results), 4)
