"""Tests for the scoring math itself.

This is the answer to "how do you know your evaluation harness is correct?":
feed in a tiny set of results with a confusion matrix worked out by hand, and
assert every number the harness reports.
"""
from sentiment_eval.metrics import baseline_majority_accuracy, compute_metrics
from sentiment_eval.runner import RowResult


def _row(id: int, expected: str, predicted: str) -> RowResult:
    return RowResult(
        id=id,
        snippet=f"snippet {id}",
        expected=expected,
        predicted=predicted,
        raw_output=predicted,
        correct=(expected == predicted),
        error=None,
    )


# A deliberately tiny, hand-checkable set.
#
#   id  expected   predicted     outcome
#   1   positive   positive      correct
#   2   positive   negative      miss
#   3   neutral    neutral       correct
#   4   neutral    positive      miss
#   5   negative   negative      correct
#   6   negative   unparseable   miss (and unparseable)
#
# Confusion (expected -> predicted):
#   positive: {positive:1, negative:1}
#   neutral:  {neutral:1,  positive:1}
#   negative: {negative:1, unparseable:1}
_RESULTS = [
    _row(1, "positive", "positive"),
    _row(2, "positive", "negative"),
    _row(3, "neutral", "neutral"),
    _row(4, "neutral", "positive"),
    _row(5, "negative", "negative"),
    _row(6, "negative", "unparseable"),
]


def test_headline_counts():
    m = compute_metrics(_RESULTS)
    assert m.total == 6
    assert m.correct == 3
    assert m.accuracy == 0.5
    assert m.unparseable == 1
    assert m.errored == 0


def test_confusion_matrix():
    m = compute_metrics(_RESULTS)
    cm = m.confusion_matrix
    assert cm["positive"]["positive"] == 1
    assert cm["positive"]["negative"] == 1
    assert cm["neutral"]["neutral"] == 1
    assert cm["neutral"]["positive"] == 1
    assert cm["negative"]["negative"] == 1
    assert cm["negative"]["unparseable"] == 1
    # everything else is zero
    assert cm["positive"]["neutral"] == 0
    assert cm["neutral"]["negative"] == 0


def test_per_class_precision_recall_f1():
    m = compute_metrics(_RESULTS)
    by_label = {c.label: c for c in m.per_class}

    # positive: tp=1, fp=1 (row4 predicted positive), fn=1 (row2) -> P=R=F1=0.5
    assert by_label["positive"].precision == 0.5
    assert by_label["positive"].recall == 0.5
    assert by_label["positive"].f1 == 0.5
    assert by_label["positive"].support == 2

    # neutral: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=0.6667
    assert by_label["neutral"].precision == 1.0
    assert by_label["neutral"].recall == 0.5
    assert by_label["neutral"].f1 == 0.6667

    # negative: tp=1, fp=1 (row2), fn=1 (row6) -> P=R=F1=0.5
    assert by_label["negative"].precision == 0.5
    assert by_label["negative"].recall == 0.5
    assert by_label["negative"].f1 == 0.5


def test_baseline_majority():
    # 2 of each label -> majority share is 2/6.
    assert baseline_majority_accuracy(_RESULTS) == round(2 / 6, 4)


def test_empty_input_does_not_crash():
    m = compute_metrics([])
    assert m.total == 0
    assert m.accuracy == 0.0
    assert baseline_majority_accuracy([]) == 0.0
