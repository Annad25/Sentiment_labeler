"""Tests for the data-quality detector.

The whole point of this project is to NOT blindly trust the ground truth.
`detect_data_issues` is what surfaces the planted label problems (rows 56/59
exact-conflict, rows 23/47 near-conflict). These tests prove it catches them
on a tiny synthetic dataset, independent of the real file.
"""
from pathlib import Path

from sentiment_eval.config import DATA_DIR
from sentiment_eval.data import (
    Example,
    _normalize_text,
    detect_data_issues,
    load_dataset,
)


def test_exact_duplicate_conflict_is_flagged():
    examples = [
        Example(1, "We already have a vendor.", "negative"),
        Example(2, "We already have a vendor.", "positive"),  # same text, diff label
    ]
    issues = detect_data_issues(examples)
    assert len(issues.exact_duplicates_conflicting) == 1
    a, b, snippet = issues.exact_duplicates_conflicting[0]
    assert {a, b} == {1, 2}
    assert issues.near_duplicates_conflicting == []


def test_near_duplicate_conflict_is_flagged():
    # Differ only by trailing punctuation -> near, not exact.
    examples = [
        Example(23, "I'd be interested in pricing.", "positive"),
        Example(47, "I'd be interested in pricing", "neutral"),
    ]
    issues = detect_data_issues(examples)
    assert issues.exact_duplicates_conflicting == []
    assert len(issues.near_duplicates_conflicting) == 1
    a, b, _, _ = issues.near_duplicates_conflicting[0]
    assert {a, b} == {23, 47}


def test_duplicate_with_same_label_is_not_flagged():
    # Same snippet, SAME label -> harmless, must not be reported.
    examples = [
        Example(1, "Sounds good.", "positive"),
        Example(2, "Sounds good.", "positive"),
    ]
    issues = detect_data_issues(examples)
    assert issues.exact_duplicates_conflicting == []
    assert issues.near_duplicates_conflicting == []


def test_unique_rows_produce_no_issues():
    examples = [
        Example(1, "Yes, let's do it.", "positive"),
        Example(2, "Take me off your list.", "negative"),
        Example(3, "What's the price?", "neutral"),
    ]
    issues = detect_data_issues(examples)
    assert issues.exact_duplicates_conflicting == []
    assert issues.near_duplicates_conflicting == []


def test_curly_apostrophe_folds_to_straight():
    # Row 28 (straight) vs row 36 (curly U+2019) must collapse to one key.
    straight = "Yes please, I'd like to proceed."
    curly = "Yes please, I’d like to proceed."
    assert _normalize_text(straight) == _normalize_text(curly)


def test_normalize_text_strips_case_and_trailing_punctuation():
    assert _normalize_text("  Sounds GREAT! ") == "sounds great"
    assert _normalize_text("Okay.") == "okay"


# --- Integration: the real provided dataset -------------------------------

def test_real_dataset_loads_and_has_known_quirks():
    path = DATA_DIR / "sentiment_eval_dataset.csv"
    if not path.exists():
        return  # dataset not present in this checkout; unit tests above still cover logic

    examples = load_dataset(path)
    assert len(examples) == 100

    issues = detect_data_issues(examples)
    # Rows 56/59 are the exact-duplicate conflict planted in the data.
    flagged_exact = {frozenset((a, b)) for a, b, _ in issues.exact_duplicates_conflicting}
    assert frozenset((56, 59)) in flagged_exact
