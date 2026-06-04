from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import LABELS, Label


@dataclass(frozen=True)
class Example:
    id: int
    snippet: str
    label: Label


@dataclass(frozen=True)
class DataIssues:
    """Ground-truth quality flags surfaced before scoring.

    Why: a few rows in the provided dataset have suspect labels. We do not
    silently fix them — we report them so a reviewer can decide.
    """
    exact_duplicates_conflicting: list[tuple[int, int, str]]   # (id_a, id_b, snippet)
    near_duplicates_conflicting: list[tuple[int, int, str, str]]  # (id_a, id_b, snippet_a, snippet_b)


# NFKC alone does NOT fold curly quotes/dashes (U+2019 has no compatibility
# decomposition), so we map them explicitly before normalizing. This is what
# makes the curly-apostrophe twin (row 36) collapse onto its straight-quote
# counterpart (row 28).
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'",   # ‘ ’ single quotes
    "“": '"', "”": '"',   # “ ” double quotes
    "–": "-", "—": "-",   # – — dashes
})


def _normalize_text(s: str) -> str:
    folded = unicodedata.normalize("NFKC", s).translate(_PUNCT_FOLD)
    return folded.strip().lower().rstrip(".!?")


def load_dataset(path: Path) -> list[Example]:
    # utf-8-sig handles a possible BOM; pandas already decodes the curly
    # apostrophe in row 36 correctly under utf-8.
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"id", "snippet", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    df = df.dropna(subset=["id", "snippet", "label"])
    df["label"] = df["label"].str.strip().str.lower()

    bad = df[~df["label"].isin(LABELS)]
    if not bad.empty:
        raise ValueError(f"unknown labels at ids {bad['id'].tolist()}")

    return [
        Example(id=int(r.id), snippet=str(r.snippet), label=r.label)  # type: ignore[arg-type]
        for r in df.itertuples(index=False)
    ]


def detect_data_issues(examples: list[Example]) -> DataIssues:
    """Find rows where the ground truth itself is internally inconsistent.

    - exact duplicates with conflicting labels (e.g. rows 56/59)
    - near-duplicates (after punctuation/case fold) with conflicting labels
      (e.g. rows 23/47)
    """
    by_key: dict[str, list[Example]] = {}
    for ex in examples:
        by_key.setdefault(_normalize_text(ex.snippet), []).append(ex)

    exact: list[tuple[int, int, str]] = []
    near: list[tuple[int, int, str, str]] = []
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        labels = {g.label for g in group}
        if len(labels) == 1:
            continue
        # Exact match on raw snippet vs only-after-normalization match
        raw_set = {g.snippet for g in group}
        a, b = group[0], group[1]
        if len(raw_set) == 1:
            exact.append((a.id, b.id, a.snippet))
        else:
            near.append((a.id, b.id, a.snippet, b.snippet))

    return DataIssues(
        exact_duplicates_conflicting=sorted(exact),
        near_duplicates_conflicting=sorted(near),
    )
