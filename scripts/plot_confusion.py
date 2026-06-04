"""Render a confusion-matrix heatmap (Jupyter-style) from a results CSV.

Usage:
    python scripts/plot_confusion.py                      # uses results/results.csv
    python scripts/plot_confusion.py results/results_1.csv
    python scripts/plot_confusion.py results/results_1.csv --out results/cm.png

Requires matplotlib:  pip install matplotlib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

# Fixed label order so the matrix is always laid out the same way.
LABELS = ["positive", "neutral", "negative"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a confusion-matrix heatmap.")
    ap.add_argument("csv", nargs="?", default="results/results.csv",
                    help="results CSV with 'expected' and 'predicted' columns")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Build the matrix in a fixed label order (include any label actually seen).
    labels = LABELS + [l for l in df["predicted"].unique() if l not in LABELS]
    cm = pd.crosstab(df["expected"], df["predicted"]).reindex(
        index=LABELS, columns=labels, fill_value=0
    )

    acc = (df["expected"] == df["predicted"]).mean()
    model = df["model"].dropna().iloc[0] if "model" in df.columns and df["model"].notna().any() else "model"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected (ground truth)")
    ax.set_title(f"Confusion matrix — {model}\naccuracy = {acc:.0%}  ({(df.expected==df.predicted).sum()}/{len(df)})")

    # Annotate each cell with its count; white text on dark cells for contrast.
    vmax = cm.values.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax / 2 else "black", fontsize=12)

    ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(cm.columns))))
    ax.yaxis.set_major_locator(mticker.FixedLocator(range(len(cm.index))))
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()

    out = Path(args.out) if args.out else csv_path.with_suffix(".confusion.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
