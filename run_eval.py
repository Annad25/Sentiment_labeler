from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from sentiment_eval.config import EvalConfig
from sentiment_eval.data import detect_data_issues, load_dataset
from sentiment_eval.metrics import compute_metrics
from sentiment_eval.report import print_console_summary, write_reports
from sentiment_eval.runner import run_eval


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentiment classifier eval harness.")
    p.add_argument("--model", default=None, help="OpenAI model id (default: gpt-4o-mini).")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Max concurrent API calls (default: 10). Use 1 for serial.")
    p.add_argument("--dataset", type=Path, default=None, help="Path to CSV dataset.")
    p.add_argument("--results-dir", type=Path, default=None, help="Output directory.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any prior raw.jsonl and re-run every row.")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    if args.concurrency is not None:
        overrides["max_concurrency"] = args.concurrency
    if args.dataset:
        overrides["dataset_path"] = args.dataset
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
    cfg = EvalConfig(**overrides)

    if not cfg.dataset_path.exists():
        print(f"ERROR: dataset not found at {cfg.dataset_path}", file=sys.stderr)
        return 2

    examples = load_dataset(cfg.dataset_path)
    issues = detect_data_issues(examples)
    print(f"Loaded {len(examples)} examples from {cfg.dataset_path}")
    if issues.exact_duplicates_conflicting or issues.near_duplicates_conflicting:
        print(f"  Pre-flight: found {len(issues.exact_duplicates_conflicting)} exact "
              f"and {len(issues.near_duplicates_conflicting)} near-duplicate "
              f"label conflicts in the ground truth.")

    jsonl_path = cfg.results_dir / "raw.jsonl"
    if args.no_resume and jsonl_path.exists():
        jsonl_path.unlink()

    results = asyncio.run(run_eval(examples, cfg, jsonl_path, resume=not args.no_resume))
    metrics = compute_metrics(results)
    write_reports(results, metrics, issues, cfg, cfg.results_dir)
    print_console_summary(metrics, issues, results, cfg)

    print(f"Wrote: {cfg.results_dir / 'summary.json'}")
    print(f"Wrote: {cfg.results_dir / 'results.csv'}")
    print(f"Wrote: {cfg.results_dir / 'raw.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
