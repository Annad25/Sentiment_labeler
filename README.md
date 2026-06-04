# Sentiment classifier eval harness

A deterministic evaluation harness for a 3-way sentiment classifier
(`positive` / `neutral` / `negative`) run against a labeled CSV of 100 customer
call snippets. It classifies each snippet with an LLM, scores predictions
against the ground truth by exact match, and reports accuracy plus a failure
breakdown — while surfacing rows where the ground truth itself looks wrong.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# put your OPENAI_API_KEY in .env (OPENROUTER_API_KEY is optional — see Fallback)

python run_eval.py
```

Outputs land in `results/`:

- `summary.json` — config, metrics, confusion matrix, data-quality flags, full disagreement list, and per-provider usage
- `results.csv` — one row per example: prediction, correctness, latency, **and the provider + exact model that produced it**
- `raw.jsonl` — streamed per-row results (also acts as a resume cache)

### CLI

```
python run_eval.py
python run_eval.py --concurrency 1        # serial instead of the default 10
python run_eval.py --no-resume            # ignore raw.jsonl and re-run all rows
python run_eval.py --model gpt-4o-mini    # use a different model (see Models)
```

### Tests

```bash
pip install pytest
python -m pytest -q          # 31 tests, no API key required
```

Use `python -m pytest` (not bare `pytest`) so the project root is on `sys.path`.

## Design choices

### Exact-match grader,
The grader is deterministic:

```
snippet → LLM → raw text → normalize() → predicted_label → string-compare to ground truth
```

There is no LLM-as-judge as the label set is closed and the ground truth
is provided, so exact match is the correct, reproducible grader. An LLM judge
would (a) be non-deterministic inside a measurement instrument and (b) tend to
agree with the classifier by construction (shared model biases), inflating the
score without measuring anything real.

### Choices

1. **Structured output + a regex backstop.** The primary path uses OpenAI
   structured output (`response_format=SentimentResponse`, a Pydantic `Literal`
   enum), constraining the model to the three valid words at decode time. The
   `normalize()` function is the safety net for any free-text output
   (`"Positive!"`, `"the sentiment is negative"`, trailing punctuation) and
   emits an explicit `unparseable` bucket rather than crashing or silently
   miscounting. Adversarially unit-tested.

2. **Prompts live in files, not code.** The system prompt is
   `prompts/system_prompt.txt`, loaded via `sentiment_eval/prompts.py`. It reads
   naturally (no `\n` escapes), can be edited without touching Python, and is
   reviewable on its own.

3. **Suspect ground truth is surfaced, not hidden.** `data.detect_data_issues`
   flags duplicate snippets carrying conflicting labels — exact duplicates and
   near-duplicates (after Unicode fold + lowercase + trailing-punct stripping).
   The report shows **raw accuracy** (the metric asked for) next to an
   **adjusted accuracy** that excludes ids in any conflicting-label group, with
   the excluded ids listed so a reviewer can audit the call. Single-row label
   errors (e.g. id 15) are *not* auto-excluded — they require human judgment, so
   they are surfaced in the disagreement list instead of silently corrected.

4. **Model-aware request building.** `config.MODEL_REGISTRY` + `resolve_model()`
   decide per model whether to send `reasoning_effort` (reasoning models like
   `gpt-5-mini`) or `temperature` (classic chat models like `gpt-4o-mini`).
   Driving this off a registry — with fallbacks for dated snapshots and
   vendor-prefixed OpenRouter ids — means adding a model is a one-line table
   edit and swapping models can never build an invalid request.

5. **A real failure breakdown.** Overall accuracy, per-class
   precision/recall/F1, the full 3×3 confusion matrix (plus an `unparseable`
   column), every disagreement (`id`, snippet, expected, predicted, raw output),
   and a **majority-class baseline** as a sanity check that the accuracy math
   itself reacts sensibly (~49% here).

6. **Bounded concurrency + resumable runs.** `asyncio.Semaphore` caps in-flight
   requests (default 10); `tenacity` retries with exponential backoff. Per-row
   results stream to `raw.jsonl` as they complete, so a crash or a re-run skips
   ids already done and doesn't re-pay the API.

## Models and provider fallback

### Default model
`gpt-5-mini`. It is a **reasoning model**, so the harness omits `temperature`
(reasoning models only accept the default) and sets `reasoning_effort="minimal"`
— 3-way sentiment needs no deliberation, and minimal effort keeps the hidden,
billed reasoning-token spend near zero.

### Fallback chain (optional)
If `OPENROUTER_API_KEY` is set, the classifier fails over across an ordered
chain; providers without a key are skipped automatically:

1. `openai` — OpenAI direct, the primary model
2. `openrouter:mirror` — the *same* model via OpenRouter (`openai/gpt-5-mini`); survives an OpenAI-side outage
3. `openrouter:deepseek-nex` — DeepSeek V3.1 Nex-N1, a *different* vendor via OpenRouter; survives the primary model being unavailable everywhere

Each row records which `provider` and `model` answered it (in `results.csv` and
`summary.json → provider_usage`). This matters: failover changes the measurement
instrument, so a reviewer must be able to see whether any row was scored by a
different model. For a pure benchmark, run with only `OPENAI_API_KEY` set so the
chain can't silently substitute a model.

> **Note on `openrouter:mirror`:** routing an OpenAI reasoning model through
> OpenRouter requires sufficient OpenRouter credits (the model reserves a large
> completion-token budget). On a free/unfunded OpenRouter account this tier
> returns HTTP 402 and the chain falls through to DeepSeek. The tier is kept
> intentionally — it is correct on a funded account, and the failover is real
> and observable.

### Determinism caveat
With `gpt-5-mini` we cannot set `temperature=0`, so runs are **near-stable, not
bit-for-bit reproducible**. Mitigations: a pinned model, `reasoning_effort=minimal`,
and persisting every raw model output to `raw.jsonl`. (A classic chat model such
as `gpt-4o-mini` *is* sent `temperature=0` and is effectively reproducible.)

### Cost
- `gpt-5-mini`: ~$0.25 / 1M input, ~$2 / 1M output → well under **$0.01** for a 100-row run (reasoning tokens billed, but minimal at `reasoning_effort=minimal`).
- DeepSeek V3.1 Nex-N1 (fallback): ~$0.135 / 1M input, ~$0.50 / 1M output — cheaper still.

Cost is identical for serial vs parallel — only wall-clock changes (~60–90s
serial, ~10s at concurrency 10).

### Scaling to 10,000 examples
Three swaps, no rewrites: (a) raise `max_concurrency` and lean on the existing
backoff/retry, (b) move to a Batch API for a large discount when latency isn't
critical, (c) keep the JSONL streaming layer — it already gives crash-resume and
is the right primitive at scale.

## Results in this repo

Two full runs are included to show the harness and the fallback both working:

| File | Model | Provider | Raw accuracy |
|------|-------|----------|--------------|
| `results/results_1.csv` | `gpt-5-mini` | OpenAI direct | 93% |
| `results/results.csv` | `nex-agi/deepseek-v3.1-nex-n1` | OpenRouter (failover) | 95% |

`summary.json` reflects the most recent run. Note its `config.model` records the
*intended primary* model, while `provider_usage` and the per-row `model` column
record what *actually* answered — in the DeepSeek run those differ, which is
exactly why per-row model capture exists.

**Run-to-run variance is real and expected.** `gpt-5-mini` is a reasoning model,
so we cannot set `temperature=0` — repeated runs land at 93–95%, with the flips
concentrated on borderline positive↔neutral snippets (e.g. ids 71, 98). This is
the determinism caveat in practice, and the reason every raw output is persisted
to `raw.jsonl`. A classic chat model at `temperature=0` (e.g. `gpt-4o-mini`)
would not show this.

### Reading the numbers (gpt-5-mini, `results_1.csv` = 93%)
Of the 7 disagreements: 3 are rows where the model is correct and the **label is
wrong** (ids 15, 59, 90); 1 is the unwinnable duplicate-with-conflicting-labels
case (id 47, twin of 23); the rest are genuine boundary errors (ids 38, 71, 98),
all positive↔neutral. Effective accuracy after the 3 label errors is ~96%, and
negatives are essentially clean.

### A note on few-shot prompting
Because the boundary errors are where the run-to-run variance lives, few-shot
prompting (held-out positive/neutral exemplars) was the natural lever to test.
It was tried as an experiment and **did not help** — a full 100-row run stayed
at 93% and merely shifted the error profile toward neutral (perfect neutral
recall, lower positive recall). The saved output is kept under
`results/fewshot/` as evidence, but the pipeline ships **zero-shot**: the
hypothesis was measured against the baseline and rejected rather than assumed.

## Notes on the dataset

A pre-flight pass flags rows where the ground truth itself looks wrong or
inconsistent (surfaced in `summary.json → data_quality` and the console, never
silently corrected):

- ids **56 / 59** — same sentence (`"We already have a vendor, we're not switching."`) labeled `negative` then `positive`. Hard contradiction.
- ids **23 / 47** — near-duplicate (`"I'd be very interested in learning more about pricing"` with/without a trailing period) labeled `positive` vs `neutral`. Genuinely fuzzy boundary.
- id **15** — `"I love that idea, let's do it."` labeled `negative`; reads unambiguously positive.
- id **90** — `"Is there a free trial available?"` labeled `negative`; but is neutral.
- id **36** — curly-apostrophe (U+2019) twin of id 28. `pandas` with `utf-8-sig` reads it cleanly; the duplicate detector folds curly quotes to straight ones explicitly (NFKC alone does **not** do this) so the twins collapse to one key.

## Layout

```
sentiment-eval/
├── run_eval.py                 CLI entrypoint
├── prompts/
│   └── system_prompt.txt       the classifier's system prompt (plain text)
├── sentiment_eval/
│   ├── config.py               model registry, request-param resolution, paths
│   ├── prompts.py              prompt loader
│   ├── providers.py            provider fallback chain (OpenAI → OpenRouter)
│   ├── data.py                 pandas load + validation + duplicate detection
│   ├── classifier.py           prompt + structured output + normalize() + failover
│   ├── runner.py               async + Semaphore + retries + JSONL streaming
│   ├── metrics.py              accuracy, per-class P/R/F1, confusion matrix
│   └── report.py               JSON + CSV + console summary
├── tests/
│   ├── test_normalize.py       output parsing (adversarial strings)
│   ├── test_metrics.py         hand-checked confusion matrix + P/R/F1
│   ├── test_data.py            duplicate/conflict detection + quote folding
│   ├── test_config.py          model registry resolution + request params
│   ├── test_prompts.py         prompt loading
│   └── test_providers.py       fallback chain order + key-based availability
├── data/sentiment_eval_dataset.csv
├── results/                    (created on first run)
├── requirements.txt
├── .env.example
└── .gitignore
```

loom video 
'https://www.loom.com/share/30f41e013326467fab8c83b2fff7cbe5'
