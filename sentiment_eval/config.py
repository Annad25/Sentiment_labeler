from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

Label = Literal["positive", "neutral", "negative"]
LABELS: tuple[Label, ...] = ("positive", "neutral", "negative")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROMPTS_DIR = ROOT / "prompts"


@dataclass(frozen=True)
class ModelSpec:
    """Capabilities that change how we build the API request.

    `reasoning` models (gpt-5*, o-series) reject a custom temperature and take
    `reasoning_effort`; classic chat models are the reverse.
    """
    reasoning: bool


# Source of truth for per-model request behaviour. Keyed by base model id
# (snapshot suffixes like '-2025-08-07' are stripped before lookup).
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # reasoning models
    "gpt-5": ModelSpec(reasoning=True),
    "gpt-5-mini": ModelSpec(reasoning=True),
    "gpt-5-nano": ModelSpec(reasoning=True),
    "o1": ModelSpec(reasoning=True),
    "o1-mini": ModelSpec(reasoning=True),
    "o3": ModelSpec(reasoning=True),
    "o3-mini": ModelSpec(reasoning=True),
    "o4-mini": ModelSpec(reasoning=True),
    # classic chat models
    "gpt-4o": ModelSpec(reasoning=False),
    "gpt-4o-mini": ModelSpec(reasoning=False),
    "gpt-4.1": ModelSpec(reasoning=False),
    "gpt-4.1-mini": ModelSpec(reasoning=False),
    "gpt-4-turbo": ModelSpec(reasoning=False),
}

_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def resolve_model(model: str) -> ModelSpec:
    """Look up a model's capabilities from the registry.

    Falls back gracefully so the harness keeps working for ids we haven't
    catalogued yet:
      0. strip a vendor prefix from OpenRouter ids ('openai/gpt-5-mini' ->
         'gpt-5-mini') so the same model resolves identically on either route
      1. exact match on the lowercased name
      2. match after stripping a dated snapshot suffix (e.g.
         'gpt-5-mini-2025-08-07' -> 'gpt-5-mini')
      3. name-based heuristic (gpt-5* / o<digit>* are reasoning models)
    """
    name = model.lower()
    if "/" in name:  # OpenRouter vendor prefix, e.g. 'openai/gpt-5-mini'
        name = name.rsplit("/", 1)[-1]
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    base = _SNAPSHOT_SUFFIX.sub("", name)
    if base in MODEL_REGISTRY:
        return MODEL_REGISTRY[base]
    reasoning = base.startswith("gpt-5") or bool(re.match(r"o\d", base))
    return ModelSpec(reasoning=reasoning)


def request_overrides_for(
    model: str,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
) -> dict:
    """Per-family request params for a given model.

    Reasoning models take `reasoning_effort` and reject a custom temperature;
    classic chat models take `temperature`. Free function (not just a method)
    so the provider chain can compute overrides per provider-specific model id
    — including vendor-prefixed OpenRouter ids like 'openai/gpt-5-mini'.
    """
    if resolve_model(model).reasoning:
        return {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return {"temperature": temperature if temperature is not None else 0.0}


@dataclass(frozen=True)
class EvalConfig:
    # gpt-5-mini is a reasoning model: it rejects a custom `temperature` (only
    # the default is allowed) and instead exposes `reasoning_effort`. We set
    # temperature=None so it is omitted from the request, and turn reasoning
    # down to "minimal" because 3-way sentiment needs no deliberation — that
    # also keeps the (hidden, billed) reasoning-token spend near zero.
    # To switch back to a classic chat model (e.g. gpt-4o-mini), set
    # temperature=0.0 and reasoning_effort=None.
    model: str = "gpt-5-mini"
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = "minimal"
    max_concurrency: int = 10
    max_retries: int = 3
    request_timeout_s: float = 30.0
    dataset_path: Path = field(default_factory=lambda: DATA_DIR / "sentiment_eval_dataset.csv")
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    @property
    def model_spec(self) -> ModelSpec:
        return resolve_model(self.model)

    @property
    def is_reasoning_model(self) -> bool:
        return self.model_spec.reasoning

    def request_overrides(self) -> dict:
        """Per-family request params for this config's primary model."""
        return request_overrides_for(self.model, self.temperature, self.reasoning_effort)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dataset_path"] = str(self.dataset_path)
        d["results_dir"] = str(self.results_dir)
        return d
