"""Load prompt text from the top-level ``prompts/`` folder.

Prompts live as plain ``.txt`` files rather than inline Python strings so they
read naturally (real line breaks, no ``\\n`` glue), can be edited without
touching code, and can be diffed/reviewed on their own.
"""
from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the contents of ``prompts/<name>.txt`` (trailing newline stripped).

    Cached so repeated lookups don't re-hit disk. Raises FileNotFoundError with
    a clear message if the prompt is missing.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
