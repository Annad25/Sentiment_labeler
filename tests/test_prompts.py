"""Tests for prompt loading from the prompts/ folder."""
import pytest

from sentiment_eval.prompts import load_prompt


def test_system_prompt_loads_and_has_content():
    text = load_prompt("system_prompt")
    assert text  # non-empty
    # The three classes must be present — the prompt is the contract.
    for label in ("positive", "neutral", "negative"):
        assert label in text


def test_missing_prompt_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")
