# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.lm import LMOutput

pytest.importorskip("dspy")

from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter  # noqa: E402


def test_stripped_lm_call_accepts_string_subclasses_without_losing_metadata():
    adapter = object.__new__(DspyAdapter)
    output = LMOutput("partial output", finish_reason="length")
    adapter.reflection_lm = lambda _prompt: [output]

    normalized = adapter.stripped_lm_call("prompt")

    assert normalized == [output]
    assert normalized[0] is output
    assert normalized[0].finish_reason == "length"
