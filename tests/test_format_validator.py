from gepa.utils import require_format, require_json_output, require_regex_match


def test_require_json_output_forwards_parsed_dict_matching_schema():
    calls = []

    @require_json_output(schema={"answer": str, "confidence": float})
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    result = evaluator({"id": 1}, '{"answer": "yes", "confidence": 0.9}')

    assert result == (1.0, {"feedback": "ok"})
    assert calls == [({"id": 1}, {"answer": "yes", "confidence": 0.9})]


def test_require_json_output_short_circuits_malformed_json():
    calls = []

    @require_json_output(schema={"answer": str})
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    score, feedback = evaluator({}, '{"answer": "yes"')

    assert score == 0.0
    assert "Malformed JSON" in feedback["feedback"]
    assert "Expecting" in feedback["feedback"]
    assert calls == []


def test_require_json_output_short_circuits_missing_schema_key():
    calls = []

    @require_json_output(schema={"answer": str, "confidence": float})
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    score, feedback = evaluator({}, '{"answer": "yes"}')

    assert score == 0.0
    assert "Schema mismatch" in feedback["feedback"]
    assert "missing required key 'confidence'" in feedback["feedback"]
    assert calls == []


def test_require_json_output_short_circuits_wrong_schema_type():
    calls = []

    @require_json_output(schema={"answer": str, "confidence": float})
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    score, feedback = evaluator({}, '{"answer": "yes", "confidence": "high"}')

    assert score == 0.0
    assert "Schema mismatch" in feedback["feedback"]
    assert "key 'confidence' expected float, got str" in feedback["feedback"]
    assert calls == []


def test_require_json_output_without_schema_accepts_any_parseable_json():
    calls = []

    @require_json_output()
    def evaluator(data, response):
        calls.append((data, response))
        return 0.5, {"feedback": "parsed"}

    result = evaluator("example", '["a", "b"]')

    assert result == (0.5, {"feedback": "parsed"})
    assert calls == [("example", ["a", "b"])]


def test_require_format_short_circuits_when_validator_returns_false():
    calls = []

    @require_format(lambda response: response.startswith("ok:"))
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    score, feedback = evaluator({}, "bad: value")

    assert score == 0.0
    assert "Format validation failed" in feedback["feedback"]
    assert calls == []


def test_require_format_short_circuits_when_validator_raises():
    calls = []

    def validator_fn(response):
        raise ValueError(f"cannot validate {response}")

    @require_format(validator_fn)
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    score, feedback = evaluator({}, "bad")

    assert score == 0.0
    assert "Format validation failed" in feedback["feedback"]
    assert "cannot validate bad" in feedback["feedback"]
    assert calls == []


def test_require_format_forwards_when_validator_returns_true():
    calls = []

    @require_format(lambda response: response.endswith("done"))
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    result = evaluator("example", "all done")

    assert result == (1.0, {"feedback": "ok"})
    assert calls == [("example", "all done")]


def test_require_regex_match_validates_full_response():
    calls = []

    @require_regex_match(r"answer: \d+")
    def evaluator(data, response):
        calls.append((data, response))
        return 1.0, {"feedback": "ok"}

    assert evaluator({}, "answer: 42") == (1.0, {"feedback": "ok"})

    score, feedback = evaluator({}, "prefix answer: 42")

    assert score == 0.0
    assert "Regex match failed" in feedback["feedback"]
    assert "answer: \\d+" in feedback["feedback"]
    assert calls == [({}, "answer: 42")]


def test_format_validators_are_public_exports():
    from gepa.utils import __all__

    assert "require_json_output" in __all__
    assert "require_format" in __all__
    assert "require_regex_match" in __all__
