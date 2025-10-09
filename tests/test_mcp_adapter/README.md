# MCP Adapter Tests

This directory contains comprehensive tests for the GEPA MCP Adapter.

## Test Structure

```
tests/test_mcp_adapter/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures
├── test_mcp_types.py        # Type definition tests
├── test_mcp_adapter.py      # Main adapter tests
├── test_mcp_integration.py  # Integration tests
└── README.md                # This file
```

## Test Categories

### Unit Tests

**test_mcp_types.py** (~140 lines)
- Tests for MCPDataInst type
- Tests for MCPOutput type
- Tests for MCPTrajectory type
- Validates all required fields

**test_mcp_adapter.py** (~450 lines)
- Adapter initialization
- Helper methods (build_system_prompt, find_tool, etc.)
- Evaluation flow (mocked MCP calls)
- Reflective dataset generation
- Error handling
- Two-pass workflow configuration

### Integration Tests

**test_mcp_integration.py** (~150 lines)
- Import tests
- Type compatibility tests
- Real MCP server tests (skipped by default)

## Running Tests

### Run All Tests

```bash
# From repo root
pytest tests/test_mcp_adapter/ -v
```

### Run Specific Test Files

```bash
# Type tests only
pytest tests/test_mcp_adapter/test_mcp_types.py -v

# Adapter tests only
pytest tests/test_mcp_adapter/test_mcp_adapter.py -v

# Integration tests only
pytest tests/test_mcp_adapter/test_mcp_integration.py -v
```

### Run Specific Test Classes

```bash
# Test initialization only
pytest tests/test_mcp_adapter/test_mcp_adapter.py::TestMCPAdapterInitialization -v

# Test evaluation only
pytest tests/test_mcp_adapter/test_mcp_adapter.py::TestMCPAdapterEvaluation -v
```

### Run with Coverage

```bash
pytest tests/test_mcp_adapter/ --cov=src/gepa/adapters/mcp_adapter --cov-report=html
```

## Test Dependencies

### Required

- `pytest` - Test framework
- `pytest-asyncio` - For async test support (if needed)
- `gepa` - The main package

### Optional (for integration tests)

- `mcp` - MCP Python SDK
- `litellm` - For model integration
- `npx/node` - For MCP filesystem server

## Mocking Strategy

Tests use `unittest.mock` to avoid:
- Spawning real MCP server processes
- Making actual LLM API calls
- File system operations

Key mocked components:
- `stdio_client` - MCP client context manager
- `ClientSession` - MCP session with tool calls
- `litellm.completion` - LLM completions
- Model callables - Custom model functions

## Test Fixtures

Defined in `conftest.py`:

- `sample_dataset` - Sample MCPDataInst items
- `seed_candidate` - Basic candidate with tool description
- `simple_metric` - Simple scoring function
- `mock_mcp_tool` - Mocked MCP Tool object
- `mock_mcp_session` - Mocked ClientSession
- `mock_stdio_client` - Mocked stdio client
- `mock_model_callable` - Mocked model function
- `server_params` - Sample server parameters

## Writing New Tests

### Pattern for Unit Tests

```python
def test_your_feature(server_params, simple_metric):
    """Test description."""
    adapter = MCPAdapter(
        server_params=server_params,
        tool_name="read_file",
        task_model="openai/gpt-4o-mini",
        metric_fn=simple_metric,
    )

    # Your test logic
    assert adapter.something == expected
```

### Pattern for Evaluation Tests

```python
@patch("gepa.adapters.mcp_adapter.mcp_adapter.stdio_client")
@patch("gepa.adapters.mcp_adapter.mcp_adapter.ClientSession")
def test_evaluation_feature(
    mock_client_session_class,
    mock_stdio_client,
    server_params,
    simple_metric,
    mock_mcp_session,
):
    # Setup mocks
    mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
    mock_client_session_class.return_value = mock_mcp_session

    # Create adapter and test
    adapter = MCPAdapter(...)
    result = adapter.evaluate(...)

    # Assertions
    assert result.something == expected
```

## Skip Integration Tests

Integration tests with real MCP servers are skipped by default:

```python
@pytest.mark.skipif(
    True,
    reason="Requires MCP server - enable manually"
)
def test_with_real_server():
    # Test code
```

To enable, change `True` to `False` or use pytest markers.

## Common Test Scenarios

### Testing Error Handling

```python
def test_handles_connection_error(mock_stdio_client, ...):
    mock_stdio_client.side_effect = Exception("Connection failed")

    result = adapter.evaluate(...)

    # Should return failure scores, not crash
    assert all(score == 0.0 for score in result.scores)
```

### Testing Reflective Dataset

```python
def test_reflective_dataset_structure():
    reflective_data = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=eval_batch,
        components_to_update=["tool_description"],
    )

    assert "tool_description" in reflective_data
    example = reflective_data["tool_description"][0]
    assert "Inputs" in example
    assert "Generated Outputs" in example
    assert "Feedback" in example
```

## Continuous Integration

These tests are designed to run in CI without requiring:
- External API keys
- Running MCP servers
- Network connectivity

All external dependencies are mocked by default.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'mcp'`:
- Tests will be skipped automatically
- To run integration tests, install: `pip install mcp`

### Async Warnings

If you see warnings about async:
```bash
pip install pytest-asyncio
```

### Mock Not Working

Ensure patch paths match the import in the module:
```python
# Patch where it's used, not where it's defined
@patch("gepa.adapters.mcp_adapter.mcp_adapter.stdio_client")
```

## Coverage Goals

Target coverage for MCP adapter:
- Type definitions: 100%
- Helper methods: 95%+
- Evaluation flow: 90%+
- Error handling: 85%+

Run coverage report:
```bash
pytest tests/test_mcp_adapter/ --cov=src/gepa/adapters/mcp_adapter --cov-report=term-missing
```

## Contributing Tests

When adding new features to MCP adapter:

1. Add unit tests in `test_mcp_adapter.py`
2. Add type tests if new types are introduced
3. Mock external dependencies (MCP SDK, models)
4. Ensure tests pass without external services
5. Document any new fixtures in conftest.py

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [GEPA contribution guide](../../CONTRIBUTING.md)
