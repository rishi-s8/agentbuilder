# AgentBuilder Test Suite

This directory contains comprehensive unit tests for the agentbuilder package.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and test configuration
├── test_action.py           # Tests for Action classes
├── test_client.py           # Tests for Client classes (BaseConversationWrapper, ConversationWrapper)
├── test_tools.py            # Tests for Tool, Response, and tool_from_function
├── test_planner.py          # Tests for AgenticPlanner
├── test_loop.py             # Tests for AgenticLoop
├── test_utils.py            # Tests for utility functions (create_agent)
└── test_error_scenarios.py  # Tests for error handling and failure scenarios
```

## Running Tests

### Install Development Dependencies

First, install the package with development dependencies:

```bash
cd agentbuilder
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage Report

```bash
pytest --cov=agentbuilder --cov-report=html --cov-report=term-missing
```

The HTML coverage report will be generated in `htmlcov/index.html`.

### Run Specific Test Files

```bash
pytest tests/test_action.py
pytest tests/test_client.py
pytest tests/test_tools.py
pytest tests/test_error_scenarios.py
```

### Run Specific Test Classes or Functions

```bash
pytest tests/test_action.py::TestExecuteToolsAction
pytest tests/test_tools.py::TestTool::test_execute_success
```

### Run Tests in Verbose Mode

```bash
pytest -v
```

### Run Tests and Stop at First Failure

```bash
pytest -x
```

## Test Coverage

The test suite aims for high coverage of the agentbuilder package:

- **Action classes**: Tests for all action types including ExecuteToolsAction, MakeLLMRequestAction, CompleteAction, EmptyAction, and message actions
- **Client classes**: Tests for BaseConversationWrapper and ConversationWrapper including conversation management and OpenAI API interactions
- **Tool classes**: Tests for Tool, Response, and tool_from_function including Pydantic model integration
- **Planner**: Tests for AgenticPlanner decision logic and conversation flow
- **Loop**: Tests for AgenticLoop execution including iteration limits and completion handling
- **Utils**: Tests for the create_agent utility function
- **Error Scenarios**: Tests for error handling including OpenAI API errors (rate limits, authentication, connection), malformed responses, tool execution errors, loop error recovery, and conversation load/save errors

## Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_openai_client`: Mock OpenAI client for testing API interactions
- `sample_tool`: A simple addition tool for testing
- `sample_tool_map`: A tool map with the sample tool
- `base_conversation`: BaseConversationWrapper instance
- `mock_conversation_wrapper`: Mock conversation wrapper with history

## Writing New Tests

When adding new features to agentbuilder, please add corresponding tests:

1. Create test functions with descriptive names prefixed with `test_`
2. Use fixtures from `conftest.py` when appropriate
3. Mock external dependencies (OpenAI API calls, file I/O, etc.)
4. Test both success and failure cases
5. Test edge cases and error handling
6. Add docstrings to test functions explaining what they test

Example:

```python
def test_new_feature(mock_conversation_wrapper, sample_tool_map):
    """Test that new feature works correctly."""
    # Arrange
    expected = "result"
    
    # Act
    result = new_feature(mock_conversation_wrapper, sample_tool_map)
    
    # Assert
    assert result == expected
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The pytest configuration in `pyproject.toml` and `pytest.ini` ensures consistent test execution across environments.
