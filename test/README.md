# Magic Agents Test Suite

Comprehensive test suite for the magic-agents framework, featuring diverse flows, logic patterns, and execution scenarios.

## Overview

The test suite is organized into multiple modules, each focusing on different aspects of the agent flow system:

### 1. **test_run1.py** - Original Tests
- **Browsing Agent Test**: Query rewriting with web search integration using Serper API
- **Loop Agent Test**: Iterative processing with dynamic LLM execution

### 2. **test_comprehensive_flows.py** - Core Functionality Tests
Tests fundamental agent flow patterns:
- Simple text-to-LLM flows
- Parser template transformations
- Conditional flow routing with JSON parsing
- Nested loops with data aggregation
- Multi-stage processing pipelines
- Inner node composition
- Fetch and parse workflows
- Parallel processing branches
- Error handling flows
- Complex routing logic

### 3. **test_advanced_flows.py** - Advanced Pattern Tests
Tests complex and advanced scenarios:
- SendMessage node with extras functionality
- Deeply nested inner flows (3+ levels)
- Complex loops with conditional exits
- Parallel fetch operations with aggregation
- Dynamic flow construction based on input
- Multi-modal flows with image support
- State management across nodes
- Recursive summarization patterns

### 4. **test_edge_cases.py** - Edge Case & Error Handling Tests
Tests robustness and error scenarios:
- Empty input handling
- Circular reference prevention
- Malformed JSON handling
- Very long input truncation
- Empty loop processing
- Special character escaping
- Unicode character support
- Missing required inputs
- Nested JSON parsing
- Timeout simulation

## Test Runner

The `run_all_tests.py` script provides a unified test runner with reporting capabilities.

### Usage

```bash
# Run all tests
python test/run_all_tests.py

# Run specific test suite
python test/run_all_tests.py --suite comprehensive
python test/run_all_tests.py --suite advanced
python test/run_all_tests.py --suite edge
python test/run_all_tests.py --suite individual

# Run with verbose output
python test/run_all_tests.py --verbose
```

### Test Report

After execution, a `test_report.json` file is generated with:
- Total test count
- Pass/fail statistics
- Execution duration
- Detailed error information

## API Keys Configuration

All tests load API keys from `/home/andres/Documents/agents_key.json`:

```json
{
  "openai_key": "your-openai-api-key",
  "serper_key": "your-serper-api-key"
}
```

## Node Types Tested

The test suite covers all available node types:
- **chat**: Chat interaction nodes
- **llm**: Language model nodes with streaming support
- **text**: Static text content nodes
- **user_input**: User message input nodes
- **parser**: Jinja2 template parsing nodes
- **fetch**: HTTP request nodes
- **client**: LLM client configuration nodes
- **send_message**: Message sending with extras
- **loop**: Iterative processing nodes
- **inner**: Nested flow composition nodes
- **end**: Flow termination nodes

## Test Flow Patterns

### Basic Patterns
1. **Linear Flow**: Input → Process → Output
2. **Branching**: Conditional routing based on input
3. **Looping**: Iterative processing with aggregation
4. **Nesting**: Flows within flows using inner nodes

### Advanced Patterns
1. **Parallel Execution**: Multiple branches processed simultaneously
2. **State Management**: Passing and transforming state across nodes
3. **Error Recovery**: Graceful handling of failures
4. **Dynamic Construction**: Building flows based on runtime conditions

## Running Individual Tests

To run tests individually with pytest:

```bash
# Run all tests in a file
pytest test/test_comprehensive_flows.py -v

# Run specific test class
pytest test/test_comprehensive_flows.py::TestComprehensiveFlows -v

# Run specific test method
pytest test/test_comprehensive_flows.py::TestComprehensiveFlows::test_simple_text_to_llm_flow -v
```

## Debugging

Enable debug mode in agent definitions:
```python
agt = {
    "type": "chat",
    "debug": True,  # Enable debug logging
    # ... rest of configuration
}
```

## Contributing

When adding new tests:
1. Follow the existing naming convention: `test_<feature>_<scenario>`
2. Include docstrings explaining the test purpose
3. Add assertions to verify expected behavior
4. Handle async operations properly with `pytest.mark.asyncio`
5. Clean up any temporary resources

## Performance Considerations

- Tests use streaming where possible to minimize memory usage
- Parallel tests demonstrate efficiency gains
- Timeout tests ensure graceful handling of slow operations
- Long input tests verify truncation and handling of large data

## Future Enhancements

Potential areas for additional testing:
- WebSocket-based real-time flows
- Database integration nodes
- Authentication and security flows
- Performance benchmarking suite
- Load testing for concurrent flows 