# Node Compatibility Examples

Practical examples demonstrating node connections and compatibility patterns.

## Example 1: Basic Chatbot

**Use Case**: Simple chatbot with user input and LLM response

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "What is the capital of France?"
      }
    },
    {
      "id": "client-1", 
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4",
        "api_info": {"api_key": "sk-..."}
      }
    },
    {
      "id": "chat-1",
      "type": "CHAT",
      "data": {
        "message": "",
        "memory": {"stm": 5}
      }
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "stream": true,
        "json_output": false,
        "extra_data": {"temperature": 0.7}
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "chat-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "chat-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-chat"
    },
    {
      "source": "llm-1",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

## Example 2: Chatbot with System Prompt

**Use Case**: Chatbot with custom system prompt for specialized behavior

```json
{
  "nodes": [
    {
      "id": "system-1",
      "type": "TEXT",
      "data": {
        "text": "You are a helpful Python programming assistant. Always provide code examples."
      }
    },
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "How do I read a file in Python?"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "chat-1",
      "type": "CHAT"
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {"stream": true}
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "system-1",
      "sourceHandle": "default",
      "target": "chat-1",
      "targetHandle": "handle-system-context"
    },
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "chat-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "chat-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-chat"
    },
    {
      "source": "llm-1",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

## Example 3: Age Verification with Conditional Branching

**Use Case**: Route users to different content based on age

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "{\"name\": \"John\", \"age\": 25}"
      }
    },
    {
      "id": "conditional-1",
      "type": "CONDITIONAL",
      "data": {
        "condition": "{{ 'adult' if age >= 18 else 'minor' }}",
        "merge_strategy": "flat"
      }
    },
    {
      "id": "text-adult",
      "type": "TEXT",
      "data": {
        "text": "Welcome! You have access to all features."
      }
    },
    {
      "id": "text-minor",
      "type": "TEXT",
      "data": {
        "text": "Welcome! You have access to age-appropriate content only."
      }
    },
    {
      "id": "end-adult",
      "type": "END"
    },
    {
      "id": "end-minor",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "conditional-1",
      "targetHandle": "handle_input"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "adult",
      "target": "text-adult",
      "targetHandle": "default"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "minor",
      "target": "text-minor",
      "targetHandle": "default"
    },
    {
      "source": "text-adult",
      "sourceHandle": "default",
      "target": "end-adult",
      "targetHandle": "default"
    },
    {
      "source": "text-minor",
      "sourceHandle": "default",
      "target": "end-minor",
      "targetHandle": "default"
    }
  ]
}
```

## Example 4: API Data Fetching with LLM Analysis

**Use Case**: Fetch weather data and analyze it with LLM

**Features Demonstrated**:
- NodeFetch URL templating with query parameters
- API response processing with NodeParser
- LLM analysis of external data

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "{\"city\": \"Paris\"}"
      }
    },
    {
      "id": "fetch-1",
      "type": "FETCH",
      "data": {
        "method": "GET",
        "url": "https://api.weather.com/v1/weather?city={{ city }}",
        "headers": {
          "Accept": "application/json"
        }
      }
    },
    {
      "id": "parser-1",
      "type": "PARSER",
      "data": {
        "text": "Analyze this weather data: {{ weather_data }}"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4",
        "api_info": {"api_key": "sk-..."}
      }
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "stream": true
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "fetch-1",
      "targetHandle": "city"
    },
    {
      "source": "fetch-1",
      "sourceHandle": "default",
      "target": "parser-1",
      "targetHandle": "weather_data"
    },
    {
      "source": "parser-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-1",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

**Note**: NodeFetch URL templating now supports Jinja2 templates for dynamic query parameters and path segments. The `city` variable comes from the `targetHandle` name in the edge configuration.

## Example 5: Loop Processing - Summarize Multiple Items

**Use Case**: Process each item in a list with LLM

**Features Demonstrated**:
- NodeLoop iteration over a list
- NodeLLM with `iterate=true` flag for re-execution per item
- Loop aggregation and result collection

```json
{
  "nodes": [
    {
      "id": "text-1",
      "type": "TEXT",
      "data": {
        "text": "[\"Article 1: AI trends\", \"Article 2: Climate change\", \"Article 3: Space exploration\"]"
      }
    },
    {
      "id": "loop-1",
      "type": "LOOP"
    },
    {
      "id": "parser-1",
      "type": "PARSER",
      "data": {
        "text": "Summarize this article in one sentence: {{ item }}"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "stream": false,
        "iterate": true,
        "json_output": false
      }
    },
    {
      "id": "parser-2",
      "type": "PARSER",
      "data": {
        "text": "All summaries: {{ results }}"
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "text-1",
      "sourceHandle": "default",
      "target": "loop-1",
      "targetHandle": "handle_list"
    },
    {
      "source": "loop-1",
      "sourceHandle": "content",
      "target": "parser-1",
      "targetHandle": "item"
    },
    {
      "source": "parser-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-1",
      "sourceHandle": "default",
      "target": "loop-1",
      "targetHandle": "handle_loop"
    },
    {
      "source": "loop-1",
      "sourceHandle": "default",
      "target": "parser-2",
      "targetHandle": "results"
    },
    {
      "source": "parser-2",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

**Important Notes**:
- **NodeLoop outputs**: Uses `sourceHandle="content"` for items (not `handle_item`) and `sourceHandle="default"` for aggregation (not `handle_end`)
- **iterate flag**: Must be set to `true` on NodeLLM to force re-execution for each loop item
- **Loop feedback**: LLM results connect back to loop's `handle_loop` input for aggregation

## Example 6: Multi-Stage LLM Pipeline

**Use Case**: Generate query → Fetch data → Analyze results

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "Find information about renewable energy"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "llm-query",
      "type": "LLM",
      "data": {
        "stream": false,
        "json_output": true,
        "extra_data": {"temperature": 0.3}
      }
    },
    {
      "id": "text-query-prompt",
      "type": "TEXT",
      "data": {
        "text": "Generate a search query JSON with 'query' field for: "
      }
    },
    {
      "id": "parser-query",
      "type": "PARSER",
      "data": {
        "text": "{{ prompt }}{{ user_input }}"
      }
    },
    {
      "id": "fetch-1",
      "type": "FETCH",
      "data": {
        "method": "GET",
        "url": "https://api.search.com/search?q={{ query }}"
      }
    },
    {
      "id": "client-2",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "llm-analyze",
      "type": "LLM",
      "data": {
        "stream": true
      }
    },
    {
      "id": "parser-analyze",
      "type": "PARSER",
      "data": {
        "text": "Analyze and summarize: {{ search_results }}"
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "text-query-prompt",
      "sourceHandle": "default",
      "target": "parser-query",
      "targetHandle": "prompt"
    },
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "parser-query",
      "targetHandle": "user_input"
    },
    {
      "source": "parser-query",
      "sourceHandle": "default",
      "target": "llm-query",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-query",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-query",
      "sourceHandle": "default",
      "target": "fetch-1",
      "targetHandle": "query_input"
    },
    {
      "source": "fetch-1",
      "sourceHandle": "default",
      "target": "parser-analyze",
      "targetHandle": "search_results"
    },
    {
      "source": "parser-analyze",
      "sourceHandle": "default",
      "target": "llm-analyze",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-2",
      "sourceHandle": "default",
      "target": "llm-analyze",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-analyze",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

## Example 7: Complex Conditional with Multiple Inputs

**Use Case**: Approve/deny based on multiple data sources

```json
{
  "nodes": [
    {
      "id": "fetch-user",
      "type": "FETCH",
      "data": {
        "method": "GET",
        "url": "https://api.example.com/users/123"
      }
    },
    {
      "id": "fetch-account",
      "type": "FETCH",
      "data": {
        "method": "GET",
        "url": "https://api.example.com/accounts/123"
      }
    },
    {
      "id": "conditional-1",
      "type": "CONDITIONAL",
      "data": {
        "condition": "{{ 'approved' if handle_input_1.age >= 18 and handle_input_2.balance > 1000 else 'denied' }}",
        "merge_strategy": "namespaced"
      }
    },
    {
      "id": "text-approved",
      "type": "TEXT",
      "data": {
        "text": "Application approved!"
      }
    },
    {
      "id": "text-denied",
      "type": "TEXT",
      "data": {
        "text": "Application denied. Requirements not met."
      }
    },
    {
      "id": "end-approved",
      "type": "END"
    },
    {
      "id": "end-denied",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "fetch-user",
      "sourceHandle": "default",
      "target": "conditional-1",
      "targetHandle": "handle_input_1"
    },
    {
      "source": "fetch-account",
      "sourceHandle": "default",
      "target": "conditional-1",
      "targetHandle": "handle_input_2"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "approved",
      "target": "text-approved",
      "targetHandle": "default"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "denied",
      "target": "text-denied",
      "targetHandle": "default"
    },
    {
      "source": "text-approved",
      "sourceHandle": "default",
      "target": "end-approved",
      "targetHandle": "default"
    },
    {
      "source": "text-denied",
      "sourceHandle": "default",
      "target": "end-denied",
      "targetHandle": "default"
    }
  ]
}
```

## Example 8: Classification with Switch Pattern

**Use Case**: Route to different handlers based on category

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "I need help with billing"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "llm-classify",
      "type": "LLM",
      "data": {
        "stream": false,
        "json_output": true,
        "extra_data": {"temperature": 0.1}
      }
    },
    {
      "id": "text-classify-prompt",
      "type": "TEXT",
      "data": {
        "text": "Classify this request into one of: billing, technical, sales. Return JSON with 'category' field."
      }
    },
    {
      "id": "parser-1",
      "type": "PARSER",
      "data": {
        "text": "{{ prompt }}\n\nRequest: {{ user_message }}"
      }
    },
    {
      "id": "conditional-1",
      "type": "CONDITIONAL",
      "data": {
        "condition": "{{ category }}",
        "merge_strategy": "flat"
      }
    },
    {
      "id": "text-billing",
      "type": "TEXT",
      "data": {
        "text": "Routing to billing department..."
      }
    },
    {
      "id": "text-technical",
      "type": "TEXT",
      "data": {
        "text": "Routing to technical support..."
      }
    },
    {
      "id": "text-sales",
      "type": "TEXT",
      "data": {
        "text": "Routing to sales team..."
      }
    },
    {
      "id": "end-1",
      "type": "END"
    },
    {
      "id": "end-2",
      "type": "END"
    },
    {
      "id": "end-3",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "text-classify-prompt",
      "sourceHandle": "default",
      "target": "parser-1",
      "targetHandle": "prompt"
    },
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "parser-1",
      "targetHandle": "user_message"
    },
    {
      "source": "parser-1",
      "sourceHandle": "default",
      "target": "llm-classify",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-classify",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-classify",
      "sourceHandle": "default",
      "target": "conditional-1",
      "targetHandle": "handle_input"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "billing",
      "target": "text-billing",
      "targetHandle": "default"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "technical",
      "target": "text-technical",
      "targetHandle": "default"
    },
    {
      "source": "conditional-1",
      "sourceHandle": "sales",
      "target": "text-sales",
      "targetHandle": "default"
    },
    {
      "source": "text-billing",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    },
    {
      "source": "text-technical",
      "sourceHandle": "default",
      "target": "end-2",
      "targetHandle": "default"
    },
    {
      "source": "text-sales",
      "sourceHandle": "default",
      "target": "end-3",
      "targetHandle": "default"
    }
  ]
}
```

## Example 9: Send Message with Extras

**Use Case**: Send LLM response with additional metadata

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "Generate a product recommendation"
      }
    },
    {
      "id": "client-1",
      "type": "CLIENT_LLM",
      "data": {
        "engine": "openai",
        "model": "gpt-4"
      }
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "stream": false,
        "json_output": true
      }
    },
    {
      "id": "send-message-1",
      "type": "SEND_MESSAGE",
      "data": {
        "message": "",
        "json_extras": "recommendation"
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "llm-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "client-1",
      "sourceHandle": "default",
      "target": "llm-1",
      "targetHandle": "handle-client-provider"
    },
    {
      "source": "llm-1",
      "sourceHandle": "default",
      "target": "send-message-1",
      "targetHandle": "handle_send_extra"
    },
    {
      "source": "send-message-1",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

## Example 10: Inner Flow (Nested Agent)

**Use Case**: Execute a specialized nested flow for complex processing

```json
{
  "nodes": [
    {
      "id": "user-1",
      "type": "USER_INPUT",
      "data": {
        "text": "Process this document"
      }
    },
    {
      "id": "inner-1",
      "type": "INNER",
      "data": {
        "magic_flow": {
          "nodes": [
            {
              "id": "inner-user",
              "type": "USER_INPUT"
            },
            {
              "id": "inner-client",
              "type": "CLIENT_LLM",
              "data": {
                "engine": "openai",
                "model": "gpt-4"
              }
            },
            {
              "id": "inner-llm",
              "type": "LLM",
              "data": {"stream": true}
            },
            {
              "id": "inner-end",
              "type": "END"
            }
          ],
          "edges": [
            {
              "source": "inner-user",
              "sourceHandle": "handle_user_message",
              "target": "inner-llm",
              "targetHandle": "handle_user_message"
            },
            {
              "source": "inner-client",
              "sourceHandle": "default",
              "target": "inner-llm",
              "targetHandle": "handle-client-provider"
            },
            {
              "source": "inner-llm",
              "sourceHandle": "default",
              "target": "inner-end",
              "targetHandle": "default"
            }
          ]
        }
      }
    },
    {
      "id": "parser-1",
      "type": "PARSER",
      "data": {
        "text": "Inner flow completed with result: {{ inner_result }}"
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-1",
      "sourceHandle": "handle_user_message",
      "target": "inner-1",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "inner-1",
      "sourceHandle": "handle_execution_content",
      "target": "parser-1",
      "targetHandle": "inner_result"
    },
    {
      "source": "parser-1",
      "sourceHandle": "default",
      "target": "end-1",
      "targetHandle": "default"
    }
  ]
}
```

## Handle Mapping Quick Reference

### Common Connections

| From Node | From Handle | To Node | To Handle | Usage |
|-----------|-------------|---------|-----------|-------|
| NodeUserInput | `handle_user_message` | NodeChat | `handle_user_message` | User text to chat |
| NodeUserInput | `handle_user_message` | NodeLLM | `handle_user_message` | Direct user to LLM |
| NodeText | `default` | NodeChat | `handle-system-context` | System prompt |
| NodeClientLLM | `default` | NodeLLM | `handle-client-provider` | LLM client (REQUIRED) |
| NodeChat | `default` | NodeLLM | `handle-chat` | Prepared chat to LLM |
| NodeLLM | `default` | NodeParser | any | Post-process LLM output |
| NodeLLM | `default` | NodeConditional | `handle_input` | Branch on LLM output |
| NodeParser | `default` | NodeFetch | any | Templated API request |
| NodeFetch | `default` | NodeLLM | `handle_user_message` | API data to LLM |
| NodeLoop | `content` | NodeLLM | `handle_user_message` | Process each item |
| NodeLLM | `default` | NodeLoop | `handle_loop` | Aggregate loop results |
| NodeLoop | `default` | NodeParser | any | Final aggregated output |
| NodeInner | `handle_execution_content` | NodeParser | any | Inner flow results |
| Any | any | NodeEND | any | Terminal connection |

## Common Patterns Summary

1. **Basic Chat**: `UserInput → Chat → LLM → END` (+ ClientLLM)
2. **System Prompt**: `Text → Chat` + `UserInput → Chat → LLM → END`
3. **Conditional**: `Input → Conditional → [Multiple branches]`
4. **Loop**: `List → Loop → Process → back to Loop → END`
5. **API**: `Input → Parser → Fetch → LLM → END`
6. **Pipeline**: `LLM → Parser → Fetch → LLM → END`
7. **Classification**: `Input → LLM (classify) → Conditional → [Handlers]`
8. **Inner Flow**: `Input → Inner → Parser → END`

---

*For more details, see NODE_COMPATIBILITY_MATRIX.md*
