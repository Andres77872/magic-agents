[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-agents)

# magic_agents

A lightweight and flexible orchestration library for building LLM-based agent flows.

magic_agents lets you compose **nodes** (user input, templating, HTTP fetch, LLM calls, etc.) in a directed graph
and execute them in order, streaming results back as they arrive.

## Documentation

Full architecture diagrams, node reference, and details on the compile/execute pipeline live in the
[`docs/`](docs/) folder. Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a guided tour.


## Features

- **Modular Node System**: build reusable building blocks (nodes) for common tasks.
- **Directed Graph Execution**: declaratively wire nodes with edges and runs in topological order.
- **Streaming & Async**: supports async streaming LLM outputs (via MagicLLM) and HTTP requests.
- **Templating**: Jinja2-based parser nodes for dynamic inputs.
- **HTTP Integration**: `fetch` nodes to call REST APIs and process JSON responses.
- **Extensible**: add your own custom nodes or extend existing ones.

## Installation

Install the latest development release:

```bash
git clone https://github.com/your/repo/magic_agents.git
cd magic_agents
pip install -e .
```

## Quickstart

Define your flow as a JSON-like spec with **nodes** and **edges**, then build and run:

```python
import asyncio
from magic_agents import run_agent
from magic_agents.agt_flow import build

spec = {
    "type": "chat",
    "debug": True,
    "nodes": [
        {"id": "user_input", "type": "user_input"},
        {"id": "text_welcome", "type": "text", "data": {"text": "Hello, please wait while I fetch data."}},
        {"id": "fetch_data", "type": "fetch", "data": {
            "url": "https://api.example.com/data", "method": "GET"
        }},
        {"id": "parser_output", "type": "parser", "data": {
            "text": "Received {{ results | length }} items."
        }},
        {"id": "finish", "type": "end"}
    ],
    "edges": [
        {"source": "user_input", "sourceHandle": "handle_user_message", "target": "text_welcome", "targetHandle": "handle-parser-input"},
        {"source": "text_welcome", "sourceHandle": "handle-void", "target": "fetch_data", "targetHandle": "handle_fetch_input"},
        {"source": "fetch_data", "sourceHandle": "handle-void", "target": "parser_output", "targetHandle": "handle_parser_input"},
        {"source": "parser_output", "sourceHandle": "handle-void", "target": "finish"}
    ],
    "master": "user_input"
}

graph = build(spec, message="User's initial request")

async def main():
    async for msg in run_agent(graph):
        print(msg.choices[0].delta.content, end="")

asyncio.run(main())
```

## Node Types

magic_agents provides a set of built-in node types for common steps:

| Type           | Class                | Description                                                  |
| -------------- | -------------------- | ------------------------------------------------------------ |
| `user_input`   | NodeUserInput        | Start a new chat, assign chat/thread IDs, inject user message.    |
| `text`         | NodeText             | Emit a static text string into the flow.                      |
| `parser`       | NodeParser           | Render a Jinja2 template against previous node outputs.      |
| `fetch`        | NodeFetch            | Perform an HTTP request (GET/POST) and parse JSON result.     |
| `client`       | NodeClientLLM        | Configure and provide a MagicLLM client instance.            |
| `llm`          | NodeLLM              | Invoke an LLM (streaming or batch), optional JSON output; supports `iterate` to re-run per Loop iteration. |
| `chat`         | NodeChat             | Memory-enabled chat interface (system + user messages).       |
| `send_message` | NodeSendMessage      | Send extra JSON payloads (via ChatCompletionModel.extras).    |
| `end`          | NodeEND              | Terminal node to finalize output or drop into void.           |
| `void`         | NodeEND (internal)   | Internal drop node for unhandled outputs.                    |
| `loop`         | NodeLoop             | Iterate over a list and aggregate per-item results.          |
| `inner`        | NodeInner            | Execute a nested agent flow graph (`magic_flow`) and stream its outputs. |

----

### Node Details

#### `user_input` (`NodeUserInput`)
Injects the initial user message and initializes `chat_log.id_chat` and `id_thread`.

**Example usage:**
```json
{
  "id": "user_input",
  "type": "user_input"
}
```

**What it does:**
- Initializes a new chat session with unique IDs
- Captures the user's initial message
- Passes the message text to downstream nodes via `handle_user_message` output
- Passes any attached files via `handle_user_files` output
- Passes any attached images via `handle_user_images` output

```python
class NodeUserInput(Node):
    HANDLER_USER_MESSAGE = 'handle_user_message'
    HANDLER_USER_FILES = 'handle_user_files'
    HANDLER_USER_IMAGES = 'handle_user_images'
    ...
    async def process(self, chat_log):
        if not chat_log.id_chat: ...
        if not chat_log.id_thread: ...
        yield self.yield_static(self._text, content_type=self.HANDLER_USER_MESSAGE)
        yield self.yield_static(self.files, content_type=self.HANDLER_USER_FILES)
        yield self.yield_static(self.images, content_type=self.HANDLER_USER_IMAGES)
```

#### `text` (`NodeText`)
Emits a static string into the flow.

**Example usage:**
```json
{
  "id": "welcome_text",
  "type": "text",
  "data": {
    "text": "Welcome! I'm processing your request..."
  }
}
```

**What it does:**
- Outputs a predefined static text message
- Useful for status updates or fixed responses

```python
class NodeText(Node):
    ...
    async def process(self, chat_log):
        yield self.yield_static(self._text)
```

#### `parser` (`NodeParser`)
Renders a Jinja2 template using all inputs received so far (`self.inputs`).

**Example usage:**
```json
{
  "id": "format_results",
  "type": "parser", 
  "data": {
    "text": "Found {{ handle_parser_input.results | length }} results for query: {{ handle_parser_input.query }}"
  }
}
```

**What it does:**
- Dynamically generates text using Jinja2 templating
- Can access any input from previous nodes
- Supports filters, conditionals, and loops

```python
class NodeParser(Node):
    ...
    async def process(self, chat_log):
        output = template_parse(template=self.text, params=self.inputs)
        yield self.yield_static(output)
```

#### `loop` (`NodeLoop`)
Iterates over a list (JSON string or Python list) via input handle `list`, emitting each element downstream and collecting per-iteration inputs on handle `loop`.

**Example usage:**
```json
{
  "id": "item_loop",
  "type": "loop"
}
```

**What it does:**
- Emits each list item as an independent content event (handle `item`).
- Aggregates any inputs received on handle `loop` into a list and emits that at the end via handle `end`.

#### `fetch` (`NodeFetch`)
Sends an HTTP request (GET/POST/etc.) with optional Jinja2 templated body or JSON, returns parsed JSON.

**Example usage:**
```json
{
  "id": "search_api",
  "type": "fetch",
  "data": {
    "url": "https://google.serper.dev/search",
    "method": "POST",
    "headers": {
      "X-API-KEY": "your-api-key",
      "Content-Type": "application/json"
    },
    "json_data": {
      "q": "{{ handle_fetch_input }}"
    }
  }
}
```

**What it does:**
- Makes HTTP requests to external APIs
- Supports templated URLs, headers, and body
- Automatically parses JSON responses
- Can handle authentication headers

```python
class NodeFetch(Node):
    ...
    async def process(self, chat_log):
        # render template on self.data or self.jsondata
        response_json = await self.fetch(...)
        yield self.yield_static(response_json)
```

#### `client` (`NodeClientLLM`)
Constructs a `MagicLLM` client from provided engine, model, API info, and extra params.

**Example usage:**
```json
{
  "id": "llm_client",
  "type": "client",
  "data": {
    "model": "gpt-4o-mini",
    "engine": "openai",
    "api_info": {
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

**What it does:**
- Creates a reusable LLM client instance
- Configures API credentials and endpoints
- Supports multiple LLM providers (OpenAI, Anthropic, etc.)

```python
class NodeClientLLM(Node):
    ...
    async def process(self, chat_log):
        yield self.yield_static(self.client)
```

#### `llm` (`NodeLLM`)
Generates LLM outputs (streamed or batch) via `MagicLLM`, optionally parsing JSON.

**Example usage:**
```json
{
  "id": "generate_response",
  "type": "llm",
  "data": {
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512,
    "json_output": false,
    "iterate": true      // re-run on each Loop iteration when inside a Loop node
  }
}
```

**What it does:**
- Invokes the LLM with configured parameters
- Supports streaming or batch responses
- Can enforce JSON output format
- Handles system prompts and user messages

```python
class NodeLLM(Node):
    ...
    async def process(self, chat_log):
        client = self.get_input('handle-client-provider', required=True)
        ...
        async for chunk in ...: yield ...
        yield self.yield_static(self.generated)
```

#### `chat` (`NodeChat`)
Provides a memory-aware chat interface. Injects system/context and user messages into `ModelChat`.

**Example usage:**
```json
{
  "id": "chat_memory",
  "type": "chat",
  "data": {
    "system": "You are a helpful assistant with access to search results."
  }
}
```

**What it does:**
- Maintains conversation history
- Manages system prompts and context
- Formats messages for LLM consumption

```python
class NodeChat(Node):
    ...
    async def process(self, chat_log):
        if c := self.get_input('handle_messages'): ...
        yield self.yield_static(self.chat)
```

#### `send_message` (`NodeSendMessage`)
Sends extra JSON payloads back to the client via `ChatCompletionModel.extras`.

**Example usage:**
```json
{
  "id": "send_references",
  "type": "send_message",
  "data": {
    "json_extras": "{{ handle_send_extra }}"
  }
}
```

**What it does:**
- Sends additional metadata alongside the main response
- Useful for passing search results, references, or UI data
- Preserves structured data that shouldn't be in the text response

```python
class NodeSendMessage(Node):
    ...
    async def process(self, chat_log):
        output = self.get_input('handle_send_extra')
        ...
        yield self.yield_static(ChatCompletionModel(..., extras=output), content_type='content')
```

#### `inner` (`NodeInner`)
Runs a **nested agent flow** (`magic_flow`). This allows reusable sub-graphs and modular flows.

**Example usage:**
```json
{
  "id": "summarize_each",
  "type": "inner",
  "magic_flow": {
    "type": "chat",
    "nodes": [
      {"id": "inner_user", "type": "user_input"},
      {"id": "inner_llm", "type": "llm"},
      {"id": "inner_end", "type": "end"}
    ],
    "edges": [
      {"source": "inner_user", "target": "inner_llm"},
      {"source": "inner_llm", "target": "inner_end"}
    ],
    "master": "inner_user"
  }
}
```

**What it does:**
- Receives inputs on handle `input` and forwards them as the `message` for the sub-flow’s `user_input` node.
- Streams all outputs from the nested flow downstream via handle `loop`/`output`.
- Useful for factoring complex flows into smaller reusable pieces.

```python
class NodeInner(Node):
    inner_graph: AgentFlowModel
    ...
    async def process(self, chat_log):
        async for chunk in run_agent(self.inner_graph):
            yield chunk
```


    ...
    async def process(self, chat_log):
        output = self.get_input('handle_send_extra')
        ...
        yield self.yield_static(ChatCompletionModel(..., extras=output), content_type='content')
```

#### `end` / `void` (`NodeEND`)
Terminal node that emits a blank `ChatCompletionModel` to close the flow.

**Example usage:**
```json
{
  "id": "finish",
  "type": "end"
}
```

**What it does:**
- Marks the end of the flow execution
- Ensures all streams are properly closed
- Required for proper flow termination

```python
class NodeEND(Node):
    ...
    async def process(self, chat_log):
        yield self.yield_static(ChatCompletionModel(id='', model='', choices=[ChoiceModel()]))
```

## Detailed Example: Building a Search-Enhanced Agent

This example demonstrates a complete agent flow that:
1. Analyzes user queries to determine if they need web search
2. Rewrites queries for better search results
3. Fetches search results from an external API
4. Formats results and extracts references
5. Generates a comprehensive response using the search data

### Flow Architecture

```
User Input ──┬──> Query Rewriter ──> Search API ──> Format Results ──┬──> Final LLM ──> End
             │                                                        │
             └────────────────────────────────────────────────────────┘
```

### Complete Flow Specification

```python
import asyncio
from magic_agents import run_agent
from magic_agents.agt_flow import build

# Define the complex agent flow
search_agent_spec = {
    "type": "chat",
    "debug": True,
    "nodes": [
        # 1. User Input Node - Entry point
        {
            "id": "user_input",
            "type": "user_input"
        },
        
        # 2. LLM Client Configuration
        {
            "id": "llm-client",
            "type": "client",
            "data": {
                "model": "gpt-4o-mini",
                "engine": "openai",
                "api_info": {
                    "api_key": "sk-...",
                    "base_url": "https://api.openai.com/v1"
                }
            }
        },
        
        # 3. Query Analysis and Rewriting
        {
            "id": "system-prompt-rewrite",
            "type": "parser",
            "data": {
                "text": """
You are a query rewrite assistant for a search engine. Analyze the user query:
<user_query>
{{ handle_parser_input }}
</user_query>

If the query requires browsing for information, rewrite it to optimize for search.
Output JSON format:
{"query": "[rewritten query]"} or {"query": ""} for non-search queries.
"""
            }
        },
        
        # 4. LLM Node for Query Rewriting
        {
            "id": "llm-rewrite",
            "type": "llm",
            "data": {
                "stream": false,
                "json_output": true,
                "temperature": 0.7,
                "max_tokens": 512
            }
        },
        
        # 5. Extract Rewritten Query
        {
            "id": "parser-browsing-rewrite",
            "type": "parser",
            "data": {
                "text": "{{ handle_parser_input.query }}"
            }
        },
        
        # 6. Search API Call
        {
            "id": "fetch",
            "type": "fetch",
            "data": {
                "url": "https://google.serper.dev/search",
                "method": "POST",
                "headers": {
                    "X-API-KEY": "your-serper-api-key",
                    "Content-Type": "application/json"
                },
                "json_data": {
                    "q": "{{ handle_fetch_input }}"
                }
            }
        },
        
        # 7. Format Search Results for LLM
        {
            "id": "parser-browsing-response",
            "type": "parser",
            "data": {
                "text": """<search_results>
{% for item in handle_parser_input.organic %}
<result>
<title>{{ item.title }}</title>
<link>{{ item.link }}</link>
<snippet>{{ item.snippet }}</snippet>
{% if item.date %}<date>{{ item.date }}</date>{% endif %}
</result>
{% endfor %}
</search_results>"""
            }
        },
        
        # 8. Extract References for UI
        {
            "id": "parser-browsing-references",
            "type": "parser",
            "data": {
                "text": """{
"results_ref": [
{% for x in handle_parser_input.organic %}
{
    "title": {{ x.title | tojson }},
    "snippet": {{ x.snippet | tojson }},
    "link": "{{ x.link }}",
    "position": {{ loop.index0 }}
}{% if not loop.last %},{% endif %}
{% endfor %}
]}"""
            }
        },
        
        # 9. System Prompt for Final Response
        {
            "id": "system-prompt",
            "type": "parser",
            "data": {
                "text": "Using the following search results:\n{{ handle_parser_input }}\n\nProvide a comprehensive answer to the user's question."
            }
        },
        
        # 10. Final LLM Response Generation
        {
            "id": "llm-final",
            "type": "llm",
            "data": {
                "stream": true,
                "temperature": 0.7,
                "max_tokens": 512
            }
        },
        
        # 11. Send References as Extra Data
        {
            "id": "send-references",
            "type": "send_message",
            "data": {
                "json_extras": "{{ handle_send_extra }}"
            }
        },
        
        # 12. End Node
        {
            "id": "finish",
            "type": "end"
        }
    ],
    
    # Define the flow connections
    "edges": [
        # User input flows to rewrite prompt and final LLM
        {
            "source": "user_input",
            "sourceHandle": "handle_user_message",
            "target": "system-prompt-rewrite",
            "targetHandle": "handle_parser_input"
        },
        {
            "source": "user_input",
            "sourceHandle": "handle_user_message", 
            "target": "llm-final",
            "targetHandle": "handle_user_message"
        },
        
        # LLM client connects to both LLM nodes
        {
            "source": "llm-client",
            "sourceHandle": "handle-client-provider",
            "target": "llm-rewrite",
            "targetHandle": "handle-client-provider"
        },
        {
            "source": "llm-client",
            "sourceHandle": "handle-client-provider",
            "target": "llm-final", 
            "targetHandle": "handle-client-provider"
        },
        
        # Query rewriting flow
        {
            "source": "system-prompt-rewrite",
            "sourceHandle": "handle_parser_output",
            "target": "llm-rewrite",
            "targetHandle": "handle_user_message"
        },
        {
            "source": "llm-rewrite",
            "sourceHandle": "handle_generated_content",
            "target": "parser-browsing-rewrite",
            "targetHandle": "handle_parser_input"
        },
        
        # Search and results processing
        {
            "source": "parser-browsing-rewrite",
            "sourceHandle": "handle_parser_output",
            "target": "fetch",
            "targetHandle": "handle_fetch_input"
        },
        {
            "source": "fetch",
            "sourceHandle": "handle_response_json",
            "target": "parser-browsing-response",
            "targetHandle": "handle_parser_input"
        },
        {
            "source": "fetch",
            "sourceHandle": "handle_response_json",
            "target": "parser-browsing-references",
            "targetHandle": "handle_parser_input"
        },
        
        # Final response generation
        {
            "source": "parser-browsing-response",
            "sourceHandle": "handle_parser_output",
            "target": "system-prompt",
            "targetHandle": "handle_parser_input"
        },
        {
            "source": "system-prompt",
            "sourceHandle": "handle_parser_output",
            "target": "llm-final",
            "targetHandle": "handle-system-context"
        },
        
        # Send references and finish
        {
            "source": "parser-browsing-references",
            "sourceHandle": "handle_parser_output",
            "target": "send-references",
            "targetHandle": "handle_send_extra"
        },
        {
            "source": "send-references",
            "sourceHandle": "handle_generated_end",
            "target": "finish",
            "targetHandle": "handle_generated_end"
        },
        {
            "source": "llm-final",
            "sourceHandle": "handle_generated_end",
            "target": "finish",
            "targetHandle": "handle_generated_end"
        }
    ],
    
    "master": "user_input"
}

# Build and run the agent
async def main():
    # Create the agent with a user message
    graph = build(search_agent_spec, message="What are the latest developments in quantum computing?")
    
    # Execute and stream results
    async for msg in run_agent(graph):
        if msg.choices[0].delta.content:
            print(msg.choices[0].delta.content, end="")
        
        # Check for extra data (references)
        if hasattr(msg, 'extras') and msg.extras:
            print("\n\nReferences:", msg.extras)

# Run the agent
asyncio.run(main())
```

### How This Flow Works

1. **User Query Analysis**: The flow starts by analyzing whether the user's query needs web search
2. **Smart Query Rewriting**: If search is needed, the query is rewritten for optimal search results
3. **External API Integration**: The rewritten query is sent to a search API (Serper)
4. **Parallel Processing**: Search results are processed in parallel to:
   - Format them for the LLM context
   - Extract references for the UI
5. **Context-Aware Response**: The final LLM generates a response using the search results
6. **Metadata Preservation**: References are sent as structured data alongside the text response

### Key Benefits of This Architecture

- **Conditional Logic**: Only performs searches when needed
- **Parallel Processing**: Extracts references while formatting search results
- **Clean Separation**: UI data (references) kept separate from text response
- **Streaming Support**: Final response streams to the user in real-time
- **Error Resilience**: Each node can handle failures gracefully

## Building & Running the Flow

Use `build` to construct a type-safe `AgentFlowModel`, then `run_agent` to execute:

```python
from magic_agents.agt_flow import build, run_agent

graph = build(spec, message="Hi")
async for chunk in run_agent(graph):
    print(chunk.choices[0].delta.content, end="")
```

## Advanced Usage Tips

### Dynamic Node Configuration
Nodes can be configured dynamically using Jinja2 templates in their data fields:

```json
{
  "id": "dynamic_fetch",
  "type": "fetch",
  "data": {
    "url": "https://api.example.com/{{ handle_parser_input.endpoint }}",
    "headers": {
      "Authorization": "Bearer {{ handle_parser_input.token }}"
    }
  }
}
```

### Chaining Multiple LLMs
You can chain multiple LLM calls for complex reasoning:

```json
{
  "edges": [
    {
      "source": "llm_analyzer",
      "sourceHandle": "handle_generated_content",
      "target": "llm_synthesizer",
      "targetHandle": "handle_user_message"
    }
  ]
}
```

### Error Handling with Parser Nodes
Use parser nodes to handle errors gracefully:

```json
{
  "id": "error_handler",
  "type": "parser",
  "data": {
    "text": "{% if handle_parser_input.error %}Error: {{ handle_parser_input.error }}{% else %}Success{% endif %}"
  }
}
```

## Limitations

- **No conditional branches**: flows are strictly linear/topological.
- **Minimal error handling**: HTTP and LLM errors bubble up.
- **Basic templating**: only Jinja2 text rendering, no complex data transforms.
- **Synchronous graph build**: building the flow is not async.
- **In-memory**: no built-in persistence or caching of intermediate results.

## Future Work

- Add conditional/looping nodes for dynamic branching.
- Rich memory store integrations (vector DBs, Redis).
- Built‑in error retry and backoff strategies.
- Tool invocation nodes (e.g., files, databases, shell).
- Graph visualization CLI/UI.

## Known Issues & Caveats

- Graph cycles raise `ValueError` and abort execution.
- Jinja template errors can break the flow at runtime.
- Large HTTP responses or model streams may need backpressure.
- Node ordering relies on correct `edges` configuration.
