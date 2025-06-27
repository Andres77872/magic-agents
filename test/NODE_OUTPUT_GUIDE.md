# Node Output Guide for Magic Agents

## Key Understanding: How Nodes Send Content to Users

### Only Three Node Types Can Send Visible Content

Only nodes that yield `ChatCompletionModel` objects can produce content visible to users:

1. **NodeLLM** - Yields actual LLM responses with streaming content
2. **NodeSendMessage** - Yields custom messages with extras
3. **NodeEND** - Yields empty ChatCompletionModel (no visible content)

### Nodes That DON'T Send Visible Content

These nodes only process/transform data internally:
- **NodeChat** - Yields ModelChat object
- **NodeClientLLM** - Yields MagicLLM client
- **NodeFetch** - Yields JSON response data
- **NodeInner** - Executes inner graphs
- **NodeLoop** - Yields items and aggregated results
- **NodeParser** - Yields parsed template output
- **NodeText** - Yields text strings
- **NodeUserInput** - Yields user input data

### How run_agent Works

The `run_agent` function yields dictionaries with this structure:
```python
{
    'node': 'NodeClassName',
    'content': <actual_content>
}
```

When processing results from `run_agent`:
```python
async for event in run_agent(graph):
    if isinstance(event, dict) and 'content' in event:
        content = event['content']
        node_name = event.get('node', 'Unknown')
        
        # Only ChatCompletionModel objects have choices
        if hasattr(content, 'choices') and content.choices:
            if content.choices[0].delta.content:
                # This is the visible text content
                print(content.choices[0].delta.content)
```

### Using NodeSendMessage to Display Processing Results

To make parser or other processing results visible, use `NodeSendMessage`:

1. **Connect processing node output to SendMessage**:
   ```json
   {
       "id": "parser-to-send",
       "source": "parser-node",
       "target": "send-message-node",
       "sourceHandle": "handle_parser_output",
       "targetHandle": "handle_send_extra"
   }
   ```

2. **Configure SendMessage node**:
   ```json
   {
       "id": "send-message-node",
       "type": "send_message",
       "data": {
           "json_extras": "This text appears as visible content"
       }
   }
   ```

3. **How it works**:
   - `json_extras` becomes the visible content (delta.content)
   - Input from `handle_send_extra` goes into the `extras` field
   - If input is a string, it's wrapped in `{'text': input_string}`

### Example: Parser → SendMessage Flow

```python
# Parser outputs: "PARSED: HELLO WORLD"
# SendMessage configured with json_extras: "Result:"

# Output structure:
ChatCompletionModel(
    choices=[ChoiceModel(
        delta=DeltaModel(content='Result:')  # From json_extras
    )],
    extras={'text': 'PARSED: HELLO WORLD'}  # From parser output
)
```

### NodeInner Considerations

`NodeInner` executes nested flows but has limitations:
- It expects the inner graph to yield `ChatCompletionModel` objects
- If inner nodes don't yield ChatCompletionModel, content won't be aggregated
- Solution: End inner flows with SendMessage or LLM nodes

### Best Practices

1. **For displaying processed data**: Always use SendMessage nodes
2. **For nested flows**: Ensure inner flows end with nodes that yield ChatCompletionModel
3. **For debugging**: Check `event['node']` to identify which node produced the output
4. **For extras**: Only capture extras from SendMessage nodes, as other nodes may only have metadata

### Common Patterns

1. **Display Parser Output**:
   ```
   User Input → Parser → SendMessage → End
   ```

2. **Process and Display Loop Results**:
   ```
   Data → Loop → Process Items → Aggregate → SendMessage → End
   ```

3. **Nested Flow with Visible Output**:
   ```
   Outer: Input → Inner Node → SendMessage → End
   Inner: Input → Process → SendMessage → End
   ``` 