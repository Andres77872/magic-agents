# `client`

## Purpose

Construct and yield a `MagicLLM` client instance.

## Runtime class

- `NodeClientLLM`
- model: `ClientNodeModel`

## Default output

- `handle-client-provider`

## Runtime-overridable inputs

- `handle-client-engine`
- `handle-client-model`

## Important behavior

- aliases `provider -> engine`, `config/credentials -> api_info`, `model_name -> model`
- runtime input handles can override `engine` and `model` before client creation/yield
- resolves `{{env.NAME}}` in API info and extra data
- maps `api_key` to `private_key` when needed for MagicLLM
- `endpoint: "responses"` selects `/responses` for `engine: "openai"`;
  `"chat_completions"` selects `/chat/completions` (the default)
- an explicit node endpoint overrides `api_info.endpoint` and `extra_data.endpoint`;
  omitted values preserve nested configuration and existing graphs
- use Responses for reasoning with function tools, including GPT-6 Luna;
  compatible providers must implement the selected endpoint
- yields a debug configuration error instead of crashing on client init failure

## Example

```json
{
  "id": "openai-client",
  "type": "client",
  "data": {
    "engine": "openai",
    "model": "gpt-4o-mini",
    "api_info": {"api_key": "{{env.OPENAI_API_KEY}}"}
  }
}
```
