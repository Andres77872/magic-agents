# `client`

## Purpose

Construct and yield a `MagicLLM` client instance.

## Runtime class

- `NodeClientLLM`
- model: `ClientNodeModel`

## Default output

- `handle-client-provider`

## Important behavior

- aliases `provider -> engine`, `config/credentials -> api_info`, `model_name -> model`
- resolves `{{env.NAME}}` in API info and extra data
- maps `api_key` to `private_key` when needed for MagicLLM
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
