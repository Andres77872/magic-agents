# `fetch`

## Purpose

Perform an HTTP request, or expose an HTTP request as a callable tool.

## Runtime class

- `NodeFetch`
- model: `FetchNodeModel`
- request rendering, execution and typed errors: `magic_agents/node_system/fetch_request.py`

## Default output

- `handle_fetch_output`

## Runtime-overridable inputs

- `handle-url`
- `handle-fetch-method`
- `handle-fetch-data`
- `handle-fetch-json_data`
- `handle-fetch-headers`
- `handle_fetch_input` — template context only; exposed to Jinja rendering and does not override a request field

`params`/`query` has no input handle; it always comes from the node data.

## Templates and data

Only the node's own static fields are templates: `url`, `headers`, `params`,
`data`, `json_data` in the node data, and the non-schema entries of
`tool_parameters`. `{{env.NAME}}` placeholders are resolved only there.

Everything that arrives at runtime is **data**: values on the input handles
above, a Hook's request content, the model's tool arguments, and anything an
upstream node produced (web content included). Data is inserted into the
rendered request as a value. It is never compiled as a Jinja template and
`{{env.NAME}}` inside it stays literal text. A request field that arrives on
its input handle (`handle-url`, `handle-fetch-headers`, `handle-fetch-data`,
`handle-fetch-json_data`) replaces the static field and is sent verbatim:
JSON text on `handle-fetch-headers`/`handle-fetch-json_data` is parsed as
JSON, not rendered.

All Fetch rendering (step mode, Hook-controlled mode, tool mode,
`tool_parameters`) uses one `jinja2.sandbox.SandboxedEnvironment` with the
project filters `fromjson`, `regex_replace`, `regex_findall` and `tojson`,
plus the Jinja built-ins (`urlencode`, `default`, ...). The sandbox blocks
attribute escapes (`__globals__`, `__class__`, `str.format`) with
`TemplateError`; ordinary idioms such as `{% set _ = items.append(x) %}`
work. Inputs are copied for each request, so a template that mutates a list
or dict changes only its own copy, never another node's data.

How each field is rendered:

- **URL, header values, `tool_parameters.method`**: plain text. Inserted values
  are not URL-encoded; use `{{ value | urlencode }}` for a query value. A header
  name or value that contains a line break fails with `TemplateError`.
- **Method** (node data, `handle-fetch-method` or `tool_parameters.method`):
  must be an HTTP token (RFC 9110: letters, digits and symbols such as `-`, `_`, `.`); anything else,
  such as a line break or a space, fails with `TemplateError` before the
  request is logged or sent.
- **Body fields (`json_data`, `data`, `params`)**: every string leaf, including
  nested dict values and list items, is rendered in a JSON string context and
  decoded back. `{{ value }}` inserts the exact text, so quotes, newlines,
  backslashes, braces, HTML and JSON-looking text arrive unchanged and can
  never add keys. Every expression result is escaped, Markup included
  (`| e` gives HTML-escaped text, `| safe` the exact text), and so is the
  output of `{% filter %}` and `{% call %}` blocks. Non-string leaves
  (numbers, booleans, null) are untouched; dict keys are not templated. The
  legacy idiom `{{ (value | tojson)[1:-1] }}` still produces the exact value.
  A leaf that is only `{{ value | tojson }}` holds that JSON text as a string.
- **`json_data`/`params` given as a string**: text that is valid JSON before
  rendering is parsed and rendered per leaf. Other text, such as
  `{"n": {{ count }}}`, `{"tags": {{ tags }}}` or `{"items": {{ items | tojson }}}`,
  is rendered as a whole and then parsed. Each expression result is placed by
  where it sits in the rendered JSON:
  - inside a JSON string literal it is escaped (exact text);
  - outside one it is inserted raw when it is exactly one JSON value:
    `tojson` output, a number, or JSON text such as `["a", "b"]` or
    `{"q": "x"}` (from an input or an `{{env.NAME}}` holding JSON);
  - anything else outside a string (`1, "admin": true`, plain words) is
    escaped and the text fails to parse.

  An inserted value therefore fills one value position and never adds a key
  or a sibling list item. Text that does not parse fails with `TemplateError`.
- **`data` given as a string** is the raw body: rendered as one leaf and sent
  as text.

## Step mode

- sends the request when at least one input is set; with no inputs set it
  yields `{}` and sends nothing
- a non-GET request without a body yields `{}` and sends nothing
- a JSON response (`Content-Type: application/json`) is decoded; any other
  content type is returned as text
- the same node sends the same request and returns the same output when a
  lifecycle Hook controls it: the Hook-controlled operation runs this
  step-mode core (the "no inputs" rule included). Only the failure surface
  differs, see below
- a Hook child call or redirect into a step Fetch always sends its request,
  also with empty content (`await context.call(edge, {})` into a static-URL
  Fetch)

### Failures

A failed step-mode Fetch raises a typed error. The node is marked failed, its
downstream nodes are bypassed at once (no input timeout), `on_node_error`
hooks and observers fire, and an `onError` lifecycle Hook can recover it. In
a Loop the failed iteration's slot aggregates `null`.

This holds for a Fetch without downstream nodes too (a best-effort
notification or webhook): its failure is a node error, so the run ends with
a graph error (`Graph execution failed: 1 node error(s)`) even though the
other branches delivered their answer. Before, such a failure was silent and
the run reported success. To keep a best-effort call from failing the run,
add an `onError` Hook that returns a success outcome (degrade).

| Error class | When |
|---|---|
| `HTTPError` | 4xx/5xx response; message `HTTP request failed with status <code>: <reason>` |
| `NetworkError` | no HTTP answer: connection refused, DNS, TLS, timeout |
| `TemplateError` | the request could not be built: URL/header/body rendering, invalid JSON text, a line break in a header |
| `UnexpectedError` | anything else, including a malformed JSON response body |

For an uncontrolled Fetch the class name is the `error_type` in executor
trace frames, debug SSE `node_error` events and observers. Both also carry a
sanitized `context`: `method`, `url` as `scheme://host[:port]/path` (no
query, fragment or user info) and `status_code`, or the failing `field` for
template errors. Messages are built from the status and reason, never from
the full URL, and response headers are never included.

Env values resolved for the request (4 characters or longer) are replaced by
`[REDACTED]` in every message, context, log line, Hook error body and
tool-mode error string, also in their encoded forms: percent-encoded as
aiohttp/yarl send them (query params, path, normalized URL), form-encoded,
and JSON-escaped. Successful response bodies are not modified.

A downstream node that has other inputs (fan-in) waits for them and runs
once without the failed Fetch's value, as with any other failed node; the
cascade bypasses only the Fetch's own edges, whatever order the inputs
finish in.

Under Hook control the failure becomes a typed outcome:

| Step error | Hook outcome `error.code` | `details` |
|---|---|---|
| `HTTPError` | `HTTP_ERROR` (429 and 5xx retryable) | `http_status`, `status_code`, `response_body` |
| `NetworkError` | `NETWORK_ERROR` (retryable) | `exception_type` |
| `TemplateError`, `UnexpectedError` | `NODE_EXCEPTION` | `exception_type` (`TemplateError`, `UnexpectedError`) |

Each of these `details` also carries `context`, the same sanitized dict as
above. When the Hook does not recover the failure, the node fails with
`OperationFailure`: that is the `error_type` in its trace frame and debug SSE
`node_error` event, the typed code is in the Hook outcome
(`HOOK_RESULT` frame), and the frame and event carry the same `context`.

The content of a child call or redirect into a step Fetch is that Fetch's
input map (its template context), so another step Fetch's
`context.request["content"]` can be replayed as is.

## Tool mode

- in `tool_mode`, the node does **not** execute immediately; it yields a
  `FetchToolCallable`
- the model's arguments are the template context; they are inserted as data
- without `tool_parameters`, the tool schema lists every `{{name}}`
  placeholder of the static fields, nested body values and list items
  included (the renderer fills those too); `tool_parameters` entries that are
  dicts or lists expose their nested placeholders the same way
- `tool_parameters` non-schema entries override request fields and are
  rendered exactly once; unknown keys become extra JSON body fields; a
  `headers` entry may be a dict or JSON object text
- the tool returns a string to the model: the JSON-normalized body, the text
  body, or `HTTP <code>: <reason>` plus the full response body on non-2xx
  (env values used by the request scrubbed); logs carry a redacted preview
- a non-GET call without a body returns `{"error": "No body provided for <METHOD> request"}`
- a network failure returns `{"error": "Network error: <text>"}`; an invalid
  URL is named only as `scheme://host[:port]/path` (no query, user info or
  secrets), other client errors name at most host and port

### Tool mode fields

- `tool_mode`
- `tool_name`
- `tool_description` — optional explicit description exposed in the tool schema
- `tool_parameters`

## Gotchas

- step mode has no HTTP timeout of its own; a slow endpoint holds the node
  until aiohttp's default (300 s total)
- inserted URL values are not encoded: `?q={{ handle_fetch_input }}` with
  `a&b=1` sends two parameters; write `{{ handle_fetch_input | urlencode }}`

## Example

```json
{
  "id": "fetch-user",
  "type": "fetch",
  "data": {
    "url": "https://api.example.com/users/{{ handle_fetch_input | urlencode }}",
    "method": "POST",
    "headers": {"Authorization": "Bearer {{env.API_TOKEN}}"},
    "json_data": {"query": "{{ handle_fetch_input }}", "filters": {"tags": ["{{ handle_fetch_input }}"]}}
  }
}
```
