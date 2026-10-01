# `conditional`

## Purpose

Choose one output handle using application-owned Jinja2 routing. Exact rules
use `evaluation_mode: "jinja"` (the default for existing graphs). Semantic
decisions use `evaluation_mode: "llm"`: a connected MagicLLM client first answers
independent typed questions about state, then the expression routes those answers.
No Jev service or Jev credentials are used.

## Runtime class

- `NodeConditional`
- model: `ConditionalNodeModel`

## Default input

- `handle_input`
- `handle-client-provider` — required for LLM judgments, excluded from state

## Outputs

- dynamic user-defined handle such as `adult`, `minor`, `approved`, `rejected`
- internal bookkeeping `end`
- system signals like `__bypass_all__` on error paths

## Important behavior

- condition template must render the **name of the output handle**
- supports `merge_strategy: flat | namespaced`
- exposes convenience alias `value` for the primary input
- stores `selected_handle` for executor bypass propagation
- uses `default_handle` only when the rendered result is empty
- `__bypass_all__` is emitted on configuration/template errors, causing executor to skip all downstream branches
- a rendered handle with no matching outgoing edge is a `GraphRoutingError`: the node counts as failed (graph error; a sub-flow fails its Inner Flow node) and all downstream branches are skipped

## Recommended fields

- `condition` (required)
- `merge_strategy`
- `handles` — custom input handle name mappings (e.g., `{'input': 'my_input'}`)
- `output_handles`
- `default_handle`
- `evaluation_mode` — `jinja` or `llm`; question definitions can remain saved while switching modes
- `questions` — required and nonempty in LLM mode
- `evaluation_timeout` — finite positive provider timeout in seconds, default 30

## Typed LLM questions

```json
{
  "evaluation_mode": "llm",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which intent does the message express?",
      "criteria": {"approve": "Explicit approval", "reject": "Explicit rejection", "other": null}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request?",
      "criteria": ["No urgency", "Time sensitive", "Immediate emergency"]
    },
    "ready": {"type": "noul", "instructions": "Does the user explicitly authorize proceeding?"}
  },
  "condition": "{{ 'approved' if answers.intent.choice == 'approve' and answers.ready.noul >= 0.8 else 'review' }}",
  "output_handles": ["approved", "review"],
  "default_handle": "review",
  "evaluation_timeout": 30
}
```

Connect a `client` output to `handle-client-provider` and state to `handle_input`.
Custom handles use `handles.client_provider` (or `handles.client`) and
`handles.input`/`handles.context`. When both client aliases are provided,
`client_provider` takes precedence.
Choice questions have 2–255 named criteria; score questions have 2–10 ordered
descriptions. Instructions and descriptions can be strings, JSON objects, or
arrays. Choice descriptions may be null. Noul optionally accepts a `criteria`
object with `true` and/or `false` descriptions. Question IDs may contain punctuation;
use bracket syntax such as `answers['question-id'].noul` in that case.

The LLM returns only an `answers` object containing complete distributions for
choice/score and a 0–1 probability for noul. The runtime derives `choice`,
`score` (expected zero-based criterion index), `probabilities`, `legend`, and
`confidence` using Typesafe's result algebra. A noul answer is
`{"type":"noul","noul":0.9}`, representing estimated P(true), not a boolean.
All judgments are independent within one provider request. State is supplied
as evidence in a separate user message. No answer becomes another question's input.

Values must be finite numbers in [0,1]. Labels and IDs must exactly match the
configuration. Positive distribution sums are normalized; original estimates
and their sums are retained in `judgment_diagnostics.normalized_distributions`
when the discrepancy exceeds 1e-6. Missing/extra labels, zero mass, invalid JSON,
provider errors, and timeouts fail closed, bypassing every business branch.
Only an empty routing result uses `default_handle`; malformed judgments and
undeclared nonempty routes do not.

Routing can read `answers` and `state` plus existing merged input variables.
Selected branch payloads remain the original merged state. Generated answers
and the client object are not inserted into business payloads. Internal `end`
metadata/debug state includes answers, normalization diagnostics and
`confidence_source: "llm_estimated_not_calibrated"`. Provider calls emit normal
LLM lifecycle hooks and usage, including responses rejected during validation.

LLM estimates do not inherit Jev's calibration claims. Keep numeric comparisons,
truthiness, exact field matching and other exact rules in Jinja mode; use LLM
questions for interpretation of text and other semantic judgments.

## Example

See [../../examples/conditional/conditional_simple_if_else.json](../../examples/conditional/conditional_simple_if_else.json).
