# `skills`

`NodeSkills` provides complete prompt definitions embedded in the agent JSON. It needs no database, resolver, external file, API or dependency injection. An LLM initially receives only enabled skill IDs, names and descriptions. The model may call the run-local `skills_load` tool to read one or more complete definitions.

## Author a skill

```json
{
  "id": "skills_release",
  "type": "skills",
  "data": {
    "schema_version": 1,
    "skills": [
      {
        "id": "release-notes",
        "name": "Release notes",
        "description": "Write release notes from supplied changes.",
        "prompt": "Group supplied changes into Features, Fixes and Known Issues. Do not invent changes. Omit empty groups.",
        "props": {"sections": ["Features", "Fixes", "Known Issues"]},
        "enabled": true
      }
    ]
  }
}
```

`props` defaults to `{}` and `enabled` to `true`. Props are inert JSON data; they do not execute scripts, resolve templates, select models or change tools. Write complete instructions in `prompt`, and describe when to use them in `description`. Keep confidential credentials out of prompts and props: graph editing/export intentionally includes complete authored definitions, and selected records are sent to the chosen model.

## Wire the source

```json
{
  "id": "skills_to_writer",
  "source": "skills_release",
  "target": "writer",
  "sourceHandle": "handle-skills",
  "targetHandle": "handle-skills"
}
```

Skills has no inputs. Its output is `handle-skills`; `data.handles.output` can rename it. The LLM input is `handle-skills`; `data.handles.skills` can rename it. Both ends must use the resolved dedicated handles. A Skills source can feed several LLMs, and one LLM can consume several distinct Skills sources through this dedicated input. Each source may connect only once to a particular LLM. Other source types, wrong targets/handles, duplicate source edges and aliases colliding with ordinary/tool inputs fail even when general graph validation is disabled.

For example, connect `skills-x` to both `llm-a` and `llm-b`, and connect `skills-y` only to `llm-b`. A discovers and can load only X; B discovers and can load X plus Y. The shared X bundle is immutable. Each consumer receives its own merged catalog, loader closure, request guard and observer state. Sources merge in original graph-edge declaration order, followed by each source's authored entry order; arrival order does not change the catalog.

The source emits a deeply immutable `SkillPromptBundle` of enabled entries. A valid all-disabled source emits an explicit empty bundle and adds no catalog, loader or loop. Empty configured lists, invalid disabled entries and unfinished definitions fail validation. Every connected source must deliver exactly one valid source bundle, including an all-disabled source. A connected source that fails or is bypassed blocks its consumer; another valid source cannot hide that failure. Bypass clears only the failed source's previous delivery. A separate consumer whose own sources succeeded can still run.

## Strict validation

| Field | Constraint |
| --- | --- |
| `schema_version` | Required integer `1` |
| `skills` | 1–16 complete entries, including disabled entries |
| `id` | Unique within the node; 1–64 lowercase ASCII letters, digits or hyphens |
| `name` | Nonblank strict string, at most 128 characters |
| `description` | Nonblank strict string, at most 1024 characters |
| `prompt` | Nonblank strict string, at most 16384 characters; original whitespace retained |
| `enabled` | Strict boolean; default `true` |
| `props` | Strict finite JSON object; default `{}`; no cycles or nonstring keys |
| Enabled prompt aggregate | At most 32768 characters |
| Props depth | At most 5 container levels; root object is level 1 |
| Props size | At most 4096 serialized UTF-8 bytes per entry and 16384 per node, including disabled entries |

Across each consumer's enabled merged catalog, the limits are 16 entries, 32768 prompt characters and 16384 serialized props bytes. Distinct connected sources defining the same enabled ID fail with `SKILLS_ID_CONFLICT`, even if their bodies match. Disabled or disconnected duplicate IDs do not conflict. No definitions are silently overwritten, and there is no additional source-count cap. These aggregate checks run during graph construction and again on runtime deliveries.

Props bytes use `json.dumps(props, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')`. Unknown entry/data fields fail. Factory validation errors contain a bounded safe code and never raw prompt/props values.

## Model selection and loading

With enabled entries, NodeLLM makes an isolated chat copy and combines existing system messages with one metadata catalog in one leading system message. User and other history messages remain in order. The generic tool manifest and `skills_load` schema contain no prompt or props.

The model can call:

```json
{"skill_ids": ["release-notes"]}
```

The list must contain 1–16 unique enabled IDs and no extra arguments. Unknown/disabled IDs, duplicates and wrong types produce typed error results with no partial definitions. Requested order is preserved. Success returns the complete selected records:

```json
{
  "schema_version": 1,
  "source_node_id": "skills_release",
  "skills": [
    {
      "id": "release-notes",
      "name": "Release notes",
      "description": "Write release notes from supplied changes.",
      "prompt": "Group supplied changes into Features, Fixes and Known Issues. Do not invent changes. Omit empty groups.",
      "props": {"sections": ["Features", "Fixes", "Known Issues"]},
      "enabled": true
    }
  ]
}
```

For multiple connected sources, the legacy `source_node_id` field is replaced by `source_node_ids` in graph-edge order and a `skill_sources` mapping for only the selected IDs. The records remain complete and retain requested order. For a B batch requesting Y then X:

```json
{
  "schema_version": 1,
  "source_node_ids": ["skills-x", "skills-y"],
  "skill_sources": {"release-notes": "skills-y", "shared-summary": "skills-x"},
  "skills": [
    {"id": "release-notes", "name": "Release notes", "description": "Write release notes.", "prompt": "Use supplied changes only.", "props": {}, "enabled": true},
    {"id": "shared-summary", "name": "Shared summary", "description": "Summarize supplied input.", "prompt": "Write a concise summary.", "props": {}, "enabled": true}
  ]
}
```

Single-source results and provenance keep their original shape for compatibility. The complete envelope enters ordinary canonical tool history before the next request. No full definition is injected into another system message. The model may load none, request several entries, retry a smaller batch or repeat a prior load; each invocation has independent loader state.

The exact executor serialization must fit the effective tool content limit. Oversized selections return an atomic error, never a truncated fragment. If the configured limit cannot hold the safe error envelopes, the host rejects the run before its first provider call.

## Execution gates and context

Enabled Skills require the callable agent loop and current `magic-llm` complete-output, mandatory request guard, final-payload guard, registration-name and observer projection APIs. `skills_load` is reserved across graph tools and registered client tasks. Schema-only Client Tools cannot share this LLM with an enabled Skills catalog. Effective tool choice, including Client `api_info`/`extra_data`, must be absent or `auto`.

The implemented engine allowlist is `openai` (Chat Completions and Responses), `anthropic` and `google`. Other engines fail before generation. Offline adapter/loop tests cover native replay and continuation; exact deployed model support and credentials still require deployment validation.

Every active request must preserve complete tool exchanges. NodeLLM uses configured `max_input_tokens`, or a finite 32768-token host cap when unset. It conservatively accounts for each serialized UTF-8 byte as a token, adds image headroom and reserves effective output tokens (including native mapped/default limits; 1024 when unspecified). It checks both assembled canonical messages/tools and the final mapped provider JSON before logging or I/O. Overflow is terminal before another provider call, with no paid retry, tool-error normalization or silent history trimming. This cap is not a guarantee of any model's context capacity; configure a smaller explicit limit where needed.

## Observer privacy and stale history

Model inference receives selected complete records. Lifecycle hooks, debug events, provider debug logs and configured provider callbacks receive separate metadata summaries. Summaries contain IDs/counts/status and host provenance `source: builtin_skills`, `ephemeral: true`; they exclude definition bodies. The final response remains ordinary model output.

New invocations atomically remove old host-marked Skills pairs before history windowing and observer invocation, including after edit/disable/removal. Mixed tool batches retain unrelated calls/results; historical same-name functions without host provenance remain untouched. OpenAI tool messages, Anthropic result blocks and Google response/native replay parts are supported. Current live continuation retains complete canonical results until the invocation ends.

The LLM lifecycle `internal_state.skills` contains `available_ids`, `available_count` and bounded `loaded_ids`. A single source adds `source_node_id`; multiple sources add `source_node_ids` and the available-ID `skill_sources` ownership map. Tool call/result summaries use `skills_source_node_id` for a single source or `skills_source_node_ids` plus selected-ID `skill_sources` for multiple sources. Summary projection is idempotent. This is diagnostics only and never controls whether a prompt is applied.

Construct a fresh graph for each request, or use the existing invocation factory for a fresh configured processor; completed built graph instances retain the runtime's existing response cache. Each fresh invocation starts with an empty loaded-ID summary and a new loader/guard/observer closure, including when it reuses a MagicLLM client.

## Verification and TBD

`test/test_embedded_skills.py` verifies 72 offline cases: strict validation and immutable props, exact batches/errors/bounds, factory/aliases/fanout/readiness, metadata-first buffered/streamed requests, repeat loads/invocations, stale native history, terminal context overflow and actual provider callback/log privacy.

`test/test_multisource_skills.py` adds 22 offline cases for the real shared-client X → A/B, Y → B graph and final OpenAI transport payloads, concurrent consumers and reused call IDs, private A-only loading and atomic guessed-Y rejection, ordered B batches with ownership, every-source readiness, reverse delivery order, fresh invocations and source failure, duplicate source/ID rejection, aggregate limits, disabled/disconnected IDs, native stale-history cleanup and idempotent observer privacy. Together these suites pass 94 cases.

All other Skills capabilities remain TBD: file/reference paging, scripts/assets, YAML/SKILL.md import, external catalog/version/ACL infrastructure, automatic host selection, slash arguments, model overrides, native hosted Skills and skill-specific hooks/subagents.
