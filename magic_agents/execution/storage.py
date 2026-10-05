"""Authoritative native execution storage contracts, separate from observers.

No implementation is selected here. Adapters must enforce configured byte/count
limits before decoding, check current caller authority, and propagate failures.
A root lease owns short state commits; it never serializes concurrent node work.
Recovery is explicitly requested by the caller, never an automatic worker.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=64)]
NodeId = Annotated[str, Field(min_length=1, max_length=256)]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
MessagingEngine = Literal['in_memory', 'db_persistence']
ExecutionStatus = Literal['pending', 'running', 'completed', 'failed', 'cancelled', 'blocked']


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


class StorageModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    @field_validator('node_path', 'effects', 'events', 'coordination_scopes', 'coordinator_state', mode='before', check_fields=False)
    @classmethod
    def json_arrays(cls, value):
        # JSON arrays reconstruct immutable tuple fields; no arbitrary iterable coercion.
        return tuple(value) if type(value) is list else value

    @model_validator(mode='after')
    def finite_json(self):
        canonical_bytes(self.model_dump(mode='json'))
        return self


class VersionedStorageModel(StorageModel):
    schema_version: Literal[1] = 1

    @model_validator(mode='before')
    @classmethod
    def exact_version(cls, value):
        if isinstance(value, dict) and 'schema_version' in value and type(value['schema_version']) is not int:
            raise ValueError('schema_version must be an integer')
        return value


class ExecutionIdentity(StorageModel):
    run_id: Identifier
    root_execution_id: Identifier
    conversation_id: Identifier | None


class ExecutionCause(StorageModel):
    """Causal identity only; never provider-supplied authorization."""
    kind: Literal['normal', 'coordinator', 'agent']
    run_id: Identifier | None = None
    execution_id: Identifier | None = None
    node_path: tuple[NodeId, ...] = Field(default=(), max_length=64)
    actor_id: Identifier | None = None
    sender_actor_id: Identifier | None = None
    activation_id: Identifier | None = None
    message_id: Identifier | None = None
    request_id: Identifier | None = None
    causal_event_id: Identifier | None = None


class EffectRecord(VersionedStorageModel):
    """Unknown effects require reconciliation, never blind redispatch."""
    effect_id: Identifier
    attempt_id: Identifier
    kind: Annotated[str, Field(min_length=1, max_length=64)]
    status: Literal['prepared', 'dispatched', 'succeeded', 'failed', 'unknown']
    request_digest: Digest
    result_digest: Digest | None = None
    result: JsonValue = None
    cause: ExecutionCause

    @model_validator(mode='after')
    def result_binding(self):
        if self.result is not None:
            digest = hashlib.sha256(canonical_bytes(self.result)).hexdigest()
            if self.result_digest != digest:
                raise ValueError('Effect result digest does not match its retained JSON')
        return self


class ExecutionEvent(VersionedStorageModel):
    event_id: Identifier
    # Producer logical sequence; SQL owns its separate run-wide audit sequence.
    sequence: int = Field(gt=0)
    kind: Annotated[str, Field(min_length=1, max_length=64)]
    cause: ExecutionCause
    data: JsonValue = None


class CoordinationScopeConfig(StorageModel):
    node_path: tuple[NodeId, ...] = Field(default=(), max_length=64)
    messaging_engine: MessagingEngine = 'in_memory'


class CoordinatorCheckpoint(VersionedStorageModel):
    scope_instance_id: Identifier
    node_path: tuple[NodeId, ...] = Field(default=(), max_length=64)
    state: dict[str, JsonValue]


class ExecutionSnapshot(VersionedStorageModel):
    """Immutable authored graph captured before connection substitution."""
    identity: ExecutionIdentity
    graph_definition: dict[str, JsonValue]
    graph_digest: Digest
    runtime_revision: Annotated[str, Field(min_length=1, max_length=128)]
    coordination_scopes: tuple[CoordinationScopeConfig, ...] = ()
    cause: ExecutionCause
    initial_input: JsonValue = None

    @model_validator(mode='after')
    def graph_binding(self):
        if hashlib.sha256(canonical_bytes(self.graph_definition)).hexdigest() != self.graph_digest:
            raise ValueError('Graph digest does not match the authored snapshot')
        paths = [scope.node_path for scope in self.coordination_scopes]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate coordination scope definition')
        return self


class ExecutionState(VersionedStorageModel):
    status: ExecutionStatus = 'pending'
    # Core checkpoints persist in BOTH engines: node outputs, edge readiness
    # and canonical LLM state, never live objects. Only coordinator state varies.
    checkpoint: dict[str, JsonValue] | None = None
    coordinator_state: tuple[CoordinatorCheckpoint, ...] = ()
    effects: tuple[EffectRecord, ...] = ()
    output_cursor: int = Field(default=0, ge=0)
    cancel_generation: int = Field(default=0, ge=0)


class ExecutionLease(VersionedStorageModel):
    identity: ExecutionIdentity
    owner_id: Identifier
    version: int = Field(ge=0)
    fence: int = Field(gt=0)
    lease_until_ms: int = Field(gt=0)


class ExecutionRecord(VersionedStorageModel):
    snapshot: ExecutionSnapshot
    state: ExecutionState
    version: int = Field(ge=0)
    fence: int = Field(ge=0)
    owner_id: Identifier | None = None
    lease_until_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def coordinator_engine(self):
        enabled = {scope.node_path for scope in self.snapshot.coordination_scopes
                   if scope.messaging_engine == 'db_persistence'}
        instances = [scope.scope_instance_id for scope in self.state.coordinator_state]
        if len(instances) != len(set(instances)):
            raise ValueError('Duplicate coordinator scope instance')
        if any(scope.node_path not in enabled for scope in self.state.coordinator_state):
            raise ValueError('Only DB scopes can persist operational coordinator state')
        return self


class VisibleOutput(StorageModel):
    """Content is written only to the existing message, never audit/checkpoint copies."""
    assistant_message_id: Identifier
    expected_cursor: int = Field(ge=0)
    next_cursor: int = Field(gt=0)
    assistant_text: str
    reasoning: str
    response_id: Digest | None = None
    text_offset: int | None = Field(default=None, ge=0)
    reasoning_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def cursor_step(self):
        if self.next_cursor != self.expected_cursor + 1:
            raise ValueError('Visible output must advance one committed batch')
        return self


class ExecutionTransition(VersionedStorageModel):
    operation_id: Identifier
    expected_version: int = Field(ge=0)
    state: ExecutionState
    events: tuple[ExecutionEvent, ...] = ()
    cause: ExecutionCause
    visible_output: VisibleOutput | None = None

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self.model_dump(mode='json'))).hexdigest()


class CommitReceipt(VersionedStorageModel):
    identity: ExecutionIdentity
    operation_id: Identifier
    content_digest: Digest
    version: int = Field(ge=0)
    fence: int = Field(ge=0)
    status: ExecutionStatus


class ExecutionUsageObservation(StorageModel):
    """Genuine final accounting for one original physical provider attempt."""
    attempt_id: Identifier
    request_digest: Digest
    provider: Annotated[str, Field(min_length=1, max_length=64)]
    model: Annotated[str, Field(min_length=1, max_length=128)]
    usage: dict[str, JsonValue]

    @model_validator(mode='after')
    def final_counts(self):
        if not self.provider.isascii() or not self.model.isascii():
            raise ValueError('Usage provider/model identity must be ASCII')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if type(self.usage.get(key)) is not int or not 0 <= self.usage[key] <= 2**31 - 1:
                raise ValueError('Usage requires actual final integer token counts')
        return self

    @property
    def usage_digest(self):
        return hashlib.sha256(canonical_bytes({'provider': self.provider,
            'model': self.model, 'usage': self.usage})).hexdigest()


class ExecutionUsageReceipt(StorageModel):
    identity: ExecutionIdentity
    attempt_id: Identifier
    request_digest: Digest
    usage_digest: Digest


class ExecutionStorageLimits(StorageModel):
    """Host-supplied finite limits, enforced before fetch/decode or mutation."""
    max_snapshot_bytes: int = Field(gt=0)
    max_state_bytes: int = Field(gt=0)
    max_event_bytes: int = Field(gt=0)
    max_events_per_commit: int = Field(gt=0)
    max_effects: int = Field(gt=0)


class ExecutionStorageError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ExecutionStatePort(Protocol):
    """Await directly from the executor, outside HookRegistry error isolation.

    Core checkpoints/effects and claim fencing apply to BOTH engine modes.
    Only DB messaging persists operational coordinator state. A null conversation
    is not anonymous read authority: the adapter still enforces recorded ownership.
    create reuses the supplied core identity and compares immutable snapshot
    bytes on retry. claim uses the database clock and an increasing fence;
    its caller is the explicitly requested manual continuation. commit/cancel
    retain operation ID + content digest + original receipt atomically with
    state/events. After current authorization, exact operation replay returns
    the original receipt even if version/fence advanced; changed bytes conflict.
    Terminal cancellation cannot be claimed again. Unknown commits must be
    reconciled by load/exact operation replay, never by repeating effects.
    """
    limits: ExecutionStorageLimits

    async def create(self, snapshot: ExecutionSnapshot) -> ExecutionRecord: ...
    async def load(self, identity: ExecutionIdentity) -> ExecutionRecord | None: ...
    async def claim(self, identity: ExecutionIdentity, *, expected_version: int,
                    owner_id: str, lease_seconds: float) -> ExecutionLease: ...
    async def commit(self, lease: ExecutionLease,
                     transition: ExecutionTransition) -> CommitReceipt: ...
    async def cancel(self, identity: ExecutionIdentity, *, expected_version: int,
                     operation_id: str, cause: ExecutionCause) -> CommitReceipt: ...

    async def record_usage(self, identity: ExecutionIdentity,
                           observation: ExecutionUsageObservation) -> ExecutionUsageReceipt:
        """Accounting only: no execution authority/state, effect or output change.

        Current ownership authorization and original effect/span identity remain
        mandatory. Missing facts may be reconciled only for a cancelled original
        attempt; exact retained facts replay, changed usage conflicts.
        """
        ...
