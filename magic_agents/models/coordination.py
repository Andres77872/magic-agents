"""Versioned authored coordination policy; no runtime identities or credentials.

Parsing preserves configuration independently from runtime capability admission.
Applications must supply a complete server budget before executing a workgroup.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

RoleAlias = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
BuiltinName = Literal["listAgents", "sendMessageToAgent", "inspectAgent", "replyToAgent", "waitForMessage", "waitForAgent"]
BUILTIN_NAMES = ("listAgents", "sendMessageToAgent", "inspectAgent", "replyToAgent", "waitForMessage", "waitForAgent")


class CoordinationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False,
                              populate_by_name=True, validate_assignment=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def strict_schema_version(cls, value):
        # Literal equality accepts True and 1.0 despite strict model settings.
        if type(value) is not int:
            raise ValueError("schemaVersion must be an integer")
        return value


class CostLimit(CoordinationModel):
    amount: Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[0-9]+(?:\.[0-9]+)?$")]
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]

    @model_validator(mode="after")
    def positive(self):
        if Decimal(self.amount) <= 0:
            raise ValueError("Cost allowance must be positive")
        return self


class CoordinationLimits(CoordinationModel):
    max_pending_messages_per_actor: int = Field(default=32, gt=0, alias="maxPendingMessagesPerActor")
    max_accepted_messages: int = Field(default=64, gt=0, alias="maxAcceptedMessages")
    max_inline_message_bytes: int = Field(default=16384, gt=0, alias="maxInlineMessageBytes")
    max_wakeups_per_actor: int = Field(default=0, ge=0, alias="maxWakeupsPerActor")
    max_wakeups_per_workgroup: int = Field(default=0, ge=0, alias="maxWakeupsPerWorkgroup")
    max_request_depth: int = Field(default=8, gt=0, alias="maxRequestDepth")
    max_single_wait_seconds: float = Field(default=30, gt=0, alias="maxSingleWaitSeconds")
    max_group_lifetime_seconds: float = Field(default=240, gt=0, alias="maxGroupLifetimeSeconds")
    max_concurrent_model_turns: int = Field(default=2, gt=0, alias="maxConcurrentModelTurns")
    max_concurrent_jobs: int = Field(default=4, gt=0, alias="maxConcurrentJobs")
    max_model_turns: int | None = Field(default=None, gt=0, alias="maxModelTurns")
    max_input_tokens: int | None = Field(default=None, gt=0, alias="maxInputTokens")
    max_output_tokens: int | None = Field(default=None, gt=0, alias="maxOutputTokens")
    max_tool_calls: int | None = Field(default=None, ge=0, alias="maxToolCalls")
    max_image_jobs: int | None = Field(default=None, ge=0, alias="maxImageJobs")
    max_cost: CostLimit | None = Field(default=None, alias="maxCost")

    def intersect_server_policy(self, server: "CoordinationLimits", *, require_complete: bool = True) -> "CoordinationLimits":
        """Effective ceilings never exceed server policy; counters live elsewhere."""
        required = ("max_model_turns", "max_input_tokens", "max_output_tokens",
                    "max_tool_calls", "max_image_jobs", "max_cost")
        if require_complete and any(getattr(server, name) is None for name in required):
            raise ValueError("A complete finite server spending policy is required")
        effective = {}
        for name in type(self).model_fields:
            requested, ceiling = getattr(self, name), getattr(server, name)
            if ceiling is None:
                effective[name] = requested
            elif name == "max_cost":
                if requested is not None and requested.currency != ceiling.currency:
                    raise ValueError("Cost currencies must match the server policy")
                effective[name] = (requested if requested is not None and
                                   Decimal(requested.amount) < Decimal(ceiling.amount) else ceiling)
            else:
                effective[name] = ceiling if requested is None else min(requested, ceiling)
        return CoordinationLimits.model_validate(effective).model_copy(deep=True)


class ContinuationPolicy(CoordinationModel):
    after_seal: Literal["new_epoch"] = Field(default="new_epoch", alias="afterSeal")
    retention_seconds: float = Field(gt=0, alias="retentionSeconds")
    wake_horizon_seconds: float = Field(gt=0, alias="wakeHorizonSeconds")
    authorized_initiators: list[Literal["owner"]] = Field(default_factory=lambda: ["owner"], alias="authorizedInitiators", min_length=1, max_length=1)

    @model_validator(mode="after")
    def retained_for_horizon(self):
        if self.retention_seconds < self.wake_horizon_seconds:
            raise ValueError("Continuation retention must cover the wake horizon")
        return self


class CoordinationPolicy(CoordinationModel):
    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    enabled: bool = False
    lifetime: Literal["attached", "durable"] = "attached"
    delivery_mode: Literal["safe_boundary", "background_jobs"] = Field(default="safe_boundary", alias="deliveryMode")
    completion: Literal["quiescent_seal"] = "quiescent_seal"
    failure_policy: Literal["fail_group", "publish_partial"] = Field(default="fail_group", alias="failurePolicy")
    allowed_participants: list[RoleAlias] = Field(default_factory=list, alias="allowedParticipants", max_length=128)
    limits: CoordinationLimits = Field(default_factory=CoordinationLimits)
    continuation: ContinuationPolicy | None = None

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.allowed_participants)) != len(self.allowed_participants):
            raise ValueError("Participant aliases must be unique")
        if self.enabled and not self.allowed_participants:
            raise ValueError("Enabled coordination needs at least one participant")
        if self.continuation is not None:
            if self.lifetime != "durable":
                raise ValueError("Continuation requires durable ownership")
            if self.continuation.wake_horizon_seconds > self.limits.max_group_lifetime_seconds:
                raise ValueError("Continuation cannot extend the original lineage deadline")
        return self


class MessagingConfig(CoordinationModel):
    enabled: bool = False
    role: RoleAlias | None = None
    description: str = Field(default="", max_length=1024)
    peers: list[RoleAlias] = Field(default_factory=list, max_length=128)
    can_wake_peers: list[RoleAlias] = Field(default_factory=list, alias="canWakePeers", max_length=128)
    wake_on_message: bool = Field(default=False, alias="wakeOnMessage")
    tools: list[BuiltinName] = Field(default_factory=lambda: list(BUILTIN_NAMES), max_length=6)

    @model_validator(mode="after")
    def permissions(self):
        if self.enabled and self.role is None:
            raise ValueError("Enabled messaging requires a stable role alias")
        for name in ("peers", "can_wake_peers", "tools"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not set(self.can_wake_peers).issubset(self.peers):
            raise ValueError("canWakePeers must be a subset of peers")
        if self.role in self.can_wake_peers:
            raise ValueError("Self-wake is not supported")
        return self


class CoordinationCapabilities(CoordinationModel):
    schema_version: Literal[1] = Field(default=1, alias="schemaVersion")
    messaging: bool = False
    wake: bool = False
    background_jobs: bool = Field(default=False, alias="backgroundJobs")
    durable: bool = False
    continuation: bool = False
    skills_checkpoint: bool = Field(default=False, alias="skillsCheckpoint")
    lifecycle_controls: bool = Field(default=False, alias="lifecycleControls")
    on_demand_participants: bool = Field(default=False, alias="onDemandParticipants")
    provider_attempt_control: bool = Field(default=False, alias="providerAttemptControl")
    publication_version: int = Field(default=0, ge=0, alias="publicationVersion")
