"""Adapters between the canonical LLM loop and execution-owned authority."""
from __future__ import annotations

from typing import Callable

from magic_llm.agent.control import AgentControlError, AgentLoopCheckpoint, InboxMessage
from magic_llm.engine.attempt_control import ProviderAttempt, ProviderAttemptControlError, ProviderAttemptOutcome

from magic_agents.coordination.budget import BudgetError, UsageBound, WorkgroupBudget
from magic_agents.coordination.service import ActorCaller, CoordinationError, native_inbox_message


class ActorLoopControl:
    def __init__(self, caller: ActorCaller):
        self.caller = caller
        self._last_checkpoint = None

    async def before_turn(self, checkpoint: AgentLoopCheckpoint):
        try:
            messages = await self.caller.service.inbox(self.caller, checkpoint=checkpoint.model_dump(mode="json"))
        except CoordinationError as exc:
            raise AgentControlError(str(exc), exc.code) from exc
        return [native_inbox_message(m) for m in messages]

    async def checkpoint(self, checkpoint: AgentLoopCheckpoint, boundary: str):
        try:
            await self.caller.service.checkpoint(self.caller, checkpoint.model_dump(mode="json"), checkpoint.consumed_message_ids)
        except CoordinationError as exc:
            raise AgentControlError(str(exc), exc.code) from exc
        self._last_checkpoint = checkpoint.detached()

    async def finish_candidate(self, checkpoint: AgentLoopCheckpoint):
        try:
            return await self.caller.service.candidate_decision(self.caller)
        except CoordinationError as exc:
            raise AgentControlError(str(exc), exc.code) from exc

    @property
    def retained(self):
        return self._last_checkpoint.detached() if self._last_checkpoint else None


class BudgetAttemptControl:
    """Trusted pricing/usage ports supply real upper bounds and final totals.

    Estimation must bind the actual provider/model, complete input and enforced
    output maximum. An absent price/usage bound is an admission error, not zero.
    This adapter also serves schema-only final authors outside the peer group.
    """
    def __init__(self, budget: WorkgroupBudget, *, estimate: Callable[[ProviderAttempt], UsageBound],
                 usage: Callable[[ProviderAttempt, ProviderAttemptOutcome], UsageBound | None],
                 authorize: Callable[[ProviderAttempt], None]):
        self.budget, self.estimate, self.usage, self.authorize = budget, estimate, usage, authorize

    async def before_attempt(self, attempt: ProviderAttempt):
        try:
            self.authorize(attempt)
            estimate = self.estimate(attempt)
            if not isinstance(estimate, UsageBound):
                raise BudgetError("usage_bound_unavailable", "A trusted complete request bound is required")
            await self.budget.reserve(attempt.provider_attempt_id, estimate, guard=lambda: self.authorize(attempt))
        except BudgetError as exc:
            raise ProviderAttemptControlError(str(exc), error_code=exc.code) from exc

    async def after_attempt(self, attempt: ProviderAttempt, outcome: ProviderAttemptOutcome):
        # Settlement remains mandatory after cancellation/lease loss. Admission
        # is not rerun here: doing so would strand an already incurred charge.
        actual = None
        failed = None
        try: actual = self.usage(attempt, outcome)
        except Exception as exc: failed = exc
        try:
            await self.budget.settle(attempt.provider_attempt_id, actual,
                                     uncertain=failed is not None or outcome.usage_uncertain or actual is None)
        except BudgetError as exc:
            raise ProviderAttemptControlError(str(exc), error_code=exc.code) from exc
        if failed is not None:
            raise ProviderAttemptControlError("Usage reconciliation failed; reservation retained", error_code="usage_unavailable") from failed
