"""Pinned private Skills state around a canonical actor-loop checkpoint.

This is JSON data, not a pickle of a request guard or a projected transcript.
The durable owner must persist it with the loop and resource/effect journals.
"""
from __future__ import annotations

import copy
import hashlib
import json

from magic_llm.agent.control import AgentControlError, AgentLoopCheckpoint
from magic_agents.coordination.control import ActorLoopControl
from magic_agents.coordination.service import CoordinationError, _json, native_inbox_message
from magic_agents.models.factory.Nodes.SkillsNodeModel import SkillPromptModel
from magic_agents.skills import (
    EmbeddedSkillPrompt, SkillPromptBundle, create_request_guard, freeze_json, merge_skill_bundles,
)

CHECKPOINT_KEY = 'coordination_skills'


def _invalid():
    return CoordinationError('skills_checkpoint_invalid', 'Pinned Skills checkpoint is missing or incompatible')


def _manifest(bundle):
    owners = dict(bundle.skill_sources) if bundle.source_node_id is None else {
        entry.id: bundle.source_node_id for entry in bundle.skills}
    return {'sources': list(bundle.ordered_source_ids), 'skills': [
        {'sourceId': owners[entry.id], 'definition': entry.definition()} for entry in bundle.skills]}


def _bundle(data):
    if (type(data) is not dict or set(data) != {'sources', 'skills'}
            or type(data['sources']) is not list or not 1 <= len(data['sources']) <= 64
            or any(type(v) is not str or not 1 <= len(v.encode('utf-8')) <= 256 for v in data['sources'])
            or len(set(data['sources'])) != len(data['sources'])
            or type(data['skills']) is not list or not 1 <= len(data['skills']) <= 16):
        raise _invalid()
    entries = {source: [] for source in data['sources']}
    for item in data['skills']:
        if (type(item) is not dict or set(item) != {'sourceId', 'definition'}
                or type(item['sourceId']) is not str or item['sourceId'] not in entries):
            raise _invalid()
        entry = SkillPromptModel.model_validate(item['definition'])
        if not entry.enabled:
            raise _invalid()
        entries[item['sourceId']].append(EmbeddedSkillPrompt(entry.id, entry.name, entry.description,
            entry.prompt, freeze_json(entry.props)))
    return merge_skill_bundles(tuple(SkillPromptBundle(source, tuple(values)) for source, values in entries.items()))


class PinnedSkills:
    """Bounded immutable catalog plus reconstructible invocation provenance."""
    def __init__(self, bundle, max_input_tokens):
        cap = 32768 if max_input_tokens is None else max_input_tokens
        if type(cap) is not int or cap < 0:
            raise _invalid()
        encoded = _json(_manifest(bundle), max_bytes=256 * 1024)
        # Validate even trusted bundle data before retaining the immutable bytes.
        _bundle(json.loads(encoded))
        self._manifest_json = encoded
        self.digest = hashlib.sha256(encoded).hexdigest()
        self.cap = cap
        self._guard_state = None

    @classmethod
    def restore(cls, data, checkpoint: AgentLoopCheckpoint):
        try:
            if (type(data) is not dict or set(data) != {
                    'schemaVersion', 'manifest', 'manifestDigest', 'maxInputTokens', 'guard'}
                    or type(data['schemaVersion']) is not int or data['schemaVersion'] != 1):
                raise _invalid()
            state = cls(_bundle(data['manifest']), data['maxInputTokens'])
            if state.digest != data['manifestDigest'] or not checkpoint.requires_context_guard:
                raise _invalid()
            guard = create_request_guard(state.cap, source_node_id=state.bundle.source_node_id,
                source_node_ids=state.bundle.ordered_source_ids, resume_state=data['guard'])
            captured = guard.checkpoint_state(checkpoint.messages)
            if captured != data['guard']:
                raise _invalid()
            state._guard_state = copy.deepcopy(captured)
            return state
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise _invalid() from exc

    @property
    def bundle(self):
        return _bundle(json.loads(self._manifest_json))

    @property
    def authorization_manifest(self):
        value = json.loads(self._manifest_json)
        return {'schemaVersion': 1, 'manifestDigest': self.digest, 'sourceNodeIds': value['sources'],
                'skillIds': [item['definition']['id'] for item in value['skills']]}

    def attach_guard(self, node):
        cap = min(self.cap, node._max_input_tokens) if node._max_input_tokens is not None else self.cap
        self.cap = cap  # narrowing is retained; a later wake cannot raise it again
        bundle = self.bundle
        guard = create_request_guard(cap, source_node_id=bundle.source_node_id,
            source_node_ids=bundle.ordered_source_ids, resume_state=self._guard_state)
        node._skills_request_guard = guard
        return guard

    def snapshot(self, checkpoint, guard):
        self._guard_state = guard.checkpoint_state(checkpoint.messages)
        return {'schemaVersion': 1, 'manifest': json.loads(self._manifest_json),
                'manifestDigest': self.digest, 'maxInputTokens': self.cap,
                'guard': copy.deepcopy(self._guard_state)}


def split_checkpoint(retained):
    data = copy.deepcopy(retained)
    skills = data.pop(CHECKPOINT_KEY, None)
    try:
        loop = AgentLoopCheckpoint.model_validate(data)
    except (ValueError, TypeError) as exc:
        raise _invalid() from exc
    return loop, skills


class SkillsActorLoopControl(ActorLoopControl):
    def __init__(self, caller, state, guard):
        super().__init__(caller)
        self.state, self.guard = state, guard

    def _snapshot(self, checkpoint):
        return {**checkpoint.model_dump(mode='json'), CHECKPOINT_KEY: self.state.snapshot(checkpoint, self.guard)}

    async def before_turn(self, checkpoint):
        try:
            messages = await self.caller.service.inbox(self.caller, checkpoint=self._snapshot(checkpoint))
        except CoordinationError as exc:
            raise AgentControlError(str(exc), exc.code) from exc
        return [native_inbox_message(message) for message in messages]

    async def checkpoint(self, checkpoint, boundary):
        try:
            await self.caller.service.checkpoint(self.caller, self._snapshot(checkpoint), checkpoint.consumed_message_ids, boundary=boundary)
        except CoordinationError as exc:
            raise AgentControlError(str(exc), exc.code) from exc
        self._last_checkpoint = checkpoint.detached()
