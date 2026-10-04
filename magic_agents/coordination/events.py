"""Bounded attached metadata journal; never a durable outbox or authority.

The coordinator appends synchronously at its serialized transition. Reads return
detached values. Progress may expire, with an explicit gap; terminal capacity is
reserved before admitting a scope and cannot be consumed by ordinary traffic.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import re
from uuid import uuid4


class EventJournalError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class PrivateEventJournal:
    def __init__(self, *, max_events=4096, max_bytes=64 * 1024 * 1024, record_bytes=32768):
        if any(type(v) is not int or v < 1 for v in (max_events, max_bytes, record_bytes)) or record_bytes > max_bytes:
            raise ValueError('Finite private journal bounds are required')
        self.max_events, self.max_bytes, self.record_bytes = max_events, max_bytes, record_bytes
        self._ordinary, self._terminal = deque(), []
        self._ordinary_bytes = self._reserved = self._sequence = self._lost_through = 0
        self._scopes = {}
        self.root_run_id = self.workgroup_id = None
        self.changed = asyncio.Event()

    def bind_scope(self, *, root_run_id, run_id, workgroup_id, epoch_id, scope_id, terminal_slots):
        identity = dict(rootRunId=root_run_id, runId=run_id, workgroupId=workgroup_id,
                        epochId=epoch_id, scopeInstanceId=scope_id)
        if any(type(v) is not str or not 1 <= len(v.encode('utf-8')) <= 256 for v in identity.values()):
            raise EventJournalError('invalid_event_identity')
        if scope_id in self._scopes:
            if self._scopes[scope_id][0] != identity:
                raise EventJournalError('event_identity_conflict')
            return
        if self.root_run_id is not None and (root_run_id, workgroup_id) != (self.root_run_id, self.workgroup_id):
            raise EventJournalError('event_identity_conflict')
        if (type(terminal_slots) is not int or terminal_slots < 0 or
                self._reserved + terminal_slots >= self.max_events or
                (self._reserved + terminal_slots + 1) * self.record_bytes > self.max_bytes):
            raise EventJournalError('event_terminal_capacity')
        self.root_run_id, self.workgroup_id = root_run_id, workgroup_id
        self._scopes[scope_id] = (identity, terminal_slots)
        self._reserved += terminal_slots
        self._evict()

    def _evict(self):
        count = self.max_events - self._reserved
        size = self.max_bytes - self._reserved * self.record_bytes
        while self._ordinary and (len(self._ordinary) > count or self._ordinary_bytes > size):
            sequence, encoded = self._ordinary.popleft()
            self._ordinary_bytes -= len(encoded)
            self._lost_through = max(self._lost_through, sequence)

    def append(self, scope_id, kind, data, *, actor=None, message_id=None, request_id=None, terminal=False):
        """No user callbacks or awaits after mutation; encoding failure is a gap.

        Invalid internal metadata must never falsely report rollback of accepted
        work. The next read explicitly refuses a complete replay across that gap.
        """
        self._sequence += 1
        sequence = self._sequence
        try:
            identity, remaining = self._scopes[scope_id]
            payload = {key: value for key, value in data.items() if key in {
                'oldState', 'newState', 'state', 'reason', 'targetActorId', 'jobId',
                'queueDepth', 'wakeScheduled', 'participants', 'disposition', 'outcome'}}
            if 'reason' in payload and (not isinstance(payload['reason'], str) or
                    re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', payload['reason']) is None):
                payload['reason'] = 'operation_failed'
            event = dict(schemaVersion=1, channel='coordination', eventId='event-' + uuid4().hex,
                eventSequence=sequence, type=kind, **identity,
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), data=payload)
            if actor is not None:
                event.update(actorId=actor.id, nodePath=list(actor.path), outputRevision=actor.output_revision)
                if actor.activation_id: event['activationId'] = actor.activation_id
            if message_id is not None: event['messageId'] = message_id
            if request_id is not None: event['requestId'] = request_id
            encoded = json.dumps(event, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
            if len(encoded) > self.record_bytes or (terminal and remaining < 1):
                raise ValueError('Journal record exceeds its reservation')
            if terminal:
                self._scopes[scope_id] = (identity, remaining - 1)
                self._terminal.append((sequence, encoded))
            else:
                self._ordinary.append((sequence, encoded))
                self._ordinary_bytes += len(encoded)
                self._evict()
        except (KeyError, TypeError, ValueError, UnicodeError):
            self._lost_through = sequence
        self.changed.set()

    @property
    def sequence(self):
        return self._sequence

    def read(self, *, after=0):
        if type(after) is not int or not 0 <= after <= self._sequence:
            raise EventJournalError('invalid_event_cursor')
        if after < self._lost_through:
            raise EventJournalError('event_retention_unavailable')
        records = sorted((*self._ordinary, *self._terminal), key=lambda item: item[0])
        return tuple(json.loads(encoded) for sequence, encoded in records if sequence > after)

    async def wait_after(self, sequence):
        while self._sequence <= sequence:
            self.changed.clear()
            if self._sequence > sequence: break
            await self.changed.wait()
