"""Async Jev System One transport using the existing aiohttp dependency."""
from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp

from magic_agents.execution.condition_evaluator_llm import _object_without_duplicates
from magic_agents.models.factory.Nodes.ConditionalNodeModel import JevConnection
from magic_agents.util.env_resolver import resolve_env_string


class JevEvaluationError(ValueError):
    """Public-safe transport errors never include credentials or response bodies."""


@dataclass
class JevResponse:
    model: str
    answers: Any
    usage: dict
    id: str | None = None

    @property
    def content(self):
        try:
            return json.dumps({'answers': self.answers}, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            # Preserve known provider usage before semantic validation rejects
            # invalid answers (including JSON exponent overflow to infinity).
            return None


def _retry_delay(headers, attempt: int) -> float:
    for name, scale in [('retry-after-ms', .001), ('Retry-After', 1.0)]:
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) * scale
            if 0 <= delay < float('inf'):
                return delay
        except (ValueError, TypeError):
            if name == 'Retry-After':
                try:
                    parsed = parsedate_to_datetime(value)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
    return min(2.0, .25 * 2 ** attempt) + random.uniform(0, .1)


def _usage(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise JevEvaluationError('Jev response is missing valid token usage')
    values = [raw.get('input_tokens'), raw.get('output_tokens')]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise JevEvaluationError('Jev token usage must contain non-negative integer counts')
    return {'prompt_tokens': values[0], 'completion_tokens': values[1],
            'total_tokens': sum(values), 'raw_usage_json': raw, 'usage_source': 'provider'}


async def _read_response(response) -> str:
    """Bound the response before buffering an unexpectedly large service body."""
    limit = 2 * 1024 * 1024
    stream = getattr(response, 'content', None)
    if stream is not None and hasattr(stream, 'iter_chunked'):
        chunks, size = [], 0
        async for chunk in stream.iter_chunked(65536):
            size += len(chunk)
            if size > limit:
                raise JevEvaluationError('Jev response exceeded the maximum size')
            chunks.append(chunk)
        try:
            return b''.join(chunks).decode('utf-8')
        except UnicodeDecodeError:
            raise JevEvaluationError('Jev returned invalid UTF-8 JSON') from None
    # Response-compatible test adapters may provide only text().
    text = await response.text()
    if len(text.encode('utf-8')) > limit:
        raise JevEvaluationError('Jev response exceeded the maximum size')
    return text


async def evaluate_jev(state: Any, questions: dict[str, dict], connection: dict | None,
                       *, timeout: float) -> JevResponse:
    """Submit every independent question in one call with a bounded retry budget.

    Only explicit transient HTTP responses are retried. Ambiguous disconnects
    and read timeouts are not retried, avoiding a second potentially billed POST.
    Redirects and environment proxies are disabled so bearer keys stay local to
    the configured service. The timeout includes retries and backoff.
    """
    settings = JevConnection(**(connection or {}))
    api_key = resolve_env_string(settings.api_key or '').strip() or os.getenv('JEV_API_KEY', '').strip()
    if not api_key:
        raise JevEvaluationError('Jev requires an API key or the server JEV_API_KEY environment variable')
    if '\r' in api_key or '\n' in api_key:
        raise JevEvaluationError('Jev API key contains invalid header characters')
    try:
        resolved_url = JevConnection(base_url=resolve_env_string(settings.base_url)).base_url
    except ValueError:
        raise JevEvaluationError('Jev base URL must resolve to a valid HTTP(S) URL without credentials, query or fragment') from None
    url = resolved_url + '/systemone'
    payload = {'state': state, 'model': 'jev-latest', 'questions': questions}
    # Validate before sending; stdlib json otherwise silently serializes NaN.
    try:
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError):
        raise JevEvaluationError('Jev state and questions must be finite JSON values') from None
    try:
        async with asyncio.timeout(timeout):
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False) as session:
                for attempt in range(3):
                    async with session.request('POST', url, json=payload,
                            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
                            allow_redirects=False) as response:
                        if response.status in {429, 500, 502, 503, 504, 529} and attempt < 2:
                            delay = _retry_delay(response.headers, attempt)
                        elif response.status != 200:
                            raise JevEvaluationError(f'Jev evaluation failed (HTTP {response.status})')
                        else:
                            # The service returns JSON; reject duplicate keys and
                            # nonfinite constants instead of accepting ambiguity.
                            text = await _read_response(response)
                            try:
                                data = json.loads(text, object_pairs_hook=_object_without_duplicates,
                                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                            except (ValueError, TypeError):
                                raise JevEvaluationError('Jev returned invalid JSON') from None
                            if not isinstance(data, dict) or not isinstance(data.get('model'), str) or 'answers' not in data:
                                raise JevEvaluationError('Jev returned an invalid response envelope')
                            return JevResponse(model=data['model'], answers=data['answers'], usage=_usage(data.get('usage')),
                                id=response.headers.get('x-typesafe-request-id') or response.headers.get('x-request-id') or response.headers.get('request-id'))
                    await asyncio.sleep(delay)
    except asyncio.TimeoutError:
        raise
    except aiohttp.ClientError:
        raise JevEvaluationError('Jev connection failed') from None
