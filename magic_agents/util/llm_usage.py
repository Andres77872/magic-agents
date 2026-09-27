"""Provider-neutral usage normalization shared by nodes making LLM calls."""

from typing import Any


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

def _usage_to_plain_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, 'model_dump'):
        dumped = usage.model_dump()
        return dict(dumped or {}) if isinstance(dumped, dict) else {}
    return {}

def usage_detail_outputs(usage: Any, *, response_id: Any = None) -> dict[str, Any]:
    data = _usage_to_plain_dict(usage)

    def attr(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None) if usage is not None else None

    def nested_detail(container_name: str, field_name: str) -> Any:
        nested = data.get(container_name)
        if nested is None and usage is not None and not isinstance(usage, dict):
            nested = getattr(usage, container_name, None)
        if isinstance(nested, dict):
            return nested.get(field_name)
        return getattr(nested, field_name, None) if nested is not None else None

    audio_tokens = _first_not_none(data.get('audio_tokens'), attr('audio_tokens'))
    if audio_tokens is None:
        audio_prompt = nested_detail('prompt_tokens_details', 'audio_tokens')
        audio_completion = nested_detail('completion_tokens_details', 'audio_tokens')
        if audio_prompt is not None or audio_completion is not None:
            audio_tokens = (audio_prompt or 0) + (audio_completion or 0)

    raw_usage_json = data.get('raw_usage_json')
    if raw_usage_json is None:
        raw_usage_json = attr('raw_usage_json')
    if raw_usage_json is None and data and hasattr(usage, 'model_dump'):
        raw_usage_json = dict(data)

    return {
        'provider_request_id': _first_not_none(
            data.get('provider_request_id'),
            attr('provider_request_id'),
            response_id,
        ),
        'prompt_tokens': _first_not_none(data.get('prompt_tokens'), attr('prompt_tokens')),
        'completion_tokens': _first_not_none(data.get('completion_tokens'), attr('completion_tokens')),
        'total_tokens': _first_not_none(data.get('total_tokens'), attr('total_tokens')),
        'cached_tokens_read': _first_not_none(
            data.get('cached_tokens_read'),
            data.get('cached_read_tokens'),
            attr('cached_tokens_read'),
            attr('cached_read_tokens'),
            nested_detail('prompt_tokens_details', 'cached_tokens'),
        ),
        'cached_tokens_write': _first_not_none(
            data.get('cached_tokens_write'),
            data.get('cached_write_tokens'),
            attr('cached_tokens_write'),
            attr('cached_write_tokens'),
        ),
        'reasoning_tokens': _first_not_none(
            data.get('reasoning_tokens'),
            attr('reasoning_tokens'),
            nested_detail('completion_tokens_details', 'reasoning_tokens'),
        ),
        'audio_tokens': audio_tokens,
        'accepted_prediction_tokens': _first_not_none(
            data.get('accepted_prediction_tokens'),
            attr('accepted_prediction_tokens'),
            nested_detail('completion_tokens_details', 'accepted_prediction_tokens'),
        ),
        'rejected_prediction_tokens': _first_not_none(
            data.get('rejected_prediction_tokens'),
            attr('rejected_prediction_tokens'),
            nested_detail('completion_tokens_details', 'rejected_prediction_tokens'),
        ),
        'provider_extra': _first_not_none(data.get('provider_extra'), attr('provider_extra')),
        'service_tier': _first_not_none(data.get('service_tier'), attr('service_tier')),
        'usage_source': _first_not_none(data.get('usage_source'), attr('usage_source')),
        'raw_usage_json': raw_usage_json,
    }

