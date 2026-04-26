import json
from typing import Any


_TRUE_VALUES = {'true', '1', 'yes', 'on'}
_FALSE_VALUES = {'false', '0', 'no', 'off', ''}


def input_has_value(inputs: dict[str, Any], handle: str) -> bool:
    return handle in inputs and inputs[handle] is not None


def coerce_bool(value: Any, *, field_name: str = 'value', use_default_for_none: bool = False) -> bool:
    if value is None:
        if use_default_for_none:
            return False
        raise ValueError(f"{field_name} cannot be None")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be a boolean-like value")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{field_name} must be a boolean-like value")


def coerce_int(value: Any, *, field_name: str = 'value', use_default_for_none: bool = False) -> int:
    if value is None:
        if use_default_for_none:
            return 0
        raise ValueError(f"{field_name} cannot be None")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer-like value")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '':
            return 0
        try:
            parsed = float(stripped) if any(ch in stripped for ch in '.eE') else int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer-like value") from exc
        if isinstance(parsed, float):
            if not parsed.is_integer():
                raise ValueError(f"{field_name} must be an integer-like value")
            return int(parsed)
        return parsed
    raise ValueError(f"{field_name} must be an integer-like value")


def coerce_float(value: Any, *, field_name: str = 'value', use_default_for_none: bool = False) -> float:
    if value is None:
        if use_default_for_none:
            return 0.0
        raise ValueError(f"{field_name} cannot be None")
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == '':
            return 0.0
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float-like value") from exc
    raise ValueError(f"{field_name} must be a float-like value")


def coerce_str(value: Any, *, field_name: str = 'value', use_default_for_none: bool = False) -> str:
    if value is None:
        if use_default_for_none:
            return ''
        raise ValueError(f"{field_name} cannot be None")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def coerce_primitive_by_type(
    value: Any,
    value_type: str,
    *,
    field_name: str = 'value',
    use_default_for_none: bool = False,
) -> Any:
    if value_type == 'bool':
        return coerce_bool(value, field_name=field_name, use_default_for_none=use_default_for_none)
    if value_type == 'int':
        return coerce_int(value, field_name=field_name, use_default_for_none=use_default_for_none)
    if value_type == 'float':
        return coerce_float(value, field_name=field_name, use_default_for_none=use_default_for_none)
    if value_type == 'str':
        return coerce_str(value, field_name=field_name, use_default_for_none=use_default_for_none)
    raise ValueError(f"Unsupported primitive type '{value_type}'")
