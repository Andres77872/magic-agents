"""Strict self-contained, inert prompt skill configuration."""
import json
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .BaseNodeModel import BaseNodeModel


def props_json_bytes(props: dict) -> int:
    return len(json.dumps(props, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'))


def validate_props(value: Any) -> dict:
    if type(value) is not dict:
        raise ValueError('props must be a JSON object')
    def check(item, depth, ancestors):
        if item is None or type(item) in (bool, str, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (dict, list):
            raise ValueError('props contains a non-JSON value')
        if id(item) in ancestors:
            raise ValueError('props must be acyclic')
        if depth > 5:
            raise ValueError('props container depth exceeds 5')
        if type(item) is dict and any(type(key) is not str for key in item):
            raise ValueError('props keys must be strings')
        for child in item.values() if type(item) is dict else item:
            check(child, depth + 1, ancestors | {id(item)})
    check(value, 1, set())
    if props_json_bytes(value) > 4096:
        raise ValueError('props exceeds 4096 serialized UTF-8 bytes')
    # A fresh graph-owned JSON value avoids caller-owned nested containers.
    return json.loads(json.dumps(value, allow_nan=False))


class SkillPromptModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9-]+$')
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1024)
    prompt: str = Field(min_length=1, max_length=16384, repr=False)
    props: dict[str, Any] = Field(default_factory=dict, repr=False)
    enabled: bool = True

    @field_validator('name', 'description', 'prompt')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('skill text must be nonblank')
        return value

    _props = field_validator('props', mode='before')(validate_props)


class SkillsNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    schema_version: Literal[1]
    skills: list[SkillPromptModel] = Field(min_length=1, max_length=16, repr=False)

    @field_validator('schema_version', mode='before')
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError('schema_version must be integer 1')
        return value

    @model_validator(mode='after')
    def aggregate_bounds(self):
        if len({entry.id for entry in self.skills}) != len(self.skills):
            raise ValueError('skill IDs must be unique within the node')
        if sum(len(entry.prompt) for entry in self.skills if entry.enabled) > 32768:
            raise ValueError('enabled prompt total exceeds 32768 characters')
        if sum(props_json_bytes(entry.props) for entry in self.skills) > 16384:
            raise ValueError('node props total exceeds 16384 serialized UTF-8 bytes')
        return self
