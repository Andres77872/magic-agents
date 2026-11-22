"""Every authored node model rejects unknown fields (``extra='forbid'``).

The backend is the authority for the graph JSON contract: a typo or a stale
frontend-only field must fail at build instead of being silently ignored.
"""

import importlib
import pkgutil

import pytest
from pydantic import ValidationError

import magic_agents.models.factory.Nodes as node_models
from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalNodeModel
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig

for _module in pkgutil.iter_modules(node_models.__path__):
    importlib.import_module(f"{node_models.__name__}.{_module.name}")


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Minimal valid payload for models with required fields; every other model
# must accept an empty payload.
REQUIRED = {
    "McpNodeModel": {"servers": [{"command": "server"}]},
    "SkillsNodeModel": {"schema_version": 1, "skills": [
        {"id": "s", "name": "S", "description": "d", "prompt": "p", "props": {}}]},
    "ToolNodeModel": {"tool": {"type": "function", "function": {
        "name": "lookup", "description": "Look up", "parameters": {"type": "object", "properties": {}}}}},
    "ConditionalNodeModel": {"condition": "{{ 'a' }}"},
}

MODELS = sorted({*_subclasses(BaseNodeModel), ConditionalNodeModel, MCPServerConfig}, key=lambda c: c.__name__)


def _minimal(model):
    if model is MCPServerConfig:
        return {"command": "server"}
    return dict(REQUIRED.get(model.__name__, {}))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_model_accepts_its_minimal_payload_and_rejects_unknown_fields(model):
    model(**_minimal(model))

    for unknown in ("unknown_field", "measured", "customName"):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model(**_minimal(model), **{unknown: True})


def test_unknown_field_inside_a_nested_server_config_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        node_models.McpNodeModel(servers=[{"command": "server", "unknown": 1}])


def test_parser_rejects_the_legacy_inputs_field():
    with pytest.raises(ValidationError, match="inputs"):
        node_models.ParserNodeModel(text="{{ x }}", inputs={"x": 1})
