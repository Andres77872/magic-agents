from typing import Any, Literal

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel
from magic_agents.util.primitive_coercion import coerce_primitive_by_type


class ConstantNodeModel(BaseNodeModel):
    """Constant node model for primitive source values."""

    value_type: Literal['int', 'bool', 'str', 'float'] = 'str'
    value: Any = None

    @model_validator(mode='after')
    def resolve_constant_value(self):
        """Coerce the explicit value to the declared primitive type."""
        self.value = coerce_primitive_by_type(
            self.value,
            self.value_type,
            field_name='value',
            use_default_for_none=True,
        )
        return self
