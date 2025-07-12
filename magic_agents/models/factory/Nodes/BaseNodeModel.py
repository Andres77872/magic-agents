from typing import Literal, Optional, Any

from pydantic import BaseModel, Field

ModelAgentFlowType = Literal[
    'user_input',
    'end',
    'parser',
    'client',
    'llm',
    'fetch',
    'send_message',
    'chat',
    'text',
    'void',
    'loop',
    'inner',
    'conditional'
]


class ModelAgentFlowTypesModel:
    CHAT = 'chat'
    LLM = 'llm'
    END = 'end'
    TEXT = 'text'
    USER_INPUT = 'user_input'
    PARSER = 'parser'
    FETCH = 'fetch'
    CLIENT = 'client'
    SEND_MESSAGE = 'send_message'
    VOID = 'void'
    LOOP = 'loop'
    INNER = 'inner'
    CONDITIONAL = 'conditional'


class BaseNodeModel(BaseModel):
    position: Optional[dict[str, int]] = {'x': 0, 'y': 0}
    extra_data: Optional[dict[str, Any]] = Field(default_factory=dict)
