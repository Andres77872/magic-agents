from typing import Literal, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

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
    'constant',
    'void',
    'loop',
    'inner',
    'conditional',
    'python_exec',
    'mcp'
]


class ModelAgentFlowTypesModel:
    CHAT = 'chat'
    LLM = 'llm'
    END = 'end'
    TEXT = 'text'
    CONSTANT = 'constant'
    USER_INPUT = 'user_input'
    PARSER = 'parser'
    FETCH = 'fetch'
    CLIENT = 'client'
    SEND_MESSAGE = 'send_message'
    VOID = 'void'
    LOOP = 'loop'
    INNER = 'inner'
    CONDITIONAL = 'conditional'
    PYTHON_EXEC = 'python_exec'
    MCP = 'mcp'


class BaseNodeModel(BaseModel):
    """
    Base model for all node types.
    Backend-authoritative validation: rejects unknown fields to enforce strict JSON contract.
    Frontend must sync TypeScript interfaces to Pydantic models.
    """
    model_config = ConfigDict(extra='forbid')  # Reject unknown fields - backend authoritative
    
    position: Optional[dict[str, int]] = {'x': 0, 'y': 0}
    extra_data: Optional[dict[str, Any]] = Field(default_factory=dict)
