from pydantic import BaseModel


class ModelAgentFlowTypesModel(BaseModel):
    USER_INPUT = 'user_input'
    END = 'end'
    PARSER = 'parser'
    CLIENT = 'client'
    LLM = 'llm'
    FETCH = 'fetch'
    SEND_MESSAGE = 'send_message'


class BaseNodeModel(BaseModel):
    id_model: int | str
    type: ModelAgentFlowTypesModel
