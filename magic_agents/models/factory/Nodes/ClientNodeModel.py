from typing import Optional, Literal

from magic_llm.engine import (EngineOpenAI,
                              EngineAnthropic,
                              EngineGoogle,
                              EngineAzure,
                              EngineAmazon,
                              EngineCohere,
                              EngineCloudFlare)

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel

ModelClientAvailableType = Literal[
    EngineOpenAI.engine,
    EngineAnthropic.engine,
    EngineGoogle.engine,
    EngineAzure.engine,
    EngineAmazon.engine,
    EngineCohere.engine,
    EngineCloudFlare.engine
]


class ClientEngineAvailableModel:
    OPENAI = EngineOpenAI.engine
    ANTHROPIC = EngineAnthropic.engine
    GOOGLE = EngineGoogle.engine
    AZURE = EngineAzure.engine
    AMAZON = EngineAmazon.engine
    COHERE = EngineCohere.engine
    CLOUDFLARE = EngineCloudFlare.engine


class ClientNodeModel(BaseNodeModel):
    engine: ModelClientAvailableType
    api_info: Optional[dict | str]
    model: str
