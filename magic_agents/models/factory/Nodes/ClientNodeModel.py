from typing import Optional

from magic_llm.engine import (EngineOpenAI,
                              EngineAnthropic,
                              EngineGoogle,
                              EngineAzure,
                              EngineAmazon,
                              EngineCohere,
                              EngineCloudFlare)
from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class ClientEngineAvailableModel(BaseModel):
    OPENAI = EngineOpenAI.engine
    ANTHROPIC = EngineAnthropic.engine
    GOOGLE = EngineGoogle.engine
    AZURE = EngineAzure.engine
    AMAZON = EngineAmazon.engine
    COHERE = EngineCohere.engine
    CLOUDFLARE = EngineCloudFlare.engine


class ClientDataModel(BaseModel):
    engine: ClientEngineAvailableModel
    api_key: Optional[str]
    base_url: Optional[str]
    model: str


class ClientNodeModel(BaseNodeModel):
    data: ClientDataModel
