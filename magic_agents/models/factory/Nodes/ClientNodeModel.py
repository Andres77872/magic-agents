from typing import Optional, Literal, Any

from pydantic import model_validator

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
    """
    Client node model - accepts various field names from JSON.
    The JSON definition is the source of truth.
    """
    engine: Optional[str] = None  # Allow any engine string, not just predefined ones
    provider: Optional[str] = None  # alias for engine
    api_info: Optional[dict | str] = None
    config: Optional[dict | str] = None  # alias for api_info
    credentials: Optional[dict | str] = None  # alias for api_info
    model: Optional[str] = None
    model_name: Optional[str] = None  # alias for model

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.engine is None and self.provider is not None:
            self.engine = self.provider
        if self.api_info is None:
            if self.config is not None:
                self.api_info = self.config
            elif self.credentials is not None:
                self.api_info = self.credentials
        if self.model is None and self.model_name is not None:
            self.model = self.model_name
        return self
