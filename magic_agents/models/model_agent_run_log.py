from typing import Optional

from pydantic import BaseModel

class ModelAgentRunLog(BaseModel):
    id_chat: str
    id_app: Optional[str] = None
    id_user: Optional[int] = None
    chat_system: Optional[str] = None
    chat_title: Optional[str] = None
    agent: Optional[str] = None