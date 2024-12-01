from typing import Optional

from pydantic import BaseModel


class ModelAgentRunLog(BaseModel):
    id_chat: Optional[int | str] = None
    id_thread: Optional[int | str] = None
    id_app: Optional[int | str] = None
    id_user: Optional[int | str] = None
    agent: Optional[str] = None
