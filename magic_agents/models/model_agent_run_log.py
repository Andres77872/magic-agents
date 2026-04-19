from typing import Optional, Any

from pydantic import BaseModel


class ModelAgentRunLog(BaseModel):
    id_chat: Optional[int | str] = None
    id_thread: Optional[int | str] = None
    id_app: Optional[int | str] = None
    id_user: Optional[int | str] = None
    agent: Optional[str] = None
    flow_state: Optional[dict[str, Any]] = None  # Per-flow volatile state (runtime-only, never persisted)
