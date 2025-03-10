from typing import Optional

from pydantic import BaseModel


class EdgeNodeModel(BaseModel):
    id: str = "chat"
    source: str
    target: Optional[str] = None
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
