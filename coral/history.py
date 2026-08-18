from sqlmodel import SQLModel, Field, create_engine
from pydantic_ai import ModelMessage
from pydantic import TypeAdapter
from typing import *
from datetime import datetime, timezone

from .utils import now

import logging
logger = logging.getLogger(__name__)

adapter = TypeAdapter(ModelMessage)

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(index=True)

    data: str

    created_at: datetime = Field(default_factory = now)

class Moderation(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    kind: str                                   # 'ban' | 'timeout'
    until: Optional[datetime] = None            # None -> permanent ban
    reason: Optional[str] = None
    created_at: datetime = Field(default_factory = now)

class Automation(SQLModel, table=True):
    name: str = Field(primary_key=True)
    event: str                            # discord.py event name e.g. `member_join`
    action: str                           # 'prompt' | 'code'
    payload: str
    channel_id: Optional[int] = None
    author_id: int
    guild_id: Optional[int] = None
    created_at: datetime = Field(default_factory = now)


def init_db(db_uri: str):
    logger.debug("Creating database engine @ %s", db_uri)
    engine = create_engine(db_uri)
    SQLModel.metadata.create_all(engine)
    return engine