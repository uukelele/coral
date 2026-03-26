from sqlmodel import SQLModel, Field, create_engine
from pydantic_ai import ModelMessage
from pydantic import TypeAdapter
from typing import *
from datetime import datetime, timezone

adapter = TypeAdapter(ModelMessage)

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(index=True)

    data: str

    created_at: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))

def init_db(db_uri: str):
    engine = create_engine(db_uri)
    SQLModel.metadata.create_all(engine)
    return engine