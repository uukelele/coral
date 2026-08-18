from sqlalchemy import Engine
from sqlmodel import Session
from datetime import datetime, timezone, timedelta

from typing import Optional
import logging

from .history import Moderation
from .utils import now

logger = logging.getLogger(__name__)


def is_blocked(engine: Engine, user_id: int) -> tuple[bool, Optional[str]]: # [blocked?, reason]
    with Session(engine) as session:
        row = session.get(Moderation, user_id)
        if row is None: return False, None
        if row.kind == 'ban':
            return True, row.reason or "You are banned from using this bot."
        if row.until is not None:
            until = row.until if row.until.tzinfo else row.until.replace(tzinfo=timezone.utc)
            if until > now():
                return True, row.reason or f"You're timed out. You can use this bot again <t:{int(until.timestamp())}:R>."

        # how did we get here?
        session.delete(row)
        session.commit()
        return False, None

def ban(engine: Engine, user_id: int, reason: Optional[str] = None) -> None:
    with Session(engine) as session:
        row = session.get(Moderation, user_id) or Moderation(user_id=user_id, kind='ban')
        row.kind, row.until, row.reason = 'ban', None, reason
        session.add(row)
        session.commit()
    logger.info("User %s banned from using the bot. Reason: %s", user_id, reason)

def unban(engine: Engine, user_id: int) -> bool:
    with Session(engine) as session:
        row = session.get(Moderation, user_id)
        if row is None or row.kind != 'ban':
            return False
        session.delete(row)
        session.commit()
    logger.info("User %s unbanned.", user_id)
    return True

def timeout(engine: Engine, user_id: int, seconds: int, reason: Optional[str] = None) -> Optional[datetime]: # expiry
    until = now() + timedelta(seconds=seconds)
    with Session(engine) as session:
        row = session.get(Moderation, user_id)
        if row is not None and row.kind == 'ban':
            return None
        row = row or Moderation(user_id=user_id, kind='timeout')
        row.kind, row.until, row.reason = 'timeout', until, reason
        session.add(row)
        session.commit()

    logger.info("User %s timed out until %s. Reason: %s", user_id, until, reason)
    return until

def cancel_timeout(engine: Engine, user_id: int) -> bool:
    with Session(engine) as session:
        row = session.get(Moderation, user_id)
        if row is None or row.kind != 'timeout':
            return False
        session.delete(row)
        session.commit()
    logger.info("Timeout cancelled for user %s.", user_id)
    return True