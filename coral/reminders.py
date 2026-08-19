from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError
from apscheduler.job import Job

from datetime import datetime
import logging, typing
from typing import Optional, Literal
import discord
from pathlib import Path

from . import moderation, utils

if typing.TYPE_CHECKING:
    from .bot import CoralBot

logger = logging.getLogger(__name__)

REMINDERS_DB = 'sqlite:////workspace/reminders.db'

_bot: 'CoralBot' = None

async def fire_automation(name: str, bot: 'CoralBot', action: Literal['code', 'prompt'], payload: str, channel_id: Optional[int], author_id: int, guild_id: Optional[int], event_args: tuple):
    if moderation.is_blocked(bot.engine, author_id)[0]: return

    channel = bot.get_channel(channel_id) if channel_id else None
    guild   = bot.get_guild(guild_id) if guild_id else None
    member  = guild.get_member(author_id) if guild else None

    match action:
        case 'prompt':
            if not (channel and member): return

            allowed, tier = bot._may_chat(member)
            if not allowed: return

            label = 'Scheduled task' if event_args is None else 'Event automation'
            await bot._respond_in_channel(
                channel,
                [f"{label} (set by {member.name}): {payload}"],
                tier = tier, author = member, message = None, footer = False,
            )
        case 'code':
            code = payload
            try:
                p = Path(payload)
                if p.is_file(): code = p.read_text()
            except OSError: ...

            res = await utils.run_code(
                code,
                'async def main(event, discord, client):', (event_args, discord, bot),
                timeout = 60,
            )
            if res.get('stderr') or (isinstance(res.get('result'), str) and 'Traceback' in res['result']):
                logger.warning("Automation %s code error: %s\n%s", name, res.get('result'), res.get('stderr'))

            log = Path(f'/workspace/automations/{name}.log')
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as lf: lf.write(f"Automation running at {utils.now()}: {res}\n")
            except OSError: ... 

async def _fire(name: str, action: Literal['code', 'prompt'], payload: str, channel_id: int, author_id: int, guild_id: int):
    if _bot: await fire_automation(name, _bot, action, payload, channel_id, author_id, guild_id, event_args=None)

class Scheduler:
    def __init__(self, bot, db_url: str = REMINDERS_DB):
        global _bot
        _bot = bot
        self.bot = bot
        self.scheduler = AsyncIOScheduler(
            jobstores = { 'default': SQLAlchemyJobStore(url=db_url) },
            timezone = 'UTC',
        )

    async def start(self):
        self.scheduler.start()

    def add(
        self, action: Literal['prompt', 'code'], payload: str, channel_id: Optional[int], author_id: int, guild_id: Optional[int], *,
        at: Optional[datetime] = None,
        every_x_seconds: Optional[int] = None,
        cron: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        chosen = [x for x in (at, every_x_seconds, cron) if x]
        if len(chosen) != 1:
            raise ValueError("Provide exactly one of `at`, `every_x_seconds`, or `cron`. Not more than one. Not less than one. No. Don't do that.")

        if at is not None:
            trigger = DateTrigger(run_date=at)
        elif every_x_seconds is not None:
            trigger = IntervalTrigger(seconds=every_x_seconds)
        else:
            trigger = CronTrigger.from_crontab(cron)

        job = self.scheduler.add_job(
            _fire, args = [ name, action, payload, channel_id, author_id, guild_id ], id=name,
            trigger=trigger, misfire_grace_time=3600, coalesce=True,
        )
        return job.id

    def cancel(self, reminder_id: str) -> bool:
        try:
            self.scheduler.remove_job(reminder_id)
            return True
        except JobLookupError:
            return False

    def list_all(self) -> list[dict]:
        jobs: list[Job] = self.scheduler.get_jobs()
        return [
            {
                'id': j.id,
                'next_run': str(j.next_run_time),
                'action': j.args[1],
                'payload': j.args[2],
                'channel_id': j.args[3],
                'author_id': j.args[4],
                'guild_id': j.args[5],
            }
            for j in jobs
        ]