from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError
from apscheduler.job import Job

from datetime import datetime
import logging, typing
from typing import Optional

if typing.TYPE_CHECKING:
    from .bot import CoralBot

logger = logging.getLogger(__name__)

REMINDERS_DB = 'sqlite:////workspace/reminders.db'

_bot: 'CoralBot' = None

async def _fire(channel_id: int, author_id: int, guild_id: int, prompt: str):
    bot = _bot

    if bot is None: return

    channel = bot.get_channel(channel_id)
    guild = bot.get_guild(guild_id) if guild_id else None
    member = guild.get_member(author_id) if guild else None

    if not (channel and member):
        logger.warning("Reminder skipped: channel/member MISSING! (%s/%s)", channel_id, author_id)
        return

    allowed, tier = bot._may_chat(member)
    if not allowed: return

    await bot._respond_in_channel(
        channel,
        [f"Scheduled task (set by {member.name}): {prompt}"],
        tier = tier, author = member, message = None, footer = False,
    )

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
        self, channel_id, author_id, guild_id, prompt, *,
        at: Optional[datetime] = None,
        every_x_seconds: Optional[int] = None,
        cron: Optional[str] = None,
    ) -> str:
        chosen = [x for x in (at, every_x_seconds, cron) if x is not None]
        if len(chosen) != 1:
            raise ValueError("Provide exactly one of `at`, `every_x_seconds`, or `cron`. Not more than one. Not less than one. No. Don't do that.")

        if at is not None:
            trigger = DateTrigger(run_date=at)
        elif every_x_seconds is not None:
            trigger = IntervalTrigger(seconds=every_x_seconds)
        else:
            trigger = CronTrigger.from_crontab(cron)

        job = self.scheduler.add_job(
            _fire, args = [ channel_id, author_id, guild_id, prompt ],
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
                'channel_id': j.args[0],
                'prompt': j.args[3],
                'author_id': j.args[1],
            }
            for j in jobs
        ]