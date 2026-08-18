import discord
import time, os, json, asyncio
from pydantic_ai import Agent, ToolCallPart, UserContent, AgentRun
from pydantic_ai.models import Model
import pydantic_ai.messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, RunCancelled
from sqlalchemy import Engine
from sqlmodel import Session, select, delete
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Sequence, TypedDict, Literal
from collections import deque

from .config import Config, Tier
from . import prompts, utils, reminders, moderation
from .agent import Deps, add_message_details
from .history import Message, adapter

logger = logging.getLogger(__name__)

class ActiveRun(TypedDict):
    run: AgentRun
    message: discord.Message
    owner_id: int

class CoralBot(discord.Client):
    def __init__(self, config: Config, agent: Agent, model: Model | str, engine: Engine, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info("Agent %s initializing...", str(model))

        logger.debug(str(config))
        logger.debug(str(agent))
        logger.debug(str(model))

        self.config = config
        self.agent  = agent
        self.model  = model
        self.engine = engine

        self.channel_locks: dict[int, asyncio.Lock] = {}    # channel id : lock
        self.active_runs: dict[int, ActiveRun] = {}         # channel id : {'run': AgentRun, 'message': latest message}
        self.rate_hits: dict[int, deque] = {}               # user    id : timestamps

        self.scheduler = reminders.Scheduler(self)

        self.tree = discord.app_commands.CommandTree(self)

        @self.tree.context_menu(name="Ask Me")
        async def ask_me(interaction: discord.Interaction, message: discord.Message):

            allowed, tier = self._may_chat(interaction.user)
            if not allowed:
                return

            await interaction.response.defer(thinking=True, ephemeral=True)
            await self._handle_message(message, [f"Triggered by {interaction.user.mention}"], tier=tier, author=interaction.user)
            await interaction.followup.send("I have responded in chat!", ephemeral=True)

        @self.tree.command()
        async def guide(interaction: discord.Interaction, prompt: str, priority: Literal['asap', 'when_idle'] = 'asap'):
            """
            When I'm midway through a long task, you can use this command to give me additional information to keep in mind while I'm working.

            Args:
                prompt: What you want to tell me.
                priority: Set to `asap` to let me know as soon as possible, or `when_idle` to let me know once I've finished my task.
            """
            blocked, _ = moderation.is_blocked(self.engine, interaction.user.id)
            allowed, _ = self._may_chat(interaction.user)
            if blocked or not allowed:
                return await interaction.response.send_message("You can't do that!", ephemeral=True)
            entry = self.active_runs.get(interaction.channel.id)
            if priority not in ('asap', 'when_idle'): priority = 'asap'
            if entry and entry['run'].enqueue(f"{interaction.user.name} (add-on to previous request): {prompt}", priority=priority):
                await interaction.response.send_message(f"> {prompt}\n> -# by {interaction.user.mention}\n\nThanks, I've seen your request and I'm now including it.")
            else:
                await interaction.response.send_message("There's nothing to guide me on right now. Try asking me to start a longer task, perhaps.", ephemeral=True)

        @self.tree.command()
        async def interrupt(interaction: discord.Interaction):
            """
            Stop what I'm doing immediately. If you want to tell me something, it's better to use /guide instead.
            """
            blocked, _ = moderation.is_blocked(self.engine, interaction.user.id)
            allowed, _ = self._may_chat(interaction.user)
            if blocked or not allowed:
                return await interaction.response.send_message("You can't do that!", ephemeral=True)
            entry = self.active_runs.get(interaction.channel.id)
            if entry:
                entry['run'].cancel()
                await interaction.response.send_message("🔴 Interrupting...")
            else:
                await interaction.response.send_message("I'm not doing anything to interrupt!", ephemeral=True)

    def _role_ids(self, user) -> list[int]:
        return [role.id for role in getattr(user, 'roles', [])]

    def _may_chat(self, user):
        tier = self.config.resolve_tier(user.id, self._role_ids(user))
        return tier.allow_chat, tier

    def _channel_lock(self, channel_id: int) -> asyncio.Lock:
        lock = self.channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self.channel_locks[channel_id] = lock
        return lock

    def _rate_limited(self, user_id: int, tier: Tier) -> bool:
        limit = tier.parsed_ratelimit()

        if limit is None:
            return False

        count, window = limit

        now = time.time()
        hits = self.rate_hits.setdefault(user_id, deque())

        while hits and hits[0] <= now - window:
            hits.popleft()

        if len(hits) >= count:
            return True

        hits.append(now)
        return False

    async def enqueue_guide(self, message: discord.Message, entry: ActiveRun, priority: Literal['asap', 'when_idle'] = 'asap') -> bool:
        text = utils.clean(message).removeprefix(self.config.DISCORD_PREFIX)
        if priority not in ('asap', 'when_idle'): priority = 'asap'
        if not entry['run'].enqueue(f"{message.author.name} (add-on to previous request): {text}", priority=priority):
            return False
        entry['message'] = message
        try:
            await message.add_reaction('👍')
        except Exception: pass
        return True

    async def try_guide(self, message: discord.Message) -> bool:
        entry = self.active_runs.get(message.channel.id)
        if not entry or not message.reference or not message.reference.message_id:
            return False

        ref_msg = message.reference.resolved

        if ref_msg is None:
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
            except Exception:
                ref_msg = None

        if not ref_msg or not isinstance(ref_msg, discord.Message) or ref_msg.author.id != self.user.id:
            return False

        return await self.enqueue_guide(message, entry)

    async def guide_same_author(self, message: discord.Message) -> bool:
        entry = self.active_runs.get(message.channel.id)
        if not entry or entry.get('owner_id') != message.author.id:
            return False

        return await self.enqueue_guide(message, entry)

    async def on_ready(self):
        logger.info("Logged in as %s",  self.user.name)
        await self.tree.sync()

        resume = Path('/workspace/.pending')
        if resume.exists():
            try:
                data = json.loads(resume.read_text())
            except Exception: data = None

            resume.unlink()

            if data:
                channel = self.get_channel(data['channel_id'])
                guild: Optional[discord.Guild] = getattr(channel, 'guild', None)
                member = guild.get_member(data['author_id']) if (guild and data.get('author_id')) else None
                if channel and member:
                    allowed, tier = self._may_chat(member)
                    if allowed:
                        await self._respond_in_channel(
                            channel,
                            ["System: Reboot complete. Continue any unfinished task that was asked, or alert the user that you have completed the reboot."],
                            tier = tier, author = member, message = None, footer = False,
                        )

        # send the agent the reboot thing before any schedules
        await self.scheduler.start()



    async def on_message(self, message: discord.Message):

        if message.author == self.user or message.author.bot: # let's not allow bots to talk to Coral
            return

        blocked, reason = moderation.is_blocked(self.engine, message.author.id)
        if blocked: return

        allowed, tier = self._may_chat(message.author)
        if not allowed:
            return

        if await self.try_guide(message):
            return

        if (self.user not in message.mentions and not message.content.startswith(self.config.DISCORD_PREFIX)):
            return

        if await self.guide_same_author(message):
            return

        if self._rate_limited(message.author.id, tier):
            try:
                await message.add_reaction('⏰')
            except Exception:
                pass

            return
        
        return await self._handle_message(message, tier=tier)

    async def _handle_message(self, message: discord.Message, extra_logs: list[str] | None = None, tier: Tier = None, author: discord.Member | discord.User = None):
        if message.author == self.user:
            return

        logger.debug("Received message from %s: %s", message.author.name, (message.content[:50] + '...') if len(message.content) > 50 else message.content)

        author = author or message.author

        parts = [
            add_message_details(message),
            message.author.name + ": " + utils.clean(message).removeprefix(self.config.DISCORD_PREFIX),
        ]

        await self._respond_in_channel(message.channel, parts, tier=tier, author=author, message=message, extra_logs=extra_logs, footer=True)

    async def _respond_in_channel(
        self,
        channel: discord.abc.Messageable,
        prompt_parts: Sequence[UserContent],
        *,
        tier: Tier,
        author: discord.Member | discord.User,
        message: discord.Message = None,
        extra_logs: list[str] = None,
        footer: bool = True
    ):

        reply_target = message
        
        async with channel.typing():
            start = time.time()

            try:
                async with self._channel_lock(channel.id):
                    with Session(self.engine) as session:
                        stmt = select(Message).where(
                            Message.channel_id == channel.id
                        ).order_by(Message.created_at.desc())

                        messages = session.exec(stmt).all()

                        history = [adapter.validate_json(msg.data) for msg in reversed(messages)]

                    guild = getattr(channel, 'guild', None)

                    deps = Deps(
                        message=message,
                        client=self,
                        config=self.config,
                        model=self.model,
                        tier=tier, 
                        scheduler=self.scheduler,
                        author_id=getattr(author, 'id', None),
                        guild_id=getattr(guild, 'id', None)
                    )

                    try:
                        async with self.agent.iter(
                            prompt_parts,
                            deps            = deps,
                            model           = self.model,
                            message_history = history,
                        ) as run:
                            self.active_runs[channel.id] = {'run': run, 'message': message, 'owner_id': getattr(author, 'id', None)}
                            async for node in run: ...

                        result = run.result
                    finally:
                        entry = self.active_runs.pop(channel.id, None)
                        if entry: reply_target = entry['message']

                    response = result.output

                    with Session(self.engine) as session:
                        session.exec(delete(Message).where(Message.channel_id == channel.id))

                        for new_msg in result.all_messages():
                            session.add(Message(
                                channel_id = channel.id,
                                data = adapter.dump_json(new_msg).decode(),
                            ))
                        session.commit()

                    if deps.reboot_requested:
                        deps.reboot_requested = False
                        import sys
                        sys.exit(0)

            except RunCancelled as e:
                result = None
                response = "## 🔴 Interrupted.\n\nAgent interrupted by user."

                try:
                    with Session(self.engine) as session:
                        session.exec(delete(Message).where(Message.channel_id == channel.id))
                        for new_msg in e.all_messages():
                            session.add(Message(channel_id=channel.id, data=adapter.dump_json(new_msg).decode()))
                        session.commit()
                except Exception as e:
                    import traceback
                    logger.error("Failed to persist history after interrupt: %s", traceback.format_exc(), exc_info=e)

            except (ModelHTTPError, ModelAPIError) as e:
                result = None
                response = f"""
## 🚨 Error

An **upstream API error** occured.

**Error Details:**
{e.message}
"""
                import traceback
                logger.error(traceback.format_exc(), exc_info=e)
            
            except Exception as e:
                import traceback
                result = None
                response = f"""
## 🚨 Error

A **critical exception** occured in my main thread.

**Error Details:**
```
{traceback.format_exc(limit=2).replace(os.path.dirname(__file__), '/')}
```
                """
                logger.error(traceback.format_exc(), exc_info=e)
            finally:
                info = extra_logs.copy() if extra_logs else []

                end = time.time()
                taken = round(end - start, 1)
                if taken > 5:
                    info.append(f"Time taken: {taken}s")
                
                if footer and result:
                    new_msgs = result.new_messages()
                    tools: list[ToolCallPart] = []
                    for msg in new_msgs:
                        if getattr(msg, 'tool_calls', False):
                            tools.extend(msg.tool_calls)

                    if tools:
                        info.append(f"Tools called: {len(tools)} - {', '.join(tool.tool_name for tool in tools)}")

                if footer and info:
                    response += f"\n\n" + '\n'.join(f"-# {msg}" for msg in info)

        guild = getattr(channel, 'guild', None)
        
        allow_everyone = tier is not None and tier.allow_ping_everyone

        response, allowed_roles = utils.sanitize_role_mentions(
            response, guild, channel, author, allow_everyone
        )

        if not allow_everyone:
            response = utils.neutralize_mass_mentions(response)

        allowed_mentions = discord.AllowedMentions(
            everyone = allow_everyone,
            users    = True,
            roles    = allowed_roles,
        )

        chunks = utils.chunk_string(response)

        first = chunks.pop(0)
        await (reply_target.reply if reply_target else channel.send)(first, allowed_mentions=allowed_mentions)

        for chunk in chunks:
            await channel.send(chunk, allowed_mentions=allowed_mentions)

    async def on_error(self, event_method: str, /, *args, **kwargs):
        import traceback, os
        response = f"""
## 🚨 Error

A **critical exception** occured in my main thread.

**Error Details:**
```
{traceback.format_exc(limit=2).replace(os.path.dirname(__file__), '/')}
```
        """

        logger.error(traceback.format_exc())

        if event_method == 'on_message' and args:
            message: discord.Message = args[0]
            
            try:
                await message.reply(response)
            except:
                try:
                    await message.channel.send(response)
                except:
                    pass

        return await super().on_error(event_method, *args, **kwargs)