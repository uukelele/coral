import discord
import time
from pydantic_ai import Agent, ToolCallPart
from pydantic_ai.models import Model
import pydantic_ai.messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from sqlalchemy import Engine
from sqlmodel import Session, select
from datetime import datetime

from .config import Config
from . import prompts, utils
from .agent import Deps
from .history import Message, adapter

class CoralBot(discord.Client):
    def __init__(self, config: Config, agent: Agent, model: Model | str, engine: Engine, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.agent  = agent
        self.model  = model
        self.engine = engine

        self.tree = discord.app_commands.CommandTree(self)

        @self.tree.context_menu(name="Ask Me")
        async def ask_me(interaction: discord.Interaction, message: discord.Message):

            # Permissions are based on the user who triggered the command, not the
            # author of the message being asked about. So a privileged user can use
            # "Ask Me" on a lower-ranked user's message and still get a response.
            allowed, tier = self._may_chat(interaction.user)
            if not allowed:
                return

            await interaction.response.defer(thinking=True, ephemeral=True)
            await self._handle_message(message, [f"Triggered by {interaction.user.mention}"], tier=tier)
            await interaction.followup.send("I have responded in chat!", ephemeral=True)

    def _role_ids(self, user) -> list[int]:
        return [role.id for role in getattr(user, 'roles', [])]

    def _legacy_allowed(self, user) -> bool:
        # Backward-compatible allow-list used when `tiers` is not configured.
        allowed = self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS
        if not allowed:
            # if no config set, allow everyone
            return True
        return user.id in allowed or any(rid in allowed for rid in self._role_ids(user))

    def _may_chat(self, user):
        """
        Returns (allowed, tier). `tier` is None in legacy mode (no tiers configured),
        in which case tool access is unrestricted for backward compatibility.
        """
        tier = self.config.resolve_tier(user.id, self._role_ids(user))
        if tier is None:
            return self._legacy_allowed(user), None
        return tier.allow_chat, tier

    async def on_ready(self):
        print(f"Logged in as {self.user.name}.")
        await self.tree.sync()

    async def on_message(self, message: discord.Message):
        allowed, tier = self._may_chat(message.author)
        if not allowed:
            return

        if (
            self.user not in message.mentions
            and
            not message.content.startswith(self.config.DISCORD_PREFIX)
        ):
            return
        
        return await self._handle_message(message, tier=tier)

    async def _handle_message(self, message: discord.Message, extra_logs: list[str] | None = None, tier=None):
        if message.author == self.user:
            return
        
        async with message.channel.typing():
            start = time.time()

            with Session(self.engine) as session:
                LIMIT = 50

                stmt = select(Message).where(
                    Message.channel_id == message.channel.id
                ).order_by(Message.created_at.desc()).limit(LIMIT + 20)

                messages = session.exec(stmt).all()

                history = [adapter.validate_json(msg.data) for msg in reversed(messages)]

                history = history[-LIMIT:]

                while history:
                    first = history[0]
                    is_orphan = False

                    for part in getattr(first, 'parts', []):
                        if part.__class__.__name__ in ['ToolReturnPart', 'ToolCallPart']:
                            is_orphan = True
                            break

                    if is_orphan:
                        history.pop(0)
                    else:
                        break

                SUMMARIZE_LIMIT = LIMIT // 2
                split_found = False
                for i in range(max(0, len(history) - SUMMARIZE_LIMIT), len(history) + 1):

                    if i < len(history) and any(part.__class__.__name__ in ['ToolReturnPart', 'ToolCallPart'] for part in getattr(history[i], 'parts', [])):
                        continue

                    if i > 0 and any(part.__class__.__name__ in ['ToolReturnPart', 'ToolCallPart'] for part in getattr(history[i-1], 'parts', [])):
                        continue

                    rest = history[:i]
                    immediate_context = history[i:]
                    split_found = True
                    break

                if not split_found:
                    if len(history) > SUMMARIZE_LIMIT:
                        rest = history[:-SUMMARIZE_LIMIT]
                        immediate_context = history[-SUMMARIZE_LIMIT:]
                    else:
                        rest = []
                        immediate_context = history
            
            try:
                if rest:
                    summary = await self.agent.run(
                        user_prompt     = prompts.SUMMARIZATION_PROMPT,
                        model           = self.model,
                        message_history = rest,
                        deps            = Deps(is_summary=True, model=self.model)
                    )

                    summary_content = prompts.SUMMARIZED_TEXT + summary.output

                    sum_msg = pydantic_ai.messages.ModelRequest(
                        parts=[
                            pydantic_ai.messages.UserPromptPart(content=summary_content)
                        ],
                    )

                    immediate_context = [
                        sum_msg,
                        *immediate_context
                    ]

                    with Session(self.engine) as session:
                        old_records = session.exec(select(Message).where(
                            Message.channel_id == message.channel.id
                        ).order_by(Message.created_at.asc()).limit(len(rest))).all()

                        [session.delete(r) for r in old_records]

                        summary_record = Message(
                            channel_id = message.channel.id,
                            data = adapter.dump_json(sum_msg).decode(),
                            created_at = datetime.fromtimestamp(0),
                        )

                        session.add(summary_record)

                        session.commit()


                result = await self.agent.run(
                    user_prompt     = message.author.display_name + ": " + utils.clean(message).removeprefix(self.config.DISCORD_PREFIX),
                    deps            = Deps(message=message, client=self, config=self.config, model=self.model, tier=tier),
                    model           = self.model,
                    message_history = immediate_context,
                )
                response = result.output

                with Session(self.engine) as session:
                    for new_msg in result.new_messages():
                        record = Message(
                            channel_id = message.channel.id,
                            data = adapter.dump_json(new_msg).decode()
                        )
                        session.add(record)
                    session.commit()

            except (ModelHTTPError, ModelAPIError) as e:
                result = None
                response = f"""
## 🚨 Error

An **upstream API error** occured.

**Error Details:**
{e.message}
"""
            
            except Exception as e:
                import traceback
                result = None
                response = f"""
## 🚨 Error

A **critical exception** occured in my main thread.

**Error Details:**
```
{traceback.format_exc(limit=2)}
```
                """
                traceback.print_exc()
            finally:
                info = extra_logs.copy() if extra_logs else []

                end = time.time()
                taken = round(end - start, 1)
                if taken > 5:
                    info.append(f"Time taken: {taken}s")
                
                if result:
                    new_msgs = result.new_messages()
                    tools: list[ToolCallPart] = []
                    for msg in new_msgs:
                        if getattr(msg, 'tool_calls', False):
                            tools.extend(msg.tool_calls)

                    if tools:
                        info.append(f"Tools called: {len(tools)} - {', '.join(tool.tool_name for tool in tools)}")

                response += f"\n\n" + '\n'.join(f"-# {msg}" for msg in info)

        
        chunks = utils.chunk_string(response)

        first = chunks.pop(0)
        await message.reply(first)

        for chunk in chunks:
            await message.channel.send(chunk)

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