import discord
import time
from pydantic_ai import Agent, ToolCallPart
from sqlalchemy import Engine
from sqlmodel import Session, select

from .config import Config
from . import prompts, utils
from .agent import Deps
from .history import Message, adapter

class CoralBot(discord.Client):
    def __init__(self, config: Config, agent: Agent, model, engine: Engine, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.agent  = agent
        self.model  = model
        self.engine = engine

        self.tree = discord.app_commands.CommandTree(self)

        @self.tree.context_menu(name="Ask Me")
        async def ask_me(interaction: discord.Interaction, message: discord.Message):

            if self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS:
                if not (
                    interaction.user.id in self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS
                    or
                    any(
                        role.id in self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS
                        for role in getattr(interaction.user, 'roles', [])
                    )
                ):
                    return
            # if no config set, allow everyone

            await interaction.response.defer(thinking=True, ephemeral=True)
            await self._handle_message(message, [f"Triggered by {interaction.user.mention}"])
            await interaction.followup.send("I have responded in chat!", ephemeral=True)

    async def on_ready(self):
        print(f"Logged in as {self.user.name}.")
        await self.tree.sync()

    async def on_message(self, message: discord.Message):      
        if self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS:
            if not (
                message.author.id in self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS
                or
                any(
                    role.id in self.config.DISCORD_ALLOWED_USER_OR_ROLE_IDS
                    for role in getattr(message.author, 'roles', [])
                )
            ):
                return
        # if no config set, allow everyone

        if (
            self.user not in message.mentions
            and
            not message.content.startswith(self.config.DISCORD_PREFIX)
        ):
            return
        
        return await self._handle_message(message)

    async def _handle_message(self, message: discord.Message, extra_logs: list[str] | None = None):
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
            
            try:
                result = await self.agent.run(
                    user_prompt     = utils.clean(message).removeprefix(self.config.DISCORD_PREFIX),
                    deps            = Deps(message=message, client=self, config=self.config),
                    model           = self.model,
                    message_history = history,
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