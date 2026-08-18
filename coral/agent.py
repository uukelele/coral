import pydantic_ai
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools, Thinking, WebSearch, WebFetch, Hooks
from pydantic_ai_harness import Coder, Memory, ToolGuardrail, GuardrailResult, TieredCompaction, ClearToolResults, SummarizingCompaction
from pydantic_ai_harness.guardrails import ToolCallInfo
from pydantic_ai_harness.memory import FileStore
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models import Model
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import *
from datetime import datetime
import discord
import discord.http
import asyncio
import dataclasses
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import subprocess as sp
from enum import Enum
import httpx
import logging
import json

from .utils import indent
from . import config as libcfg, prompts, reminders, moderation

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Deps:
    model: Model | str
    message: discord.Message = None
    client: discord.Client = None
    config: libcfg.Config = None
    is_message: bool = True
    tier: Optional[libcfg.Tier] = None
    reboot_requested: bool = False
    scheduler: Optional[reminders.Scheduler] = None
    author_id: Optional[int] = None
    guild_id: Optional[int] = None

async def denial_reason(ctx: RunContext[Deps], tool_name: str) -> str | None:
    tier: libcfg.Tier = getattr(ctx.deps, 'tier', None)
    if tier is None:
        return None
    return None if tier.can_use_tool(tool_name) else f"The user does not have permission to use tool {tool_name}."

async def annotate_unavailable_tools(ctx: RunContext[Deps], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    res = []
    for t in tools:
        reason = await denial_reason(ctx, t.name)
        if reason:
            t = dataclasses.replace(t,
                description = f"[UNAVAILABLE] This tool is currently unavailable because: {reason}\nTrying to call it will not work.\n{t.description}"
            )
        res.append(t)
    return res

async def block_unauthorized(ctx: RunContext[Deps], call: ToolCallInfo) -> GuardrailResult:
    logger.info("Agent calling tool %s", call.name)
    reason = await denial_reason(ctx, call.name)
    if reason:
        logger.warning("Agent attempted to call blocked tool %s; denying.", call.name)
        return GuardrailResult.block(f"Tool unavailable: {reason}")
    return GuardrailResult.allow()

hooks = Hooks()

@hooks.on.after_model_request
async def send_text_updates(ctx: RunContext[Deps], *, request_context, response):
    if not ctx.deps.message:
        return response

    parts = response.parts
    if any(isinstance(p, ToolCallPart) for p in parts):
        for part in parts:
            if isinstance(part, TextPart) and part.content.strip():
                await ctx.deps.message.channel.send(part.content)

    return response

agent = Agent(
    deps_type = Deps,
    capabilities = [
        TieredCompaction(
            tiers = [
                ClearToolResults(max_tokens=1, keep_pairs=5),
                SummarizingCompaction(max_messages=1, keep_messages=25)
            ],
            target_tokens = 200_000, # should be enough for 256k models
        ),
        PrepareTools(annotate_unavailable_tools),
        ToolGuardrail(guard=block_unauthorized),
        Thinking(),
        WebSearch(local='duckduckgo'),
        WebFetch(local=True),
        Coder(workspace='/workspace', allowed_commands=[]),
        Memory(
            FileStore('/workspace/MEMORY'),
            namespace = lambda ctx: str(ctx.deps.guild_id) if ctx.deps.guild_id else (str(ctx.deps.message.guild.id) if (ctx.deps.message and ctx.deps.message.guild) else 'global'),
            heading = 'Agent Memory',
        ),
        hooks,
    ],
    retries = 100,
)

@agent.instructions
def system_prompt(ctx: RunContext[Deps]):   
    if ctx.deps.client and ctx.deps.config:
        return prompts.SYSTEM_PROMPT.render(client=ctx.deps.client, config=ctx.deps.config)
    
    return ''

def add_message_details(msg: discord.Message, indent=1):
    if not msg: return
    data = f"""
Message Author: {msg.author.display_name} (ID: {msg.author.id}). (Use the `get_user_info` tool to get more information about the user.)
Message ID: {msg.id} - use this in code if you want to do something like download attachments from the message.
"""
    
    if len(msg.attachments) > 0:
        data += f"""
{len(msg.attachments)} attachments (if you want to view them, try `analyse_file` with param `message`, or if that doesn't work download them using code):
    {[a.filename for a in msg.attachments]}
"""

    if not msg.reference:
        # data += "\n\nThe message is not replying to anything."
        ...
    else:
        data += f"""
Message is replying to this message:
    {add_message_details(msg.reference.resolved, indent+1) if indent <= 2 else '...'}
"""
        
    lines = data.splitlines()
    data = ''.join([(' '* 4 * indent) + line for line in lines])

    return data

class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    discriminator: str
    global_name: Optional[str] = None
    bot: bool
    system: bool
    created_at: datetime
    
    mention: str
    display_name: str
    
    avatar_url: Optional[str] = Field(None, alias="avatar")
    banner_url: Optional[str] = Field(None, alias="banner")
    accent_color: Optional[int] = None

    @field_validator("avatar_url", "banner_url", mode="before")
    @classmethod
    def transform_asset(cls, v):
        if isinstance(v, discord.Asset):
            return v.url
        return v

    @field_validator("accent_color", mode="before")
    @classmethod
    def transform_color(cls, v):
        if isinstance(v, discord.Color):
            return v.value
        return v
    
class Member(User):
    nick: Optional[str] = None
    joined_at: Optional[datetime] = None
    premium_since: Optional[datetime] = None
    
    roles: List[str] = Field(default_factory=list)

    @field_validator("roles", mode="before")
    @classmethod
    def transform_roles(cls, v):
        if isinstance(v, list):
            return[role.name for role in v if getattr(role, 'name', '') != '@everyone']
        return v

class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
    author: User
    created_at: datetime
    edited_at: Optional[datetime] = None

    mention_everyone: bool
    mentions: List[User] = Field(default_factory=list)
    role_mentions: List[str] = Field(default_factory=list)

    attachments: List[str] = Field(default_factory=list)
    embeds: List[dict] = Field(default_factory=list)

    pinned: bool
    tts: bool
    type: int

    @field_validator("mentions", mode="before")
    @classmethod
    def transform_mentions(cls, v):
        if isinstance(v, list):
            return [User.model_validate(user) for user in v]
        return v

    @field_validator("role_mentions", mode="before")
    @classmethod
    def transform_role_mentions(cls, v):
        if isinstance(v, list):
            return [role.name for role in v if getattr(role, "name", "") != "@everyone"]
        return v

    @field_validator("attachments", mode="before")
    @classmethod
    def transform_attachments(cls, v):
        if isinstance(v, list):
            return [attachment.url for attachment in v if isinstance(attachment, discord.Attachment)]
        return v

    @field_validator("embeds", mode="before")
    @classmethod
    def transform_embeds(cls, v):
        if isinstance(v, list):
            return [embed.to_dict() for embed in v if isinstance(embed, discord.Embed)]
        return v

class HasType(str, Enum):
    LINK = 'link'
    EMBED = 'embed'
    POLL = 'poll'
    FILE = 'file'
    VIDEO = 'video'
    IMAGE = 'image'
    SOUND = 'sound'
    STICKER = 'sticker'
    FORWARD = 'forward'

class SortOrder(str, Enum):
    ASCENDING = 'asc'
    DESCENDING = 'desc'

class SearchParams(BaseModel):
    author_id: Optional[str] = None
    mentions: Optional[str] = None
    has: Optional[HasType] = None
    channel_id: Optional[str] = None
    pinned: Optional[bool] = None
    sort_by: str = 'timestamp'
    sort_order: Optional[SortOrder] = SortOrder.DESCENDING
    offset: int = 0

class SearchResponse(BaseModel):
    messages: list[Message]
    total_results: int

@agent.tool()
async def search_discord(
    ctx: RunContext[Deps],
    search_params: SearchParams
): # -> SearchResponse:
    """
    Search through the entire Discord guild to find certain messages.

    Use this when, for example, a user asks to find the first message sent by a user, in a specific channel, or in the entire server, or containing a specific phrase.

    Param Names
    - Author ID:
      - The author of the user who sent the method. Leave empty to not check any authors.
    - Mentions:
      - The ID of the user who the message should mention. Leave empty to not check the message mentions.
    - Has:
      - Filter only messages which have a certain thing.
    - Channel ID:
      - The ID of the channel to search for. Self-explanatory. Like the others, leave this empty to not filter out any channels.
    - Pinned:
      - Set this to True to only include pinned messages in the results.
    - Sort By:
      - There is only one available option here, that is `timestamp`. I don't know why I even made this an option.
    - Sort Order:
      - Descending or Ascending. Self-explanatory.
    - Offset:
      - If you want to view page 2, page 3, of results until you find what you are looking for, you can use this. Because each search request returns the total result count as well as the first 20 after your offset.
    """
    try:
        # return SearchResponse.model_validate(await ctx.deps.client.http.request(
        return await ctx.deps.client.http.request(
            discord.http.Route(
                method = 'GET',
                path = f'/guilds/{ctx.deps.message.guild.id}/messages/search'
            ),
            params = search_params.model_dump(mode='json', exclude_none=True),
        )
    except Exception as e:
        return {"error": str(e)}

@agent.tool()
def get_user_info(ctx: RunContext[Deps]) -> Union[Member, User]:
    """Get the information of the user who sent the message."""
    author = ctx.deps.message.author
    if isinstance(author, discord.Member):
        return Member.model_validate(author)
    
    return User.model_validate(author)

@agent.tool()
async def run_shell(ctx: RunContext[Deps], command: str, timeout: int = 10) -> str:
    """
    This tool allows you to run shell commands on the system.

    Use this to install Python packages, navigate the filesystem, or download files.

    Prefer using the `run_command` tool over this, but you can use this as a fallback if the `run_command` tool decides to restrict your command for whatever reason.
    """
    logger.debug("Agent attempted shell command: %s", command)

    try:
        result = sp.run(command, shell=True, text=True, capture_output=True, timeout=timeout)
        print(result.stdout + result.stderr)
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except sp.TimeoutExpired:
        logger.debug("Agent shell command timed out after %ss", timeout)
        return f'Command timed out after {timeout}s.'
    except Exception as e:
        import traceback
        logger.debug("Agent code errored: %s", traceback.format_exc(), exc_info=e)
        return traceback.format_exc()

@agent.tool()
async def run_code(ctx: RunContext[Deps], code: str, timeout: int = 10):
    """
    This tool allows you to run Python code on the system.

    You have the following variables available to you:

    `message` - contains a `discord.Message` object of the current message, if necessary.
    `discord` - the `discord` library.
    `client`  - the `discord.Client` which you are running on.
    - All other builtins.

    You are allowed to use `async`/`await` keywords.

    Timeout is how long to wait for the function to run, in seconds.

    When writing code, always begin with `async def main(message, discord, client):` so that you have access to the `discord.Message` and `discord` and `discord.Client` objects.

    Inside your function, you can `return` with anything you want to send back to yourself, the AI agent.
    
    Whatever you return MUST be JSON-serializable (or a Pydantic object). If it is not, attempt to serialize it yourself first by e.g. writing a wrapper dictionary.

    If there is an error, provide error details to the user.

    If you need a 3rd party package, you can use `run_shell` to install it before running the code. For this, set the timeout to something higher e.g. 120.
    """
    
    warnings = []

    if not code.strip().startswith('async def main(message, discord, client):') or not 'async def main(message, discord, client):' in code:
        warnings.append("Your code didn't start with `async def main(message, discord, client):`. So the system added it for you and indented your code appropriately. If you don't receive any output / receive None, it's because you didn't have a `return` statement. You should try again and format the code properly within the function and return properly.")

        code = f"""
async def main(message, discord, client):
{indent(code, 4)}
        """

    locals  = {}
    globals = { '__builtins__': __builtins__ }

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    logger.debug("Agent attempted to run code:")
    logger.debug(code)
    logger.debug("Running...")

    stdout = ''
    stderr = ''
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, globals, locals)
            func = locals['main']

            result = await asyncio.wait_for(
                func(ctx.deps.message, discord, ctx.deps.client),
                timeout = timeout,
            )

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        logger.debug("Result: %s", result)
        print(stdout + stderr)

        return {'warnings': warnings, 'result': result, 'stdout': stdout, 'stderr': stderr}
    except asyncio.TimeoutError:
        logger.debug("Execution timed out.")
        return {'warnings': warnings, 'result': "Execution timed out.", 'stdout': stdout, 'stderr': stderr}
    except Exception as e:
        import traceback
        logger.debug(traceback.format_exc(), exc_info=e)
        return {'warnings': warnings, 'result': traceback.format_exc(), 'stdout': stdout, 'stderr': stderr}
    
class FileType(str, Enum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    DOCUMENT = 'document'
    TXT = 'txt'

    @staticmethod
    def from_mimetype(mimetype: str) -> 'FileType' | None:
        if not mimetype: return None

        if not '/' in mimetype: return None

        mtype, *_ = mimetype.split('/')

        if mtype in ('image', 'video', 'audio', 'document'): return FileType(mtype)
        if mtype == 'text': return FileType.TXT

        return None

@agent.tool()
async def analyse_file(ctx: RunContext[Deps], url: str, file_type: FileType, query: Optional[str] = None) -> str:
    """
    This tool analyses a file.
    Supported file types are dependent on the model, so some models may not support every single input type.
    However, here are all the possible accepted types:

    - image
    - audio
    - video
    - document (pdf, docs, etc.)
    - txt (plaintext .txt or .md files that you can read raw; this would return a summary of the content instead, or you can read it yourself)

    The url is the path to the file. It can either be a HTTP(S) URL to the file (useful for e.g. Discord CDN links), or
    an absolute / relative file path. You can also simply pass `message`, and it will return summarizations for all the attachments on a message.

    The query is the query to give the summarization model.

    If there is no query given, you will receive a summary of the file.
    If you have a specific query, you will receive a brief summary as well as an answer to the query, e.g. "What colour is the man's shirt?".
    """

    if url == 'message':
        msg = ctx.deps.message
        if not msg or not msg.attachments:
            return "No attachments found on the current message."

        results = []
        for attachment in msg.attachments:

            result = await analyse_file(ctx, attachment.url, FileType.from_mimetype(attachment.content_type) or file_type, query)
            results.append(f"[{attachment.filename}]: {result}")

        return '\n\n'.join(results)

    if url.startswith('http'):
        match file_type:
            case FileType.IMAGE:
                part = pydantic_ai.ImageUrl(url=url, force_download=True)
            case FileType.AUDIO:
                part = pydantic_ai.AudioUrl(url=url, force_download=True)
            case FileType.VIDEO:
                part = pydantic_ai.VideoUrl(url=url, force_download=True)
            case FileType.DOCUMENT:
                part = pydantic_ai.DocumentUrl(url=url, force_download=True)
            case FileType.TXT:
                try:
                    res = httpx.get(url=url, follow_redirects=True, headers={"Accept": "text/markdown, text/plain"})
                    res.raise_for_status()
                    part = pydantic_ai.TextContent(content=res.text)
                except Exception as e:
                    return f"Failed to fetch from URL: {e}"

    else:
        url = url.removeprefix('file://')
        path = Path(url)
        part = pydantic_ai.TextContent(path.read_text()) if path.suffix in ('.txt', '.md') else pydantic_ai.BinaryContent(path.read_bytes())

    try:
        response = await agent.run(
            user_prompt = [
                part,
                prompts.CONTENT_SUMMARIZATION_PROMPT.render(query=query),
            ],
            model = ctx.deps.model,
            deps = Deps(is_message=False, model=ctx.deps.model),
        )
        logger.debug("Summarized content: %s", response.output)
        return response.output
    except ModelAPIError as e:
        logger.warning("File analysis failed with model error: %s", e.message, exc_info=e)
        return f"There was an API error during the file parsing. See details: {e.message}\n\nThis is likely because your model does not support the specified file type."
    except Exception as e:
        logger.warning("File analysis failed with unknown error: %s", e, exc_info=e)
        return f"There was an unknown error during the operation. {e}"
    
@agent.tool()
async def trigger_reboot(ctx: RunContext[Deps]):
    """
    Triggers a reboot of the container you are running in.

    Only use this as a last resort, when you really have to.

    When this tool is ran, a message will be sent in the current channel saying that you are restarting.

    You can use this tool for things like e.g. when you modify your configuration and want to restart.
    """

    if ctx.deps.message:
        await ctx.deps.message.channel.send(embed = discord.Embed(
            title = "Rebooting...",
            description = "Agent triggered a reboot of the container.",
            timestamp = datetime.now(),
        ))
        Path('/workspace/.pending').write_text(json.dumps({
            'channel_id': ctx.deps.message.channel.id,
            'author_id': ctx.deps.author_id or ctx.deps.message.author.id,
            'guild_id': ctx.deps.guild_id or (ctx.deps.message.guild.id if ctx.deps.message.guild else 0),
        }))

    logger.info("Agent triggering a reboot of the container.")
    
    ctx.deps.reboot_requested = True

    return "The container will reboot now."

@agent.tool()
async def set_reminder(ctx: RunContext[Deps], prompt: str,
                       at: Optional[datetime] = None,
                       every_x_seconds: Optional[int] = None,
                       cron: Optional[str] = None) -> str:

    """
    This tool allows you to schedule yourself to be reminded (re-prompted) about something in this channel later.

    Provide one of the following:
        - `at` - a datetime for a reminder that executes once at a specific time.
        - `every_x_seconds` - repeat this reminder every X seconds. Try not to set anything under 300 for this value, or you will burn unnecessary tokens.
        - `cron` - a 5-field crontab string, e.g. `0 9 * * 1` (09:00 each Monday)

    `prompt` is what you wanna remind yourself of on each fire.
    """

    if not ctx.deps.scheduler or not ctx.deps.message:
        logger.warning("Reminders aren't available, but agent tried to call them.")
        return "Reminders are not available."
    
    try:
        rid = ctx.deps.scheduler.add(
            ctx.deps.message.channel.id,
            ctx.deps.author_id or ctx.deps.message.author.id,
            ctx.deps.guild_id or (ctx.deps.message.guild.id if ctx.deps.message.guild else 0),
            prompt, at=at, every_x_seconds=every_x_seconds, cron=cron,
        )
    except ValueError as e:
        logger.warning("Failed to set reminder: %s", e)
        return f"Could not set reminder: {e}"

    return f"Reminder scheduled (ID: {rid})."

@agent.tool()
async def cancel_reminder(ctx: RunContext[Deps], reminder_id: str) -> str:
    """Cancel a scheduled reminder by its ID."""

    if not ctx.deps.scheduler:
        logger.warning("Reminders aren't available, but agent tried to call them.")
        return "Reminders are not available."
    
    return "Cancelled." if ctx.deps.scheduler.cancel(reminder_id) else "Reminder not found."

@agent.tool()
async def list_reminders(ctx: RunContext[Deps]) -> list:
    """List all pending reminders."""

    if not ctx.deps.scheduler:
        logger.warning("Reminders aren't available, but agent tried to call them.")
        return []
    
    return ctx.deps.scheduler.list_all()

@agent.tool()
async def ban_user(ctx: RunContext[Deps], user_id: int, reason: str = '') -> str:
    """
    Ban a user from interacting with YOU (this bot), permanently, until you unban them.

    This controls access to this bot only, and doesn't ban the user from the Discord server.

    If you want to ban the user from the server, then use the `run_code` tool to access the discord API instead.
    """
    moderation.ban(ctx.deps.client.engine, user_id, reason or None)
    return f"User {user_id} is now banned from using the bot."

@agent.tool()
async def unban_user(ctx: RunContext[Deps], user_id: int) -> str:
    """Remove a bot-access ban (see `ban_user`). Bot access only, not a Discord server unban."""
    return "Unbanned." if moderation.unban(ctx.deps.client.engine, user_id) else "That user was not banned."

@agent.tool()
async def timeout_user(ctx: RunContext[Deps], user_id: int, seconds: int, reason: str = "") -> str:
    """
    Temporarily block a user from interacting with YOU (this bot) for `seconds` seconds.

    This controls access to this bot only, and doesn't timeout the user in the Discord server.

    If you want to timeout the user in the server, then use the `run_code` tool to access the discord API instead.
    """
    until = moderation.timeout(ctx.deps.client.engine, user_id, seconds, reason or None)
    if until is None:
        return f"User {user_id} is permanently banned; unban them first if you want a timeout instead."
    return f"User {user_id} is timed out from the bot until {until:%Y-%m-%d %H:%M UTC}."

@agent.tool()
async def cancel_timeout(ctx: RunContext[Deps], user_id: int) -> str:
    """Cancel a bot-access timeout early (see `timeout_user`). Doesn't cancel Discord server-wide timeouts, only timeouts on this bot."""
    return "Timeout cancelled." if moderation.cancel_timeout(ctx.deps.client.engine, user_id) else "That user is not timed out."