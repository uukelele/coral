import pydantic_ai
import pydantic_core
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools, Thinking, WebSearch, WebFetch, Hooks, LocalWorkspace, NativeTool
from pydantic_ai.native_tools import WebSearchTool, WebFetchTool
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Coder, Memory, ToolGuardrail, GuardrailResult, TieredCompaction, ClearToolResults, SummarizingCompaction, ClampOversizedMessages, ReportContextUsage, DeduplicateFileReads, PromptInjectionDefender, CapabilityCreation, PydanticAIDocs
from pydantic_ai_harness.guardrails import ToolCallInfo
from pydantic_ai_harness.memory import FileStore
from pydantic_ai_harness.memory._toolset import list_subfiles
from pydantic_ai_harness.compaction import ContextUsageEvent
from pydantic_ai_harness.capability_creation._capability import _DEFAULT_GUIDANCE
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import TextPart, ToolCallPart, RetryPromptPart
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
import mimetypes
import httpx
import logging
import json
import re
import time

from . import config as libcfg, prompts, reminders, moderation, utils

logger = logging.getLogger(__name__)

ATTACHMENT_MAX_BYTES = 8 * 1024 * 1024 # discord limit 8MB file size for bots
ATTACHMENT_MAX_COUNT = 10
ATTACHMENT_SOURCES = ('/workspace', '/tmp')

MAX_RETRIES = 3

@dataclasses.dataclass
class Deps:
    model: Model | str
    message: discord.Message = None
    channel: discord.abc.Messageable = None
    client: discord.Client = None
    config: libcfg.Config = None
    is_message: bool = True
    tier: Optional[libcfg.Tier] = None
    reboot_requested: bool = False
    scheduler: Optional[reminders.Scheduler] = None
    author_id: Optional[int] = None
    guild_id: Optional[int] = None
    attached_files: list[str] = dataclasses.field(default_factory=list)
    status_message: Optional[discord.Message] = None
    context_usage: Optional[ContextUsageEvent] = None

async def denial_reason(ctx: RunContext[Deps], tool_name: str) -> str | None:
    tier: libcfg.Tier = getattr(ctx.deps, 'tier', None)
    if tier is None:
        return None
    return None if tier.can_use_tool(tool_name) else f"The user does not have permission to use tool {tool_name}. This is usually because the user's permission tier is not high enough to grant access to this tool."

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

def log_injection(ctx: RunContext[Deps], call: ToolCallPart, verdict) -> None:
    logger.warning("Prompt-injection defender flagged %s (risk=%s): %s", call.tool_name, getattr(verdict, 'risk_level', '?'), getattr(verdict, 'detections', None))

hooks = Hooks()

@hooks.on.model_request
async def retry_5xx(ctx: RunContext[Deps], *, request_context, handler):
    attempt = 0
    while True:
        try:
            resp = await handler(request_context)
        except ModelHTTPError as e:
            if not (500 <= e.status_code <= 599):
                raise

            attempt += 1
            if attempt > MAX_RETRIES:
                raise

            if ctx.deps.message:
                text = f"⚠️ Provider returned error {e.status_code}. Retrying ({attempt}/{MAX_RETRIES})..."
                try:
                    if ctx.deps.status_message is None:
                        ctx.deps.status_message = await ctx.deps.message.channel.send(text)
                    else:
                        await ctx.deps.status_message.edit(content=text)
                except Exception: ...

            await asyncio.sleep(min((getattr(e, 'retry_after', None) or 0) or 2 ** (attempt - 1), 30))
            continue

        if ctx.deps.status_message is not None:
            try: await ctx.deps.status_message.delete()
            except Exception: ...

        return resp

@hooks.on.after_model_request
async def send_text_updates(ctx: RunContext[Deps], *, request_context, response):
    if not ctx.deps.message:
        return response

    parts = response.parts
    if any(isinstance(p, ToolCallPart) for p in parts):
        for part in parts:
            if isinstance(part, TextPart) and part.content.strip():
                if stripped := utils.strip_thinking(part.content):
                    await ctx.deps.message.channel.send(stripped)

    return response

@hooks.on.tool_execute_error
async def tool_error_to_result(ctx: RunContext[Deps], *, call: ToolCallPart, tool_def: ToolDefinition, args, error: Exception):
    logger.warning("Tool %s raised %s: %s", call.tool_name, type(error).__name__, error, exc_info=error)
    return f"Tool `{call.tool_name}` failed with {type(error).__name__}: {error}\n\nTry a different approach, or use different arguments."

def _native_tool(tool_cls: type[WebSearchTool] | type[WebFetchTool], permission: str) -> NativeTool[Deps]:
    def build(ctx: RunContext[Deps]):
        if isinstance(ctx.model, TestModel): return None # TestModel raises errors when native tools are given to it
        if tool_cls is WebSearchTool and isinstance(ctx.model, OpenAIChatModel) and not ctx.model.profile.get('openai_chat_supports_web_search', False):
            return None # /chat/completions raises error instead of silently ignoring , even when marked as optional
        tier = ctx.deps.tier
        if tier is not None and not tier.can_use_tool(permission): return None
        return tool_cls(optional=True) # optional, so models that can't use it, don't.
    return NativeTool(build)

def _fk(call: ToolCallPart) -> str | None:
    if call.tool_name != 'read_file':
        return None
    return call.args_as_dict().get('path')

CAPABILITY_CREATION_GUIDANCE = _DEFAULT_GUIDANCE + f"""

Before authoring or changing a capability, read the current docs instead of relying on memory.

For core pydantic-ai APIs, call `read_pyai_docs(topic)` (topics: `capabilities`, `hooks`, `tools`, `tools-advanced`, `toolsets`, `agent`).
For harness capabilities and anything else, use the `web_fetch` tool: fetch the docs index at "https://pydantic.dev/docs/ai/llms.txt" to find the right page, then fetch its markdown content by appending `/index.md` to the page path (e.g. "https://pydantic.dev/docs/ai/harness/capability-creation/index.md").
Always fetch the `.md` / `index.md` form, not the browser-rendered HTML page instead.
"""

creation = CapabilityCreation(directory=Path('/workspace/capabilities'), guidance=CAPABILITY_CREATION_GUIDANCE)
# defined outside the agent constructor so that it can be imported from `bot.py`

def _memory_namespace(ctx: RunContext[Deps]) -> str: return str(ctx.deps.guild_id) if ctx.deps.guild_id else (str(ctx.deps.message.guild.id) if (ctx.deps.message and ctx.deps.message.guild) else 'global')

Path('/workspace/MEMORY').mkdir(parents=True, exist_ok=True) # LocalWorkspaceBackend never creates its own directory
memory_store = FileStore('.', workspace=LocalWorkspaceBackend('/workspace/MEMORY'))

agent = Agent(
    deps_type = Deps,
    capabilities = [
        TieredCompaction(
            tiers = [
                ClampOversizedMessages(40_000),
                DeduplicateFileReads(_fk),
                SummarizingCompaction(max_messages=1, keep_messages=25)
            ],
            target_fraction = 0.9,
        ),
        ReportContextUsage(),
        PrepareTools(annotate_unavailable_tools),
        ToolGuardrail(guard=block_unauthorized),
        PromptInjectionDefender(
            block_high_risk = True,
            on_detection = log_injection,
        ),
        Thinking(),
        WebSearch(native=False, local='duckduckgo'),
        _native_tool(WebSearchTool, 'native_web_search'),
        WebFetch(native=False, local=True),
        _native_tool(WebFetchTool, 'native_web_fetch'),
        PydanticAIDocs(),
        creation,
        LocalWorkspace('/workspace'), # CapabilityCreation requires a writable local workspace
        Coder(unrestricted_filesystem=True),
        Memory(
            memory_store,
            namespace = _memory_namespace,
            heading = 'Agent Memory',
            inject_memory = False,
        ),
        hooks,
    ],
    retries = 100,
)

@agent.on_event(ContextUsageEvent)
async def report_context(ctx: RunContext[Deps], event: ContextUsageEvent):
    logger.debug("Context: %s / %s (resolved=%s, %.1f%%)", event.used_tokens, event.window_tokens, event.resolved, event.fraction * 100)
    ctx.deps.context_usage = event

@agent.instructions
def system_prompt(ctx: RunContext[Deps]):   
    if ctx.deps.client and ctx.deps.config:
        return prompts.SYSTEM_PROMPT.render(client=ctx.deps.client, config=ctx.deps.config)
    
    return ''

@agent.instructions
async def memory_context(ctx: RunContext[Deps]):
    scope = f"{_memory_namespace(ctx)}/main"
    try:
        f = await memory_store.read(f"{scope}/MEMORY.md", max_chars=8000)
        subfiles, truncated = await list_subfiles(memory_store, scope, limit=50)
    except Exception:
        return ''

    body = (f.content if f and f.content.strip() else '(empty)')
    out = ["## Your memory (persistent notes from your previous sessions)", '', body]
    if subfiles:
        out += ['', "Other memory files (use `read_memory` or `search_memory`):", *[f"- {p}" for p in subfiles]]

    return '\n'.join(out)

_GUILD_CTX_CACHE: dict[int, tuple[float, str]] = {} # guild_id -> (time, context)
_GUILD_CTX_TTL = 60.0 # cache bust after 60 secs
_MAX_EMOJIS = 200
_MAX_ROLES = 200
_MAX_CHANNELS = 200

def _guild_static_context(guild: discord.Guild) -> str:
    hit = _GUILD_CTX_CACHE.get(guild.id)
    if hit and time.monotonic() - hit[0] < _GUILD_CTX_TTL:
        return hit[1]

    lines: list[str] = []

    emojis = list(guild.emojis)[:_MAX_EMOJIS]
    if emojis:
        lines.append("### Custom emojis (use these a lot if you want)")
        for e in emojis:
            code = f"<a:{e.name}:{e.id}>" if e.animated else f"<:{e.name}:{e.id}>"
            lines.append(f"- :{e.name}: -> `{code}`" + ('' if e.available else " (may be unavailable!)"))
        if len(guild.emojis) > _MAX_EMOJIS:
            lines.append(f"- ...and {len(guild.emojis) - _MAX_EMOJIS} more (omitted)")
        lines.append('')

    roles = [r for r in sorted(guild.roles, key=lambda r: r.position, reverse=True) if not r.is_default()][:_MAX_ROLES]
    if roles:
        lines.append("### Roles (in hierarchy order)")
        for r in roles:
            lines.append(f"- {r.name} ({len(r.members)} members) -> `<@&{r.id}>`")
        lines.append('')

    chnls = []
    count = 0
    for category, channels in guild.by_category():
        if category is not None:
            chnls.append(f"- **{category.name}**")
            indent = "  - "
        else:
            indent = "- "
        for c in channels:
            if count >= _MAX_CHANNELS: break
            kind = type(c).__name__.replace('Channel', '').lower() # 'text' | 'voice' | 'forum' | 'stage' | ...
            chnls.append(f"{indent}#{c.name} (<#{c.id}>) [{kind}]" + ('' if c.permissions_for(guild.me).view_channel else ' [hidden]'))
            count += 1
        if count >= _MAX_CHANNELS: break

    if chnls:
        lines.append("### Channels")
        lines += chnls
        if count >= _MAX_CHANNELS:
            lines.append("- ...and some more (omitted)")
        lines.append('')
    

    text = '\n'.join(lines)
    _GUILD_CTX_CACHE[guild.id] = (time.monotonic(), text)
    return text

@agent.instructions
async def dynamic_ctx(ctx: RunContext[Deps]):
    msg = ctx.deps.message
    channel = ctx.deps.channel or (msg.channel if msg else None)
    guild = getattr(channel, 'guild', None) or (msg.guild if msg else None)

    lines = ["# Live Context", '', f"Current Time: {utils.now().strftime('%Y-%m-%d %H:%M:%S UTC')} (this is not the time the session started, this is the time RIGHT NOW)"]

    if channel is not None:
        cid = getattr(channel, 'id', None)
        cname = getattr(channel, 'name', None)
        lines += ['', "## Current channel"]
        if cname and cid: lines.append(f"You are in #{cname} (<#{cid}>).")
        topic = getattr(channel, 'topic', None)
        if topic: lines.append(f"Channel topic: {topic}")
        if isinstance(channel, discord.Thread):
            lines.append(f"This is a thread named '{channel.name}'.")
            if channel.owner: lines.append(f"Thread started by @{channel.owner.name} (<@{channel.owner.id}>).")
            elif channel.owner_id: lines.append(f"Thread started by <@{channel.owner_id}>.")

            tags = [t.name for t in channel.applied_tags]
            if tags: lines.append(f"Tags: {', '.join(tags)}")

            parent = channel.parent
            if isinstance(parent, discord.ForumChannel):
                lines.append(f"Parent forum: #{parent.name} (<#{parent.id}>)" + (f" -- {parent.topic}" if parent.topic else ''))

            try:
                starter = await channel.fetch_message(channel.id)
                if starter and starter.content:
                    body = starter.content
                    if len(body) > 1000:
                        body = body[:1000] + '...'
                    lines.append(f"Original post by {starter.author.name}:\n\n{body}")
            except Exception: ...

    if guild is not None:
        lines += ['', '## Server',
            f"{guild.name} (id `{guild.id}`, {guild.member_count} members). "
            f"Your [nick]name here is `{guild.me.display_name if getattr(guild, 'me', None) else ctx.deps.client.user.name}`."
        ]

        static = _guild_static_context(guild)
        if static:
            lines += ['', static]

    return '\n'.join(lines)


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

    Prefer using the `shell` tool over this, but you can use this as a fallback if the `shell` tool decides to restrict your command for whatever reason.
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
    
    return await utils.run_code(code, "async def main(message, discord, client):", (ctx.deps.message, discord, ctx.deps.client), timeout)
    
class FileType(str, Enum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    DOCUMENT = 'document'
    TXT = 'txt'

    @staticmethod
    def from_mimetype(mimetype: str) -> 'FileType | None':
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
    an absolute / relative file path. You can also simply pass `message`, and it will return summarizations for all the attachments on the latest message.
    You can pass `discord://<channel_id>/<message_id>` to get for a specific message ID, or `discord://<message_id>` and it will auto detect the channel ID as the current channel.
    `discord://message` also works.

    The query is the query to give the summarization model.

    If there is no query given, you will receive a summary of the file.
    If you have a specific query, you will receive a brief summary as well as an answer to the query, e.g. "What colour is the man's shirt?".
    """

    if url == 'message' or url == 'discord://message':
        msg = ctx.deps.message
        if not msg or not msg.attachments:
            return "No attachments found on the current message."

        results = []
        for attachment in msg.attachments:

            result = await analyse_file(ctx, attachment.url, FileType.from_mimetype(attachment.content_type) or file_type, query)
            results.append(f"[{attachment.filename}]: {result}")

        return '\n\n'.join(results)

    if url.startswith('discord://'):
        url = url.removeprefix('discord://').removesuffix('/')
       
        if '/' in url:
            cid, mid = url.split('/', 1)
        else:
            cid = ctx.deps.channel or (ctx.deps.message.channel if ctx.deps.message else None)
            mid = url

        try:
            if isinstance(cid, str):
                cid = ctx.deps.client.get_channel(int(cid)) if ctx.deps.client else (ctx.deps.channel or (ctx.deps.message.channel if ctx.deps.message else None))

            message = await cid.fetch_message(mid)

            if not message.attachments:
                return f"Message {mid} has no attachments."

            results = []
            for attachment in message.attachments:
                result = await analyse_file(ctx, attachment.url, FileType.from_mimetype(attachment.content_type) or file_type, query)
                results.append(f"[{attachment.filename}]: {result}")

            return '\n\n'.join(results)

        except Exception as e:
            return f"Failed to fetch message {mid} in channel {cid}: {e}"

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
        try:
            if path.suffix in ('.txt', '.md', '.html'):
                part = pydantic_ai.TextContent(path.read_text())
            else:
                mtype = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
                part = pydantic_ai.BinaryContent(data=path.read_bytes(), media_type=mtype)
        except OSError as e: return f"Failed to read {url}: {e}"

    try:
        summarizer = Agent(
            model=ctx.deps.model,
            name='FileAnalysis',
            capabilities=[PromptInjectionDefender(block_high_risk=True)]
        )

        response = await summarizer.run(
            user_prompt = [
                part,
                prompts.CONTENT_SUMMARIZATION_PROMPT.render(query=query),
            ],
            model = ctx.deps.model,
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

PROMPT_BLOCKS = {'on_message', 'on_message_edit', 'on_typing', 'on_raw_typing', 'on_presence_update', 'on_socket_event_type', 'on_socket_raw_receive'} # because they fire too frequently

@agent.tool()
async def set_automation(
    ctx: RunContext[Deps],
    name: str, action: Literal['prompt', 'code'], payload: str,
    at: Optional[datetime] = None, every_x_seconds: Optional[int] = None,
    cron: Optional[str] = None, event: Optional[str] = None,
    channel_id: Optional[int] = None) -> str:
    """
    Schedule yourself to be triggered later by TIME or by a Discord EVENT, running either a
    prompt (re-prompt yourself) or code (runs without an LLM call, it's more efficient, use for deterministic tasks).

    Exactly ONE trigger:
      - at / every_x_seconds / cron  (time)
      - event  (a discord.py event name, e.g. `on_member_join`, `on_member_remove`)

    action:
      - 'prompt': `payload` is text you'll be re-prompted with. (Prompt actions aren't allowed on events that occur frequently like `on_message` - too costly.)
      - 'code':   `payload` is Python:  `async def main(event, discord, client): ...`
        `event` is the discord event's args tuple (e.g. `(member,)` for `on_member_join`), or
        `None` for time triggers. Don't react to your client's own events to avoid loops.
        You can also pass a path to a Python file here, where the path contains the function.

    `name` identifies the automation for list_automations / cancel_automation. It's basically an ID.
    """

    if action not in ('prompt', 'code'): return "Action must be 'prompt' or 'code'."

    triggers = [t for t in (at, every_x_seconds, cron, event) if t]
    if len(triggers) != 1: return "You have to provide exactly **ONE** trigger: `at`, `every_x_seconds`, `cron`, `event`."

    author_id = ctx.deps.author_id or (ctx.deps.message.author.id if ctx.deps.message else None)
    guild_id  = ctx.deps.guild_id or (ctx.deps.message.guild.id if (ctx.deps.message and ctx.deps.message.guild) else None)
    channel   = channel_id or (ctx.deps.message.channel.id if ctx.deps.message else None)
    if author_id is None:
        return "Can't set an automation without a user to be in charge."

    if event is not None:
        ev = event if event.startswith('on_') else f'on_{event}'
        bare = ev[3:]
        if action == 'prompt' and ev in PROMPT_BLOCKS:
            return f"You can't call yourself '{ev}' because it's too costly and frequent. But you can use `code` action with '{ev}'."

        if not ctx.deps.client: return "Automations are not available for some reason."

        ok = ctx.deps.client.register_event_automation({"name": name, "event": bare, "action": action, "payload": payload, "channel_id": channel, "author_id": author_id, "guild_id": guild_id})

        return (f"Event automation '{name}' registered for {ev} (action={action})." if ok else f"An automation named '{name}' already exists. Cancel it first.")

    if not ctx.deps.scheduler:
        return "Automations are not available for some reason."

    try:
        rid = ctx.deps.scheduler.add(action, payload, channel, author_id, guild_id, at=at, every_x_seconds=every_x_seconds, cron=cron, name=name)
    except Exception as e:
        logger.warning("Failed to set automation: %s", e, exc_info=e)
        return f"Could not set automation: {e}"

    return f"Automation '{rid}' scheduled (action={action})."
    

@agent.tool()
async def cancel_automation(ctx: RunContext[Deps], name: str) -> str:
    """Cancel an automation by name."""
    cancelled = False
    if ctx.deps.scheduler and ctx.deps.scheduler.cancel(name):
        cancelled = True
    if ctx.deps.client and ctx.deps.client.unregister_event_automation(name):
        cancelled = True
    return "Cancelled." if cancelled else f"No automation named '{name}'."


@agent.tool()
async def list_automations(ctx: RunContext[Deps]) -> list:
    """List all automations."""
    out = []
    if ctx.deps.scheduler:
        out.extend(ctx.deps.scheduler.list_all())
    if ctx.deps.client:
        for bare, autos in ctx.deps.client.ev_automations.items():
            for a in autos:
                out.append({'id': a['name'], 'event': f"on_{bare}", 'action': a['action'], 'payload': a['payload'], 'channel_id': a['channel_id'], 'author_id': a['author_id']})
    return out


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

@agent.tool()
async def attach_file(ctx: RunContext[Deps], paths: list[str]) -> str:
    """
    Attach one (or more) files to your next reply in this channel.

    Pass a list of file paths accessible in your environment.
    After this tool returns, you can send your final message and the files will be uploaded to Discord.

    You can call this tool multiple times, but there must never be more than **10** files (discord hard limit) and each file must be a maximum of 8MB.

    You can only upload files in `/workspace` or `/tmp`.

    Suggestion: If you want to provide the user with a large piece of code, it is **always** recommended to instead of putting it in a codeblock, use `write_file` to write it to a temporary path and upload it here, but only do so if you think the script is big enough that it won't fit in a single Discord response.
    """

    added, errors = [], []

    for p in paths:
        rp = Path(p).resolve()

        if not any(str(rp) == r or str(rp).startswith(r + '/') for r in ATTACHMENT_SOURCES):
            errors.append(f"{p}: must be in one of the following directories: {ATTACHMENT_SOURCES}")
            continue
        if not rp.is_file():
            errors.append(f"{p}: not a file.")
            continue
        if (size := rp.stat().st_size) > ATTACHMENT_MAX_BYTES:
            errors.append(f"{p}: exceeds max size of {ATTACHMENT_MAX_BYTES} bytes (file is {size} bytes)")
            continue
        if len(ctx.deps.attached_files) + len(added) >= ATTACHMENT_MAX_COUNT:
            errors.append(f"{p}: attachment limit ({ATTACHMENT_MAX_COUNT}) reached")
            continue

        added.append(str(rp))
    ctx.deps.attached_files.extend(added)
    msg = f"Attached {len(added)} files: {[Path(a).name for a in added]}. You can now send a final message, or keep working (but these files will be attached on your final message)."
    if errors: msg += f"\nSkipped: {errors}"
    return msg

def _sup(ctx):
    return getattr(ctx.deps.client, 'supervisor', None) if ctx.deps.client else None

@agent.tool()
async def create_service(ctx: RunContext[Deps], name: str, command: list[str], cwd: Optional[str] = None, env: Optional[dict] = None, autorestart: bool = True) -> str:
    """
    Create and start a persistent background service that survives reboots (systemd-lite).

    Write your script file(s) first (e.g. with write_file), then register how to run them here.
    `command` = binary + args, e.g. ["python", "/workspace/services/giveaway_bot.py"].
    You can run your own Discord bot: os.getenv("DISCORD_TOKEN") is available in the environment.
    The manifest is stored as YAML at /workspace/services/{name}.yaml. Services auto-restart on
    crash unless they crash-loop (3 quick crashes) - then fix the bug and call `restart_service`.
    """
    sup = _sup(ctx)
    return await sup.create(name, command, cwd=cwd, env=env, autorestart=autorestart) if sup else "Services are not available."

@agent.tool()
async def list_services(ctx: RunContext[Deps]) -> list:
    """List all registered services and their status."""
    sup = _sup(ctx)
    return sup.list_all() if sup else []

@agent.tool()
async def service_status(ctx: RunContext[Deps], name: str) -> dict:
    """Detailed status + recent log tail for one service."""
    sup = _sup(ctx)
    return sup.status(name) if sup else {'error': 'Services are not available.'}

@agent.tool()
async def stop_service(ctx: RunContext[Deps], name: str) -> str:
    """Stop a service and keep it stopped across reboots."""
    sup = _sup(ctx)
    return await sup.stop_service(name) if sup else "Services are not available."

@agent.tool()
async def restart_service(ctx: RunContext[Deps], name: str) -> str:
    """Restart a service (also clears crash-loop state, e.g. after you fixed a bug)."""
    sup = _sup(ctx)
    return await sup.restart_service(name) if sup else "Services are not available."

@agent.tool()
async def remove_service(ctx: RunContext[Deps], name: str) -> str:
    """Stop and permanently remove a service (deletes its manifest)."""
    sup = _sup(ctx)
    return await sup.remove(name) if sup else "Services are not available."

@agent.tool()
async def conversation_catchup(ctx: RunContext[Deps], limit: int = 20) -> str:
    """
    Read the last `limit` (default 20, max 100) messages in the CURRENT channel as markdown, so you can catch up on recent conversation.

    This is better than `run_code` for reading chat history, because it is less verbose and more structured.

    Messages with attachments include their message ID, so you can use that in `analyse_file` if you wish.
    """

    channel = ctx.deps.channel or (ctx.deps.message.channel if ctx.deps.message else None)
    if channel is None: return "No available channel to read."

    limit = max(1, min(limit, 100))
    try:
        msgs = [m async for m in channel.history(limit=limit)]
    except Exception as e: return f"Failed to read channel history: {e}"

    msgs.reverse()

    out = []
    for m in msgs:
        ts = m.created_at.strftime('%Y-%m-%d %H:%M UTC')
        head = f"**{m.author.display_name}** (`{m.author.id}`) · {ts} · message ID {m.id}"
        if m.reference and m.reference.message_id:
            head += f" · ↩ reply to `{m.reference.message_id}`"
        content = utils.clean(m) if m.content else ''
        out.append(f"{head}\n{content}" if content else head)
        if m.attachments:
            out.append("  attachments: " + ", ".join(a.filename for a in m.attachments))

    return '\n\n'.join(out) if out else "No recent messages."
