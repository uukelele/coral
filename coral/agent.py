from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import *
from datetime import datetime
import discord
import discord.http
import asyncio
from dataclasses import dataclass
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import subprocess as sp
from enum import Enum

from .utils import indent
from . import config, prompts

@dataclass
class Deps:
    message: discord.Message
    client: discord.Client
    config: config.Config

agent = Agent(
    deps_type = Deps,
    tools=[duckduckgo_search_tool()]
)

@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]):
    return prompts.SYSTEM_PROMPT.render(client=ctx.deps.client, config=ctx.deps.config)

@agent.instructions
def add_message_details(ctx: RunContext[Deps] | discord.Message, indent=1):
    if not ctx: return 'Message not found.'
    msg = ctx.deps.message if isinstance(ctx, RunContext) else ctx
    data = f"""
Message Author: {msg.author.display_name} (ID: {msg.author.id}). (Use the `get_user_info` tool to get more information about the user.)
Message ID: {msg.id} - use this in code if you want to do something like download attachments from the message.
"""

    if not msg.reference:
        data += "\n\nThe message is not replying to anything."
    else:
        data += f"""
Message Reference:
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
    author_id: Optional[int] = None
    mentions: Optional[int] = None
    has: Optional[HasType] = None
    channel_id: Optional[int] = None
    pinned: Optional[bool] = None
    sort_by: str = 'timestamp'
    sort_order: Optional[SortOrder] = SortOrder.DESCENDING
    offset: int = 0

class SearchResponse(BaseModel):
    messages: list[Message]
    total_results: int

@agent.tool
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

@agent.tool
def get_user_info(ctx: RunContext[Deps]) -> Union[Member, User]:
    """Get the information of the user who sent the message."""
    author = ctx.deps.message.author
    if isinstance(author, discord.Member):
        return Member.model_validate(author)
    
    return User.model_validate(author)

@agent.tool
async def run_shell(ctx: RunContext[Deps], command: str, timeout: int = 10) -> str:
    """
    This tool allows you to run shell commands on the system.

    Use this to install Python packages, navigate the filesystem, or download files.
    """
    print(f"Agent running shell command: {command}")

    try:
        result = sp.run(command, shell=True, text=True, capture_output=True, timeout=timeout)
        print(result.stdout + result.stderr)
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except sp.TimeoutExpired:
        return f'Command timed out after {timeout}s.'
    except Exception as e:
        import traceback
        traceback.print_exc()
        return traceback.format_exc()


@agent.tool
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

    print("Agent attempted to run code:")
    print(code)
    print("Running...")

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

        print(f"Result: {result}")
        print(stdout + stderr)

        return {'warnings': warnings, 'result': result, 'stdout': stdout, 'stderr': stderr}
    except asyncio.TimeoutError:
        print("Execution timed out.")
        return {'warnings': warnings, 'result': "Execution timed out.", 'stdout': stdout, 'stderr': stderr}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'warnings': warnings, 'result': traceback.format_exc(), 'stdout': stdout, 'stderr': stderr}