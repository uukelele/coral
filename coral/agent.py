from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import *
from datetime import datetime
import discord
import asyncio
from dataclasses import dataclass
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

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
    return prompts.SYSTEM_PROMPT.render(client=ctx.deps.client, config=ctx.deps.config),

@agent.instructions
def add_message_details(ctx: RunContext[Deps], indent=1):
    msg = ctx.deps.message
    data = f"""
Message Author: {msg.author.display_name} (ID: {msg.author.id}). (Use the `get_user_info` tool to get more information about the user.)
Message ID: {msg.id} - use this in code if you want to do something like download attachments from the message.
"""

    if not msg.reference:
        data += "\n\nThe message is not replying to anything."
    else:
        data += f"""
Message Reference:
    {add_message_details(ctx, indent+1)}
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


@agent.tool
def get_user_info(ctx: RunContext[Deps]) -> Union[Member, User]:
    """Get the information of the user who sent the message."""
    author = ctx.deps.message.author
    if isinstance(author, discord.Member):
        return Member.model_validate(author)
    
    return User.model_validate(author)

@agent.tool
async def run_code(ctx: RunContext[Deps], code: str, timeout: int = 10):
    """
    This tool allows you to run Python code on the system.

    You have the following variables available to you:

    `message` - contains a `discord.Message` object of the current message, if necessary.
    `discord` - the `discord` library.
    - All other builtins.

    You are allowed to use `async`/`await` keywords.

    Timeout is how long to wait for the function to run, in seconds.

    When writing code, always begin with `async def main(message, discord, client):` so that you have access to the `discord.Message` and `discord` and `discord.Client` objects.

    Inside your function, you can `return` with anything you want to send back to yourself, the AI agent.
    
    Whatever you return MUST be JSON-serializable (or a Pydantic object). If it is not, attempt to serialize it yourself first by e.g. writing a wrapper dictionary.

    If there is an error, provide error details to the user.
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