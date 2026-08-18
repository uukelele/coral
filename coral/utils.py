import discord
import re, asyncio
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import logging
import tempfile, os

logger = logging.getLogger(__name__)

def now() -> datetime: return datetime.now(timezone.utc)

_CODEBLOCK = re.compile(r'```([\w+.\-]*)\n(.*?)```', re.DOTALL)
_EXT_BY_LANG = {
    'python': 'py', 'py': 'py', 'js': 'js', 'javascript': 'js', 'ts': 'ts', 'typescript': 'ts',
    'json': 'json', 'yaml': 'yaml', 'yml': 'yml', 'sh': 'sh', 'bash': 'sh', 'html': 'html',
    'css': 'css', 'sql': 'sql', 'c': 'c', 'cpp': 'cpp', 'java': 'java', 'go': 'go',
    'rust': 'rs', 'rs': 'rs', 'md': 'md', 'xml': 'xml', 'markdown': 'md', '': 'txt',
}

def extract_large_codeblocks(text: str, size: int = 2000):
    files = []
    def repl(m):
        block = m.group(0)
        if len(block) <= size:
            return block
        lang = (m.group(1) or '').lower()
        fd, path = tempfile.mkstemp(suffix=f".{_EXT_BY_LANG.get(lang, 'txt')}", prefix='coral_code_', dir='/tmp')
        with os.fdopen(fd, 'w') as f:
            f.write(m.group(2))
        files.append(path)
        return f"[{lang} code attached]"
    return _CODEBLOCK.sub(repl, text), files


def _split_long_line(text: str, size: int) -> list[str]:
    out = []

    while len(text) > size:
        cut = text.rfind(' ', 1, size)
        if cut == -1:
            out.append(text[:size])
            text = text[size:]
        else:
            out.append(text[:cut])
            text = text[cut + 1:]

    if text: out.append(text)

    return out

def chunk_string(s: str, size: int = 2000) -> list[str]:
    if len(s) <= size: return [s]

    chunks: list[str] = []
    current = ""

    for line in s.splitlines(keepends=True):
        if len(line) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_line(line, size))
            continue

        if len(current) + len(line) > size:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks

_MASS_MENTION_RE = re.compile(r'@(everyone|here)')

def neutralize_mass_mentions(text: str) -> str:
    return _MASS_MENTION_RE.sub(r'\1', text)

_ROLE_MENTION_RE = re.compile(r'<@&([0-9]{15,20})>')

def sanitize_role_mentions(text: str, guild, channel, member, allow_everyone: bool = False):
    if guild is None or member is None:
        return text, []

    try:
        can_mention_any = channel.permissions_for(member).mention_everyone
    except Exception:
        can_mention_any = False

    allowed_roles = []

    def repl(match):
        rid = int(match.group(1))
        role = guild.get_role(rid)
        if role is None:
            return match.group(0)

        if rid == guild.id or getattr(role, 'is_default', lambda: False)():
            if allow_everyone:
                return match.group(0)
            return 'everyone'

        if role.mentionable or can_mention_any:
            allowed_roles.append(role)
            return match.group(0)
        return '@' + role.name

    return _ROLE_MENTION_RE.sub(repl, text), allowed_roles


def indent(text, spaces):
    prefix = " " * spaces
    return '\n'.join(prefix + line for line in text.splitlines())

async def run_code(code: str, header: str, args: tuple, timeout: int) -> dict:
    warnings = []
    if not re.search(r'(?m)^' + re.escape(header), code):
        warnings.append(f"Your code didn't define `{header}` at the top level, so the system wrapped it for you.")
        code = f"{header}\n{indent(code, 4)}"

    ns = { '__builtins__': __builtins__ }
    out, err = StringIO(), StringIO()

    logger.debug("Agent attempted to run code:")
    logger.debug('\n' + code)
    logger.debug("Running...")

    stdout, stderr = '', ''
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(code, ns)
            result = await asyncio.wait_for(ns['main'](*args), timeout = timeout)

        stdout = out.getvalue()
        stderr = err.getvalue()

        logger.debug("Result: %s", result)
        logger.debug(stdout + stderr)

        return {'warnings': warnings, 'result': result, 'stdout': stdout, 'stderr': stderr}
    except asyncio.TimeoutError:
        logger.debug("Execution timed out.")
        return {'warnings': warnings, 'result': 'Execution timed out.', 'stdout': stdout, 'stderr': stderr}
    except Exception as e:
        import traceback
        logger.debug(traceback.format_exc(), exc_info=e)
        return {'warnings': warnings, 'result': traceback.format_exc(), 'stdout': stdout, 'stderr': stderr}
    

def clean(message: discord.Message):
    if message.guild:

        def resolve_member(id: int) -> str:
            m = message.guild.get_member(id) or discord.utils.get(message.mentions, id=id)  # type: ignore
            return f'@{m.display_name}' if m else '@deleted-user'

        def resolve_role(id: int) -> str:
            r = message.guild.get_role(id) or discord.utils.get(message.role_mentions, id=id)  # type: ignore
            return f'@{r.name}' if r else '@deleted-role'

        def resolve_channel(id: int) -> str:
            c = message.guild._resolve_channel(id)  # type: ignore
            return f'#{c.name}' if c else '#deleted-channel'

    else:

        def resolve_member(id: int) -> str:
            m = discord.utils.get(message.mentions, id=id)
            return f'@{m.display_name}' if m else '@deleted-user'

        def resolve_role(id: int) -> str:
            return '@deleted-role'

        def resolve_channel(id: int) -> str:
            return '#deleted-channel'

    transforms = {
        '@': resolve_member,
        '@!': resolve_member,
        '#': resolve_channel,
        '@&': resolve_role,
    }

    def repl(match: re.Match) -> str:
        type = match[1]
        id = int(match[2])
        transformed = transforms[type](id) + f' (ID: {id})'
        return transformed

    result = re.sub(r'<(@[!&]?|#)([0-9]{15,20})>', repl, message.content)

    return discord.utils.escape_mentions(result)