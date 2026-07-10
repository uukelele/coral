def chunk_string(s: str, size: int = 2000):
    return [s[i:i+size] for i in range(0, len(s), size)]

import re as _re

_MASS_MENTION_RE = _re.compile(r'@(everyone|here)')

def neutralize_mass_mentions(text: str) -> str:
    """
    Hard-strip `@everyone` / `@here` mass mentions from outgoing text by removing
    the leading `@`, turning them into the harmless words `everyone` / `here`.
    """
    return _MASS_MENTION_RE.sub(r'\1', text)

_ROLE_MENTION_RE = _re.compile(r'<@&([0-9]{15,20})>')

def sanitize_role_mentions(text: str, guild, channel, member) -> str:
    """
    For each role mention (`<@&id>`) in outgoing text, keep it as a real ping only
    if `member` (the user who triggered the bot) is actually allowed to ping that
    role in `channel`; otherwise replace it with the plain-text role name (e.g.
    `<@&1088558118113378434>` -> `@Member`).

    A user may ping a role when the role is `mentionable`, or when the user has the
    "Mention @everyone, @here, and All Roles" permission in that channel (which
    Administrator implies).
    """
    if guild is None or member is None:
        return text

    try:
        can_mention_any = channel.permissions_for(member).mention_everyone
    except Exception:
        can_mention_any = False

    def repl(match):
        role = guild.get_role(int(match.group(1)))
        if role is None:
            # Unknown/deleted role won't ping anyone real; leave untouched.
            return match.group(0)
        if role.mentionable or can_mention_any:
            return match.group(0)
        return '@' + role.name

    return _ROLE_MENTION_RE.sub(repl, text)


def indent(text, spaces):
    prefix = " " * spaces
    return '\n'.join(prefix + line for line in text.splitlines())

import discord
import re

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