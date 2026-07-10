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

def sanitize_role_mentions(text: str, guild, channel, member, allow_everyone: bool = False):
    """
    For each role mention (`<@&id>`) in outgoing text, keep it as a real ping only
    if `member` (the user who triggered the bot) is actually allowed to ping that
    role in `channel`; otherwise replace it with the plain-text role name (e.g.
    `<@&1088558118113378434>` -> `@Member`).

    The `@everyone` role shares its id with the guild id, so `<@&guild_id>` is a
    mass mention in disguise; it is governed by `allow_everyone` (and stripped to
    plain `everyone` when not allowed) rather than the per-role logic.

    A user may ping a normal role when the role is `mentionable`, or when the user
    has the "Mention @everyone, @here, and All Roles" permission in that channel
    (which Administrator implies).

    Returns `(sanitized_text, allowed_role_objects)`. `allowed_role_objects` is the
    list of roles that were kept as real pings, suitable for passing straight to
    `discord.AllowedMentions(roles=...)` as a hard, API-level safety net.
    """
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
            # Unknown/deleted role won't ping anyone real; leave untouched.
            return match.group(0)

        # The @everyone role is a mass mention; never let it fall through to the
        # `@{name}` path (its name is literally "@everyone", which would recreate a
        # live ping). It is controlled solely by `allow_everyone`.
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