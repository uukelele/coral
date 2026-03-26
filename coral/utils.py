def chunk_string(s: str, size: int = 2000):
    return [s[i:i+size] for i in range(0, len(s), size)]

def indent(text, spaces):
    prefix = " " * spaces
    return '\n'.join(prefix + line for line in text.splitlines())

import discord
import re

def clean(message: discord.Message):
    if message.guild:

        def resolve_member(id: int) -> str:
            m = message.guild.get_member(id) or utils.get(message.mentions, id=id)  # type: ignore
            return f'@{m.display_name}' if m else '@deleted-user'

        def resolve_role(id: int) -> str:
            r = message.guild.get_role(id) or utils.get(message.role_mentions, id=id)  # type: ignore
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