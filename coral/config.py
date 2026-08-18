from pydantic import BaseModel
from pathlib import Path
from typing import *
import yaml, logging, re

logger = logging.getLogger(__name__)

TOOL_GROUPS = {
    'web':            ['duckduckgo_search', 'web_fetch'],
    'media':          ['analyse_file'],
    'discord':        ['get_user_info'],
    'discord:admin':  ['search_discord'], # this is in the `discord:admin` group as search_discord can access even hidden/private/admin channels which we don't always want to expose
    'planning':       ['write_plan', 'read_plan', 'add_task', 'update_task_status', 'update_task_statuses', 'remove_task', 'add_subtask', 'set_dependency', 'get_available_tasks', 'delegate_task'],
    'coder':          [
                        'read_file', 'write_file', 'edit_file', 'list_directory', 'search_files', 'find_files', 'create_directory', 'file_info', 'attach_file', # filesystem # attach file is here as it provides arbitrary file read (constrained to /workspace, but still file read of sensitive e.g. config.yaml)
                        'run_command', 'start_command', 'check_command', 'stop_command', 'run_shell', 'run_code', 'trigger_reboot', # code execution
                        'create_service', 'list_services', 'service_status', 'stop_service', 'restart_service', 'remove_service', # services
                      ],
    'memory':         [
                        'set_automation', 'cancel_automation', 'list_automations', # automations
                        'read_memory', 'write_memory', 'search_memory', 'delete_memory', # memory
                      ],
    'moderation':     ['timeout_user', 'ban_user', 'unban_user', 'cancel_timeout'],
}

_UNITS = {
    'seconds': 1, 'second': 1, 'secs': 1, 'sec': 1, 's': 1,
    'minutes': 60, 'minute': 60, 'mins': 60, 'min': 60, 'm': 60,
    'hours': 3600, 'hour': 3600, 'hrs': 3600, 'hr': 3600, 'h': 3600,
    'days': 86400, 'day': 86400, 'd': 86400,
}

def parse_limit(limit: str) -> tuple[int, int]:
    if not isinstance(limit, str):
        raise ValueError(f"Invalid rate limit: {limit!r}")

    count_part, sep, period = limit.replace(' per ', '/').partition('/')
    if not sep:
        raise ValueError(f"Invalid rate limit: {limit!r} (expected a form like '5/m' or '10 per 30s')")

    try:
        count = int(count_part.strip())
    except ValueError:
        raise ValueError(f"Invalid rate limit count: {count_part.strip()!r}") from None

    match = re.fullmatch(r'(\d+)?\s*([a-z]+)', period.strip().lower())
    if not match:
        raise ValueError(f"Invalid rate limit period: {period.strip()!r}")

    multiplier = int(match.group(1) or 1)
    unit = match.group(2)
    if unit not in _UNITS:
        raise ValueError(f"Invalid rate limit unit: {unit!r} (expected one of s, m, h, d or their full names)")

    return count, multiplier * _UNITS[unit]

class Tier(BaseModel):
    allowed_roles_or_user_ids: Optional[List[int]] = None
    allowed_tools: List[str] = []
    allow_chat: bool = True
    allow_ping_everyone: bool = False
    ratelimit: Optional[str] = '6/m'

    def can_use_tool(self, tool_name: str) -> bool:
        kanuze = False # kanuze = can use. get it? no? ok..

        for rule in self.allowed_tools:
            neg = rule.startswith('!')
            body = rule[1:] if neg else rule
            if body == '*': # wildcard, match all
                match = True # don't return yet, there could be a neg of this underneath
            elif body.startswith('@'): # group
                match = tool_name in TOOL_GROUPS.get(body[1:], ())
            else:
                match = body == tool_name

            if match:
                kanuze = not neg

        return kanuze

    def parsed_ratelimit(self) -> Optional[tuple[int, int]]:
        return parse_limit(self.ratelimit) if self.ratelimit else None


class Config(BaseModel):
    DISCORD_TOKEN: Optional[str] = None
    DISCORD_PREFIX: str = '-- '
    TIERS: Optional[Dict[str, Tier]] = None

    AI_MODEL_NAME: str
    AI_API_KEY: Optional[str] = None
    AI_OPENAI_COMPATIBLE_BASE_URL: Optional[str] = None
    AI_ANTHROPIC_COMPATIBLE_BASE_URL: Optional[str] = None
    AI_EXTRA_CONTEXT_PATH: str = 'config.md.j2'
    AI_EXTRA_CONFIG: dict[str, Any] = {}

    DB_PATH: str = 'sqlite:///memory.db'

    def resolve_tier(self, user_id: int, role_ids: Optional[Iterable[int]] = None) -> Tier:
        if not self.TIERS: return Tier()

        candidate_ids = {user_id, *(role_ids or [])}

        for name, tier in self.TIERS.items():
            if name == 'default':
                continue
            ids = tier.allowed_roles_or_user_ids
            if ids and candidate_ids.intersection(ids):
                return tier

        return self.TIERS.get('default') or Tier()



def load_config(path: str | Path = 'config.yaml') -> Config:
    logger.debug("Loading config from %s", path)

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist. Please created it and add the required fields. Quickstart: `coral create {path.parent}`")

    return Config.model_validate(yaml.full_load(path.read_text()))
