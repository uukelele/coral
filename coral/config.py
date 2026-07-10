from pydantic import BaseModel
from pathlib import Path
from typing import *
import yaml


class Tier(BaseModel):
    """
    A single permission tier.

    - `allowed_roles_or_user_ids`: the user IDs and/or role IDs that belong to this
      tier. A user belongs to a tier if their own ID, or any of their role IDs, is
      present in this list. This is not required for the special `default` tier,
      which acts as the fallback for anyone who does not match any other tier.
    - `allowed_tools`: the list of tool names this tier may use. Defaults to an empty
      list (no tools). The special value `"*"` allows every tool.
    - `allow_chat`: whether members of this tier may talk to the bot at all.
      Defaults to `True`.
    """

    allowed_roles_or_user_ids: Optional[List[int]] = None
    allowed_tools: List[str] = []
    allow_chat: bool = True

    def can_use_tool(self, tool_name: str) -> bool:
        return '*' in self.allowed_tools or tool_name in self.allowed_tools


class Config(BaseModel):
    DISCORD_TOKEN: Optional[str] = None
    DISCORD_PREFIX: str = '-- '
    # Legacy allow-list. Still fully supported for backward compatibility.
    # Ignored when `tiers` is configured.
    DISCORD_ALLOWED_USER_OR_ROLE_IDS: Optional[List[int]] = None
    # New tiered access control. Insertion order defines rank (highest first).
    # The `default` tier (if present) is used for anyone matching no other tier.
    tiers: Optional[Dict[str, Tier]] = None

    AI_MODEL_NAME: str
    AI_API_KEY: Optional[str] = None
    AI_OPENAI_COMPATIBLE_BASE_URL: Optional[str] = None
    AI_EXTRA_CONTEXT_PATH: str = 'config.md.j2'
    AI_EXTRA_CONFIG: dict[str, Any] = {}

    DB_PATH: str = 'sqlite:///memory.db'

    def resolve_tier(self, user_id: int, role_ids: Optional[Iterable[int]] = None) -> Optional[Tier]:
        """
        Resolve the effective tier for a user given their user ID and role IDs.

        Returns `None` when tiers are not configured, signalling that callers should
        fall back to the legacy `DISCORD_ALLOWED_USER_OR_ROLE_IDS` behaviour.

        When tiers are configured, the user is matched against each tier in order
        (the first tier listed is the highest rank), and the highest matching tier
        is returned. Users matching no tier receive the `default` tier if one is
        defined, otherwise a permissive-chat / no-tools default.
        """
        if not self.tiers:
            return None

        candidate_ids = {user_id, *(role_ids or [])}

        for name, tier in self.tiers.items():
            if name == 'default':
                continue
            ids = tier.allowed_roles_or_user_ids
            if ids and candidate_ids.intersection(ids):
                return tier

        return self.tiers.get('default') or Tier()



def load_config(path: str | Path = 'config.yaml') -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist. Please created it and add the required fields. Quickstart: `coral create {path.parent}`")

    return Config.model_validate(yaml.full_load(path.read_text()))
