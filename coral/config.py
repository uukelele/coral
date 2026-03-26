from pydantic import BaseModel
from pathlib import Path
from typing import *
import yaml

class Config(BaseModel):
    DISCORD_TOKEN: Optional[str] = None
    DISCORD_PREFIX: str = '-- '
    DISCORD_ALLOWED_USER_OR_ROLE_IDS: Optional[List[int]] = None
    
    AI_MODEL_NAME: str
    AI_API_KEY: Optional[str] = None
    AI_OPENAI_COMPATIBLE_BASE_URL: Optional[str] = None
    AI_EXTRA_CONTEXT_PATH: str = 'config.md.j2'

    DB_PATH: str = 'sqlite:///memory.db'



def load_config(path: str | Path = 'config.yaml') -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist. Please created it and add the required fields. Quickstart: `coral create {path.parent}`")

    return Config.model_validate(yaml.full_load(path.read_text()))