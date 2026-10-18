import os
import httpx2, tomllib
from pathlib import Path
import logging

from pydantic_ai.models import Model

from .config import Config

logger = logging.getLogger(__name__)

def _coral_version() -> str:
    try:
        pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
        return tomllib.loads(pyproject.read_text())['project']['version']
    except Exception:
        try:
            from importlib.metadata import version
            return version('coral')
        except Exception:
            return 'unknown'

UA = f"coral/{_coral_version()}"

async def _set_ua(request: httpx2.Request):
    request.headers['user-agent'] = UA

def _http_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        timeout = httpx2.Timeout(600.0, connect=15.0), # allow 600 seconds for a model (thinking, etc.) + 15s TTFT
        event_hooks = {'request': [_set_ua]},
    )

def build_model(config: Config) -> Model | str:
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    if config.AI_OPENAI_RESPONSES_COMPATIBLE_BASE_URL:
        return OpenAIResponsesModel(
            config.AI_MODEL_NAME,
            provider = OpenAIProvider(
                base_url = config.AI_OPENAI_RESPONSES_COMPATIBLE_BASE_URL,
                api_key  = config.AI_API_KEY or os.getenv('AI_API_KEY') or os.getenv('OPENAI_API_KEY') or 'X', # some APIs are keyless
                http_client = _http_client(),
            ),
            settings = config.AI_EXTRA_CONFIG,
        )
    elif config.AI_OPENAI_COMPATIBLE_BASE_URL:
        return OpenAIChatModel(
            config.AI_MODEL_NAME,
            provider = OpenAIProvider(
                base_url = config.AI_OPENAI_COMPATIBLE_BASE_URL,
                api_key  = config.AI_API_KEY or os.getenv('AI_API_KEY') or os.getenv('OPENAI_API_KEY') or 'X', # some APIs are keyless
                http_client = _http_client(),
            ),
            settings = config.AI_EXTRA_CONFIG,
        )
    elif config.AI_ANTHROPIC_COMPATIBLE_BASE_URL:
        return AnthropicModel(
            config.AI_MODEL_NAME,
            provider = AnthropicProvider(
                base_url = config.AI_ANTHROPIC_COMPATIBLE_BASE_URL,
                api_key  = config.AI_API_KEY or os.getenv('AI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY') or 'X',
                http_client = _http_client(),
            ),
            settings = config.AI_EXTRA_CONFIG,
        )
    else:
        model = config.AI_MODEL_NAME
        # google-gla:gemini-flash-latest -> GOOGLE_API_KEY
        # xai:grok-4-1-fast-non-reasoning -> XAI_API_KEY
        # openai:gpt-5.2 -> OPENAI_API_KEY

        if config.AI_API_KEY:
            os.environ[model.split(':')[0].split('-')[0].upper() + '_API_KEY'] = config.AI_API_KEY
            # set the env var here, because pydantic-ai reads this. this is also a lazy implementation because
            # i don't want to write a map of {"openai", "OPENAI_API_KEY"} etc.

        return model
