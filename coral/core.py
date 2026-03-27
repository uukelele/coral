import discord
import os

import typer

from .config import load_config
from .bot import CoralBot
from .history import init_db
from .agent import agent

def main():
    config = load_config()

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if config.AI_OPENAI_COMPATIBLE_BASE_URL:
        model = OpenAIChatModel(
            config.AI_MODEL_NAME,
            provider = OpenAIProvider(
                base_url = config.AI_OPENAI_COMPATIBLE_BASE_URL,
                api_key  = config.AI_API_KEY or os.getenv('AI_API_KEY') or 'X', # some APIs are keyless
            )
        )
    else:
        model = config.AI_MODEL_NAME
        # google-gla:gemini-flash-latest -> GOOGLE_API_KEY
        # xai:grok-4-1-fast-non-reasoning -> XAI_API_KEY
        # openai:gpt-5.2 -> OPENAI_API_KEY

        os.environ[model.split(':')[0].split('-')[0].upper() + '_API_KEY'] = config.AI_API_KEY

    engine = init_db(config.DB_PATH)

    intents = discord.Intents.all()

    client = CoralBot(
        config  = config,
        agent   = agent,
        model   = model,
        intents = intents,
        engine  = engine,
    )

    token = config.DISCORD_TOKEN or os.getenv('DISCORD_TOKEN')

    if not token:
        typer.secho("DISCORD_TOKEN not found in config or environment variables. Please set it and rerun the command.", fg='red')
        typer.Exit(1)

    client.run(token)

if __name__ == "__main__":
    main()