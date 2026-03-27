import discord
import typer
from pathlib import Path
import yaml
import os

from .config import load_config, Config
from .history import init_db
from .agent import agent
from .bot import CoralBot
from . import prompts

app = typer.Typer()

@app.command()
def create(path: Path = typer.Argument(Path('.'))):
    path = path.resolve()
    if (
        path.exists()
        and
        (
            path.is_file()
            or
            path.is_dir() and not any(path.iterdir())
        )
    ):
        typer.secho(f"Path {path} must be an empty folder.", fg='red')
        raise typer.Exit(1)
    
    if not path.exists():
        typer.secho(f"[+] Creating directory {path}...", fg='green')
        path.mkdir(parents=True, exist_ok=True)

    name = path.name

    base_config = Config(
        DISCORD_TOKEN  = "Paste your Discord token here.",
        DISCORD_PREFIX = '--',
        DISCORD_ALLOWED_USER_OR_ROLE_IDS = None,

        AI_MODEL_NAME = 'google-gla:gemini-flash-latest',
        AI_API_KEY = "Put your API key here.",
        AI_OPENAI_COMPATIBLE_BASE_URL = None,
        AI_EXTRA_CONTEXT_PATH = 'config.md.j2',

        DB_PATH = 'sqlite:///memory.db'
    )

    (path / 'config.yaml').write_text(yaml.dump(base_config.model_dump(mode='json')))
    (path / 'config.md.j2').write_text(prompts.DEFAULT_EXTRA_PROMPT.render(path=path))

    typer.secho(f"[+] Success! All set up! Now go and customize your bot!", fg='green')

@app.command()
def clear(path: Path = typer.Argument(Path('.'))):
    os.chdir(path.resolve())

    config = load_config()

    from urllib.parse import urlparse, unquote

    parsed = urlparse(config.DB_PATH)

    if parsed.scheme != 'sqlite':
        typer.secho('The database is not a SQLite .db file. Coral cannot find the database path to clear.', fg='red')
        raise typer.Exit(1)

    db_path: Path

    if parsed.path.startswith('/'):
        db_path = Path(unquote(parsed.path))
    else:
        db_path = Path(unquote(parsed.path.lstrip('/')))

    if not db_path: return

    confirm = input(f"Clear memory file at {db_path}? [y/N]: ").strip().lower()[0] == 'y'

    if not confirm: return

    db_path.unlink()

    typer.secho("Memory cleared successfully.", fg='green')

@app.command()
def run(path: Path = typer.Argument(Path('.'))):
    os.chdir(path.resolve())

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
    app()