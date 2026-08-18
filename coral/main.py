import typer
from pathlib import Path
import yaml
import os
import subprocess as sp
import logging

from .config import load_config, Config, Tier, TOOL_GROUPS
from .prompts import DEFAULT_EXTRA_PROMPT
from . import log

app = typer.Typer()

logger = logging.getLogger(__name__)

@app.callback()
def main(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Disable verbose output.",
    ),
):
    log.setup(not quiet)

@app.command(name='create-docker')
def create_dockerfiles(path: Path = typer.Argument(Path('.')), force=False):
    path = path.resolve()
    if not (path / 'config.yaml').exists() and not force:
        # typer.secho('`config.yaml` does not exist in this directory. Please run `coral create` first, or pass --force=True.', fg='yellow')
        return

    p_dockerfile = path / 'Dockerfile'
    p_compose    = path / 'docker-compose.yml'

    if p_dockerfile.exists() and p_compose.exists() and not force:
        # typer.secho('`Dockerfile` and `docker-compose.yml` already exist. Please remove them first first, or pass --force=True.', fg='yellow')
        return
    
    repo = Path(__file__).resolve().parent.parent
    from_source = (repo / 'pyproject.toml').exists()

    dockerfile = """
FROM python:3.13

WORKDIR /workspace
"""
    if not from_source:
        dockerfile += """
RUN pip install git+https://github.com/uukelele/coral.git
"""
    else: "Coral is installed at runtime from a mounted volume. This is for easier development."

    dockerfile += """
CMD ["python", "-m", "coral.core"]
"""

    compose = f"""
services:
    bot:
        build: .
        container_name: coral-{path.name.lower().replace(' ', '-')}
        restart: unless-stopped
        volumes:
            - .:/workspace
"""
    
    if from_source:
        compose += f"""
            - {repo}:/opt/coral:ro
        
        command: /bin/sh -c "mkdir -p /tmp/coral && cp -au /opt/coral/. /tmp/coral && pip install /tmp/coral && python -m coral.core"
"""
        
    if not p_dockerfile.exists():
        logger.debug("Writing %s...", "Dockerfile")
        # typer.secho("[+] Writing Dockerfile...")
        p_dockerfile.write_text(dockerfile)

    if not p_compose.exists():
        logger.debug("Writing %s...", "docker-compose.yml")
        # typer.secho("[+] Writing docker-compose.yml...")
        p_compose.write_text(compose)



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
        logger.error("Path %s must be an empty folder.", str(path))
        # typer.secho(f"Path {path} must be an empty folder.", fg='red')
        raise typer.Exit(1)
    
    if not path.exists():
        logger.debug("Creating directory %s...", str(path))
        # typer.secho(f"[+] Creating directory {path}...", fg='green')
        path.mkdir(parents=True, exist_ok=True)

    name = path.name

    base_config = Config(
        DISCORD_TOKEN  = "Paste your Discord token here.",
        DISCORD_PREFIX = '--',

        TIERS = {
            'admin': Tier(
                allowed_roles_or_user_ids = [1234],
                allowed_tools = ['*'],
                ratelimit = None,
            ),
            'default': Tier(
                allowed_tools = ['@memory', '@discord', '@web', '@media'],
                allow_chat = True,
                ratelimit = '6/m'
            ),
        },

        AI_MODEL_NAME = 'google-gla:gemini-flash-latest',
        AI_API_KEY = "Put your API key here.",
        AI_OPENAI_COMPATIBLE_BASE_URL = None,
        AI_ANTHROPIC_COMPATIBLE_BASE_URL = None,
        AI_EXTRA_CONTEXT_PATH = 'config.md.j2',

        DB_PATH = 'sqlite:///memory.db',
    )

    config = yaml.dump(base_config.model_dump(mode='json'), sort_keys=False)

    auto_lines = ["These are the available tools that you can write:"]
    for group, tools in TOOL_GROUPS.items():
        auto_lines.append(f'  @{group} --------------')
        for tool in tools:
            auto_lines.append(f'    > {tool}')
        auto_lines.append('')

    auto_lines += ['', '', 'You can also do things like ["*", "!search_discord"], or ["@web", "!web_fetch"], or even ["*", "!@media"], or even ["!*", "get_user_info"]']
    comment = "\n".join(f"# {l}" for l in auto_lines)


    out = []
    for line in config.splitlines():
        if line.startswith("TIERS:"):
            out.append(comment)
        out.append(line)

    config = '\n'.join(out)

    (path / 'config.yaml').write_text(config)
    (path / 'config.md.j2').write_text(DEFAULT_EXTRA_PROMPT.render(path=path))

    create_dockerfiles(path)

    logger.info(f"Success! All set up! Now go and customize your bot!")

@app.command()
def clear(path: Path = typer.Argument(Path('.'))):
    os.chdir(path.resolve())

    config = load_config()

    from urllib.parse import urlparse, unquote

    parsed = urlparse(config.DB_PATH)

    if parsed.scheme != 'sqlite':
        logger.error('The database is not a SQLite .db file. Coral cannot find the database path to clear.')
        raise typer.Exit(1)

    db_path: Path

    if parsed.path.startswith('/'):
        db_path = Path(unquote(parsed.path))
    else:
        db_path = Path(unquote(parsed.path.lstrip('/')))

    if not db_path: return

    confirm = (input(f"Clear memory file at {db_path}? [y/N]: ").strip().lower() or 'n')[0] == 'y'

    if confirm:
        db_path.unlink()
        logger.info("Memory cleared successfully.")

    reminders = path / 'reminders.db'
    if reminders.exists():
        if (input(f"Clear reminders at {reminders}? [y/N]: ").strip().lower() or 'n')[0] == 'y':
            reminders.unlink()
            logger.info("Reminders cleared successfully.")

    if (path / 'docker-compose.yml').exists():
        logger.debug("Shutting down and removing container...")
        sp.run(['docker', 'compose', 'down', '-v'])
        logger.info("Workspace cleared.")


@app.command()
def run(path: Path = typer.Argument(Path('.'))):
    os.chdir(path.resolve())

    create_dockerfiles(path)

    logger.info("Booting Coral...")
    
    try:
        sp.run(['docker', 'compose', 'up', '--build'])
    except KeyboardInterrupt:
        logger.error('\nStopping workspace...')
        sp.run(["docker", "compose", "stop"])
        logger.info("Stopped.")

    

if __name__ == "__main__":
    app()