import discord
import typer
from pathlib import Path
import yaml
import os
import docker
import docker.errors
import hashlib
from alive_progress import alive_bar

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

        DB_PATH = 'sqlite:///memory.db',
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

    confirm = (input(f"Clear memory file at {db_path}? [y/N]: ").strip().lower() or 'n')[0] == 'y'

    if confirm:
        db_path.unlink()
        typer.secho("Memory cleared successfully.", fg='green')

    container_id = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    container_name = f'coral-workspace-{container_id}'

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        container.remove(force=True)
        typer.secho(f"Deleted workspace <{container_id}>.", fg='green')
    except docker.errors.NotFound:
        typer.secho("No workspace found to clean.", fg='white')


@app.command()
def run(path: Path = typer.Argument(Path('.'))):
    os.chdir(path.resolve())

    client = docker.from_env()
    image = 'python:3.12'

    typer.secho("Checking for Docker image...", fg='white')
    
    try:
        client.images.get(image)
        # typer.secho("Image found!", fg='green')
    except docker.errors.ImageNotFound:
        typer.secho(f"Image not found. Pulling {image}...", fg='yellow')

        layers = {}
        with alive_bar(1, title="Pulling image") as bar:
            last_total = 0
            last_current = 0

            for line in client.api.pull(image, stream=True, decode=True):
                if 'id' in line and 'progressDetail' in line:
                    layer_id = line['id']
                    detail = line['progressDetail']

                    total = detail.get('total', 0)
                    current = detail.get('current', 0)

                    if total:
                        layers[layer_id] = (current, total)

                        total_sum = sum(t for _, t in layers.values())
                        current_sum = sum(c for c, _ in layers.values())

                        bar.text = f"Layers: {len(layers)} active"

                        if total_sum != last_total:
                            bar.total = total_sum
                            last_total = total_sum

                        delta = current_sum - last_current
                        if delta > 0:
                            bar(delta)
                            last_current = current_sum

    container_id = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    container_name = f'coral-workspace-{container_id}'

    try:
        container = client.containers.get(container_name)
        if container.status == 'running':
            typer.secho(f"Workspace <{container_id}> already running. Re-attaching...", fg='yellow')
        else:
            typer.secho(f"Starting workspace <{container_id}>...", fg='green')
            container.start()

    except docker.errors.NotFound:
        typer.secho(f"Creating new workspace <{container_id}>...", fg='green')

        installed_from_source = False

        coral_repo = Path(__file__).resolve().parent.parent
        if not (coral_repo / 'pyproject.toml').exists():
            # Coral is not installed from source.
            ...
        else:
            installed_from_source = True

        volumes = {
            str(path.resolve()): { 'bind': '/workspace', 'mode': 'rw' }
        }

        source: str
        setup: str

        if installed_from_source:
            volumes[str(coral_repo)] = { 'bind': '/opt/coral', 'mode': 'ro' }
            source = '/tmp/coral'
            setup = 'mkdir -p /tmp/coral && cp -a /opt/coral/. /tmp/coral'
        else:
            setup = "apt-get update -y && apt-get install -y git"
            source = 'git+https://github.com/uukelele/coral.git'

        cmd = f'/bin/sh -c "{setup} && pip install -q uv && uv pip install --system {source} && python -m coral.core"'

        with alive_bar(total=None, title=typer.style(f'Booting workspace <{container_id}>...', fg='green')) as bar:
            typer.secho(cmd, fg='white')

            container = client.containers.run(
                image,
                name = container_name,
                detach = True,
                working_dir = '/workspace',
                volumes = volumes,
                command = cmd,
            )

    try:
        for log in container.logs(stream=True, follow=True):
            print(log.decode(), end='')
    except KeyboardInterrupt:
        pass

    typer.secho('Stopping workspace...', fg='red')
    container.stop(timeout=3)
    typer.secho('Stopped.', fg='white')

    

if __name__ == "__main__":
    app()