import typer, logging, re, httpx
from pathlib import Path

def _fmt(record: logging.LogRecord) -> str:
    _ARG_RE = re.compile(r"%(?:[-+#0 ]*\d*(?:\.\d+)?)?[diouxXeEfFgGcrsa]")

    if not record.args:
        return str(record.msg)

    message = str(record.msg)
    args = record.args

    if isinstance(args, dict):
        return message % {
            key: typer.style(str(value), bold=True)
            for key, value in args.items()
        }

    parts = []
    arg_index = 0
    position = 0

    for match in _ARG_RE.finditer(message):
        parts.append(message[position:match.start()])

        if arg_index < len(args):
            # parts.append(typer.style(str(args[arg_index]), bold=True))
            #                    ^^^^^ this function resets EVERYTHING after so we can't use it as it will turn off other styles
            #                          instead, we can use this ─╮
            parts.append(f"\033[1m{args[arg_index]}\033[22m") # ←╯

            arg_index += 1

        position = match.end()

    parts.append(message[position:])

    result = "".join(parts)
    result = result.replace("%%", "%")

    return result


# https://github.com/fastapi/typer/issues/203#issuecomment-840690307
class CoralLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:

        text = _fmt(record)

        if record.levelno >= logging.CRITICAL:
            typer.secho(
                typer.style(f"[ERROR] {text}"),
                fg = typer.colors.BRIGHT_WHITE,
                bg = typer.colors.RED,
            )

        elif record.levelno >= logging.ERROR:
            typer.echo(
                typer.style("[ERROR] ", fg=typer.colors.BRIGHT_RED) +
                typer.style(text, fg=typer.colors.RED),
            )

        elif record.levelno >= logging.WARNING:
            typer.echo(
                typer.style("[WARN] ", fg=typer.colors.YELLOW) +
                typer.style(text, fg=typer.colors.YELLOW),
            )

        elif record.levelno >= logging.INFO:
            typer.echo(
                typer.style("● ", fg=typer.colors.BRIGHT_MAGENTA) +
                typer.style(text, fg=None),
            )

        else:
            typer.echo(
                typer.style(f"○ [{record.name}] ", dim=True) +
                typer.style(text, dim=True),
            )

def setup(verbose: bool = True):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    term_handler = CoralLogHandler()
    term_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(term_handler)

    for name in ("openai", "httpcore"): # ignore these guys
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)