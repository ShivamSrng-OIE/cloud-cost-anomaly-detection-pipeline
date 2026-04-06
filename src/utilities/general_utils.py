from rich.console import Console
from src.utilities.log_handler import LogHandler


console = Console()


def console_and_logger(
    logger: LogHandler,
    message: str,
    level: str = "info",
) -> None:
    """
    Prints a rich-formatted message to the console and simultaneously
    writes it to the log file through the provided logger. Empty or
    whitespace-only messages are rendered as blank separator lines.

    logger (LogHandler): Active logger whose .info/.warning/.error/.debug
        methods write to the log file.
    message (str): The text to display and record.
    level (str): Severity tier — one of "info", "warning", "error", or
        "debug". Controls both the rich colour tag and the logger method
        used. Defaults to "info".
    """
    if message.strip() == "":
        console.print(" >> |")
        logger.info("")
        return

    if level == "info":
        console.print(f" >> | [bold green]INFO   [/bold green]: {message}")
        logger.info(message)

    elif level == "warning":
        console.print(f" >> | [bold yellow]WARNING[/bold yellow]: {message}")
        logger.warning(message)

    elif level == "error":
        console.print(f" >> | [bold red]ERROR  [/bold red]: {message}")
        logger.error(message)

    elif level == "debug":
        console.print(f" >> | [dim blue]DEBUG  [/dim blue]: {message}")
        logger.debug(message)

    else:
        console.print(f" >> | [bold cyan]LOG    [/bold cyan]: {message}")
        logger.info(message)

    return None
