import logging
import sys
from pathlib import Path
from typing import Optional


class LogHandler:
    """
    Configures and manages a Python logger that writes to both a file
    and optionally to stdout. Designed to be instantiated once early in
    the pipeline; subsequent calls with the same name reuse the existing
    logger and skip re-configuration.
    """

    DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DEFAULT_LOG_LEVEL = logging.INFO

    def __init__(
            self,
            name: str = "cloud-cost-anomaly-detection-pipeline",
            log_path: str = "logs/app.log",
            log_level: Optional[int] = None,
            log_format: Optional[str] = None,
            show_console: bool = False,
        ) -> None:
        """
        Sets up the internal logger with the given name. If a logger with
        this name already has handlers attached, it is reused as-is to
        prevent duplicate log entries.

        name (str): Logger namespace used by Python's logging module.
        log_path (str): Filesystem path for the log file; parent directories
            are created automatically if they do not exist.
        log_level (int, optional): Logging threshold, e.g. logging.DEBUG.
            Defaults to INFO.
        log_format (str, optional): Custom format string for log lines.
        show_console (bool): When True, log messages are also written to stdout.
        """
        if not log_path:
            raise ValueError("Log path cannot be empty")

        self._logger = logging.getLogger(name)
        self.__show_console = show_console
        self._logger.propagate = False

        if self._logger.handlers:
            return

        self.__configure_logger(
            log_path=log_path,
            log_level=log_level if log_level is not None else self.DEFAULT_LOG_LEVEL,
            log_format=log_format if log_format is not None else self.DEFAULT_LOG_FORMAT,
        )

    def __configure_logger(
            self,
            log_path: str,
            log_level: int,
            log_format: str,
        ) -> None:
        """
        Attaches a file handler (and optionally a console handler) to the
        logger. Creates the log directory tree if it doesn't exist and
        raises PermissionError if directory or file creation fails.

        log_path (str): Full path for the output log file.
        log_level (int): Minimum severity level to record.
        log_format (str): Format string passed to logging.Formatter.
        """
        self._logger.setLevel(log_level)

        try:
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Failed to create log directory: {str(e)}")

        formatter = logging.Formatter(
            fmt=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self.__show_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(" >> | %(message)s"))
            self._logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler(
                filename=log_path,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        except Exception as e:
            raise PermissionError(f"Failed to create log file: {str(e)}")

    def get_logger(self) -> logging.Logger:
        """Returns the configured logging.Logger instance."""
        return self._logger
