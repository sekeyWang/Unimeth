"""Logging setup used only by the inference package."""

import logging
import os
from typing import TextIO


INFERENCE_LOGGER_NAME = "unimeth.inference"
_HANDLER_MARKER = "_unimeth_inference_handler"
_LOG_LEVEL_ENV = "UNIMETH_LOG_LEVEL"
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_log_level(level: int | str | None) -> int:
    """Resolve an explicit level or the inference-only environment setting."""
    if level is None:
        level = os.environ.get(_LOG_LEVEL_ENV, "INFO")
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    try:
        return _LOG_LEVELS[normalized]
    except KeyError as exc:
        choices = ", ".join(_LOG_LEVELS)
        raise ValueError(
            f"{_LOG_LEVEL_ENV} must be one of: {choices}; got {level!r}"
        ) from exc


def _is_main_process() -> bool:
    """Return whether this process is the global main inference process."""
    for variable in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            continue
    return True


class _MainProcessFilter(logging.Filter):
    """Allow records only on the global main process."""

    def __init__(self, is_main_process: bool):
        super().__init__()
        self.is_main_process = is_main_process

    def filter(self, record: logging.LogRecord) -> bool:
        return self.is_main_process


def configure_inference_logging(
    level: int | str | None = None,
    *,
    is_main_process: bool | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the package-level inference logger."""
    level = _resolve_log_level(level)
    logger = logging.getLogger(INFERENCE_LOGGER_NAME)
    logger.disabled = False
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    main_process = _is_main_process() if is_main_process is None else is_main_process
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_MainProcessFilter(main_process))
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)
    return logger
