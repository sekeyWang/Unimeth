"""Logging setup used only by the inference package."""

import logging
from typing import TextIO


INFERENCE_LOGGER_NAME = "unimeth.inference"
_HANDLER_MARKER = "_unimeth_inference_handler"


def _is_main_process() -> bool:
    """Return whether this process is the global main inference process."""
    try:
        from accelerate.state import PartialState

        return PartialState().is_main_process
    except Exception:
        return True


class _MainProcessFilter(logging.Filter):
    """Allow records only on the global main process."""

    def __init__(self, is_main_process: bool):
        super().__init__()
        self.is_main_process = is_main_process

    def filter(self, record: logging.LogRecord) -> bool:
        return self.is_main_process


def configure_inference_logging(
    level: int = logging.INFO,
    *,
    is_main_process: bool | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure and return the package-level inference logger."""
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
