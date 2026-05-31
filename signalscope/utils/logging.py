"""Logging utilities."""

import logging
import sys


def setup_logging(level: str = "INFO", fmt: str = None) -> None:
    """
    Configure logging for SignalScope.

    Parameters
    ----------
    level : str
        Log level: DEBUG, INFO, WARNING, ERROR.
    fmt : str, optional
        Custom format string.
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger = logging.getLogger("signalscope")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers = [handler]
