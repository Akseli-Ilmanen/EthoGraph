"""Console output and per-run file capture for the segmentation pipeline's ``logging``.

``ethograph.segment`` enables INFO-level console output on import (see
:func:`enable_console_logging`) — there is no reason to hide session/
materialise/train/infer progress by default. :func:`log_to_file` is on top of
that: a pipeline stage wraps its own body in it so a log of what happened
always lands beside the stage's other outputs too.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_console_configured: set[str] = set()


def enable_console_logging(logger_name: str = "ethograph.segment", level: int = logging.INFO) -> None:
    """Print *logger_name*'s log records (and its children's) to the console.

    Called once when :mod:`ethograph.segment` is imported. Safe to call again
    — it will not add a second handler.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if logger_name in _console_configured:
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    _console_configured.add(logger_name)


@contextmanager
def log_to_file(path: Path, logger_name: str = "ethograph.segment", level: int = logging.INFO) -> Iterator[None]:
    """Capture *logger_name*'s records to *path* for the duration of the block.

    Every pipeline stage writes its own log file beside its other outputs.
    Restores the logger's previous level and handlers on exit, so
    nested/repeated calls (e.g. a benchmark loop over several runs) never
    leak a handler into the next call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    previous_level = logger.level
    if previous_level == logging.NOTSET or previous_level > level:
        logger.setLevel(level)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
