"""Console output and per-run file capture for the segmentation pipeline's ``logging``.

``ethograph.segment`` enables INFO-level console output on import (see
:func:`enable_console_logging`) — there is no reason to hide session/
materialise/train/infer progress by default. :func:`log_to_file` is on top of
that: a pipeline stage wraps its own body in it so a log of what happened
always lands beside the stage's other outputs too.

:func:`start_session_log` is unrelated to the ``logging`` module records above
— it tees raw ``stdout``/``stderr`` (prints, tracebacks, third-party library
output, not just log records) to a timestamped file under
``~/.ethograph/logs/``, one per GUI session.
"""

from __future__ import annotations

import atexit
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator

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


class _TeeStream:
    """Writes to *stream* as before, plus a timestamp-prefixed copy to *log_file*."""

    def __init__(self, stream: IO[str], log_file: IO[str]) -> None:
        self._stream = stream
        self._log_file = log_file
        self._at_line_start = True

    def write(self, data: str) -> int:
        self._stream.write(data)
        for line in data.splitlines(keepends=True):
            if self._at_line_start:
                self._log_file.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ")
            self._log_file.write(line)
            self._at_line_start = line.endswith("\n")
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def start_session_log(prefix: str = "session") -> Path:
    """Tee ``stdout``/``stderr`` to a fresh timestamped file under ``~/.ethograph/logs/``.

    Captures everything a session prints to the terminal — log records,
    plain ``print()`` calls, uncaught tracebacks, third-party library
    chatter — so it survives after the terminal window that launched it is
    gone. Call once, as early as possible in a long-running entry point
    (e.g. the GUI's ``ethograph launch``).

    Returns
    -------
    Path
        The log file's path (also printed to the console).
    """
    from ethograph.utils.paths import logs_dir as _logs_dir

    logs_dir = _logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    path = logs_dir / f"{prefix}_{timestamp:%Y%m%d_%H%M%S}.log"
    log_file = path.open("w", encoding="utf-8")
    log_file.write(f"# ethograph session log started {timestamp:%Y-%m-%d %H:%M:%S}\n")
    atexit.register(log_file.close)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)
    return path
