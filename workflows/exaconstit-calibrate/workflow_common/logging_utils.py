"""
Logging utilities based on Python's standard ``logging`` module.

Purpose
-------
Replaces the old ``ExaConstit_Logger`` module, which relied on a
module-level global ``logger`` variable that had to be explicitly
initialized before use. That pattern had two problems:

1. If any module imported before ``initialize_ExaProb_log`` was called
   tried to log something, the global did not yet exist and logging
   silently did nothing.
2. The compatibility shim printed every log line twice - once through
   the logger and once through ``print`` - which produced unreadable
   output on terminals that were already capturing the logger.

Standard Python logging solves both problems. Every module acquires
its own named logger via ``get_logger(__name__)``. Logger names form
a hierarchy, so filtering can be done per-subsystem without code
changes. Configuration is done exactly once, at application startup,
via ``configure_logging``.

Migration path for existing code
--------------------------------
The old names ``initialize_ExaProb_log`` and ``write_ExaProb_log`` are
re-exported as thin shims that forward to the new machinery. Existing
scripts keep working with no edits. New code should use the standard
pattern::

    from workflow_common import get_logger
    logger = get_logger(__name__)
    logger.info("hello")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Mapping from the friendly string names the old logger used to the
# numeric levels the stdlib logging module wants. Exposed as a module
# constant so tests can reference it and so the set of supported
# names is discoverable.
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(name: str) -> logging.Logger:
    """Return the logger with the given name.

    Standard usage is ``logger = get_logger(__name__)`` at the top of
    a module. This gives the logger a name that matches the module's
    import path, which in turn makes it easy to enable debug logging
    for just one subsystem (``logging.getLogger("workflow_common").
    setLevel("DEBUG")``) without flooding the screen with messages
    from the rest of the program.

    Args:
        name: A dotted name for the logger. Conventionally
            ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance. Repeated calls with the same
        name return the same object - Python's logging module caches
        loggers by name internally.
    """
    return logging.getLogger(name)


def configure_logging(
    *,
    level: Union[str, int] = "info",
    logfile: Optional[Union[str, Path]] = None,
    append: bool = False,
    stream: bool = True,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the root logger for the whole application.

    Intended to be called exactly once, at program startup, before any
    heavy work begins. Every logger obtained via ``get_logger`` then
    inherits the format and handlers set up here.

    Calling this function a second time is safe: existing handlers are
    removed first, so a restart scenario that wants to reopen its log
    file in append mode can simply call this again with ``append=True``.

    Args:
        level: Minimum level of messages to emit. May be the string
            "debug" / "info" / "warning" / "error" / "critical", or
            a numeric value from the ``logging`` module
            (e.g. ``logging.INFO``). Defaults to ``"info"``.
        logfile: If given, log messages are written to this file in
            addition to the stream handler. The file's parent
            directory is created if needed. If omitted, logging goes
            to stderr only.
        append: If True and ``logfile`` is set, the file is opened in
            append mode so existing log content is preserved. Use
            this on restart. If False (the default), the file is
            truncated.
        stream: If True (default), messages are also written to
            stderr. Useful for interactive runs. Turn off to log only
            to file on cluster runs where the stdout is already being
            captured by the job scheduler.
        fmt: Python logging format string. The default includes a
            timestamp, level, logger name, and message. See
            ``logging.Formatter`` docs for available fields.
        datefmt: Date/time format for the ``asctime`` field.

    Example:
        Normal interactive run with a log file::

            configure_logging(level="info", logfile="opt.log")

        Restart case, picking up the previous log::

            configure_logging(level="info", logfile="opt.log",
                              append=True)

        Cluster run, logfile only (the scheduler is already capturing
        stdout/stderr, no need to duplicate)::

            configure_logging(level="info", logfile="opt.log",
                              stream=False)
    """
    # Allow callers to pass a string like "info" or an already-resolved
    # numeric level constant like logging.INFO. Coerce string inputs
    # using the table defined at module top; unknown strings fall back
    # to INFO rather than raising, because a typo in a level name
    # should not kill an optimization run halfway through.
    if isinstance(level, str):
        level = _LEVELS.get(level.lower(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Wipe existing handlers so repeat calls to configure_logging do
    # not produce duplicated output. Without this, configure_logging
    # called twice in one process would add a second stderr handler
    # and every message would print twice.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        fh = logging.FileHandler(str(logfile), mode=mode, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if stream:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setFormatter(formatter)
        root.addHandler(sh)


# --- Backwards-compatibility shim -----------------------------------------
#
# These two names (``initialize_ExaProb_log`` / ``write_ExaProb_log``)
# exist so that existing driver scripts and modules that reference the
# old API keep working during the transition. New code should NOT use
# them - use ``configure_logging`` and ``get_logger(__name__).info(...)``
# instead.
#
# Once every caller has been migrated, the shim can be deleted.

_compat_logger = logging.getLogger("ExaProb")


def initialize_ExaProb_log(
    glob_loglvl: str = "debug",
    filename: str = "logbook_ExaProb.log",
    restart: bool = False,
) -> None:
    """Deprecated. Use :func:`configure_logging` instead.

    Kept as a thin forwarding shim so existing ``ExaConstit_NSGA3.py``
    and other driver scripts continue to work without edits during the
    migration.

    Args:
        glob_loglvl: Old-style string log level ("debug", "info", ...).
        filename: Path to log file.
        restart: If True, append rather than overwrite. Matches the
            old kwarg name exactly for drop-in compatibility.
    """
    configure_logging(level=glob_loglvl, logfile=filename, append=restart)


def write_ExaProb_log(
    text: str, type: str = "info", changeline: bool = False
) -> None:
    """Deprecated. Use ``get_logger(__name__).info(...)`` etc. instead.

    Forwards to the ``ExaProb`` logger so the existing call sites in
    ``ExaConstit_Problems.py`` and ``flux_map.py`` continue to produce
    output during the transition. The ``changeline`` parameter from
    the old API is emulated by logging an empty line before the real
    message.

    Args:
        text: Message text to log.
        type: Severity string. Accepts "debug", "info", "warning",
            "error". Anything else is treated as "info".
        changeline: If True, emit a blank line before the message.
            This was the old module's way of visually separating log
            sections; stdlib logging provides no direct equivalent
            so we fake it.
    """
    if changeline:
        _compat_logger.info("")
    level_name = type.lower()
    if level_name == "error":
        _compat_logger.error("ERROR: %s", text)
    elif level_name == "warning":
        _compat_logger.warning("WARNING: %s", text)
    elif level_name == "debug":
        _compat_logger.debug("%s", text)
    else:
        _compat_logger.info("%s", text)
