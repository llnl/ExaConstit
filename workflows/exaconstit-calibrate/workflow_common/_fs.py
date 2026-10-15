"""
Filesystem helpers shared across the workflow framework.

What's in here
--------------
Three small utilities that used to be copy-pasted into every workflow
driver. Centralizing them gives one authoritative implementation that
has been thought about carefully, tested, and documented - so downstream
code can stop worrying about the pitfalls.

1. ``cd`` - an exception-safe context manager for temporarily changing
   the working directory.
2. ``atomic_write_text`` - writing a text file in a way that cannot
   leave a half-written file on disk if the process crashes mid-write.
3. ``atomic_replace`` / ``ensure_dir`` - small wrappers that make intent
   explicit at the call site.

Why "atomic" writes matter on HPC
---------------------------------
On shared filesystems (Lustre, GPFS, NFS) a simulation job can be killed
at any moment when its SLURM allocation runs out. If that kill arrives
while the process is midway through writing a state file, a naive
``open("w")`` leaves a truncated file behind. Any future restart that
reads it will get corrupt or partial data, often without noticing.

The safe idiom, baked into ``atomic_write_text``, is:

1. Create a tempfile in the same directory as the target.
2. Write the new contents and ``fsync`` so bytes are on disk.
3. Rename the tempfile over the target.

POSIX guarantees that the rename operation is atomic when source and
destination live on the same filesystem, so from the point of view of
any other process the file either has the old contents or the new
contents - never a half-written state. This is the same pattern that
editors like vim and tools like git use for their own writes.

NFS caveat: the rename is atomic per POSIX, but ``fsync`` semantics on
NFS depend on mount options. If you need strict durability on NFS,
talk to your sysadmin about ``sync`` mount flags.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

# Type alias for "anything that can be treated as a filesystem path":
# strings, Path objects, or os.PathLike-implementing objects. Using this
# in signatures lets callers pass whichever form they already have
# without explicit conversion.
PathLike = Union[str, os.PathLike]


@contextmanager
def cd(new_path: PathLike) -> Generator[Path, None, None]:
    """Temporarily change the working directory, restoring it on exit.

    This is the exception-safe replacement for::

        old = os.getcwd()
        os.chdir(new_path)
        try:
            ... do work ...
        finally:
            os.chdir(old)

    Using this as a context manager with ``with`` guarantees the
    directory is restored even if an exception is raised inside the
    block, which is especially important during optimization loops
    where a single failed case should not leave the whole driver in
    the wrong directory.

    Args:
        new_path: The directory to change into for the duration of
            the ``with`` block. May be a ``str`` or ``Path``. Tilde
            expansion (``~/foo``) is applied.

    Yields:
        The resolved ``Path`` of the new working directory, so the
        ``as`` clause of the ``with`` statement can use it directly.

    Raises:
        FileNotFoundError: If ``new_path`` does not exist.
        NotADirectoryError: If ``new_path`` exists but is not a directory.

    Example:
        Run a command inside a per-case subdirectory::

            from workflow_common import cd
            import subprocess

            with cd("wf/gen_0/gene_3") as here:
                subprocess.run(["./run_sim.sh"], check=True)
                # Now back to the original directory automatically,
                # even if the subprocess raised.
    """
    new_path = Path(new_path).expanduser()
    saved = Path.cwd()
    os.chdir(new_path)
    try:
        yield new_path
    finally:
        # Even if the body raised, restore the caller's working directory.
        # This is the whole reason to use a context manager instead of a
        # bare chdir pair - exception safety is free here.
        os.chdir(saved)


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = True,
) -> None:
    """Write ``text`` to ``path`` atomically.

    After this function returns successfully, either the file at
    ``path`` has the new contents or it retains its old contents.
    There is no observable state in which a reader can see a
    truncated or half-written file. If the process is killed mid-call,
    a stray tempfile may be left behind in the parent directory
    (named ``.<basename>.<suffix>.tmp``) but the target file itself
    is never corrupted.

    The parent directory of ``path`` is created if missing (``mkdir -p``
    semantics), so callers do not need to prepare it separately.

    Args:
        path: Destination file path. Parent directory is created if
            it does not exist.
        text: String contents to write. The string is encoded per
            ``encoding`` before being placed on disk.
        encoding: Text encoding. Defaults to UTF-8, which is what
            virtually all simulation input files use in practice.
        fsync: If True (default), call ``os.fsync`` on the tempfile
            before the rename, forcing the data to be flushed from
            the OS page cache to physical storage. This is the safe
            default for HPC state files. Set to False only if you
            are writing so frequently that the fsync cost dominates
            and you accept that a sudden power loss may lose the
            last few writes.

    Raises:
        OSError: If the tempfile cannot be created, written, or
            renamed. The tempfile is cleaned up on failure if
            possible.

    Example:
        Write a per-case options file that must be complete before
        the simulation reads it::

            from workflow_common import atomic_write_text
            atomic_write_text(
                "wf/gen_0/gene_3/options.toml",
                rendered_toml_text,
            )

    Notes:
        Implementation uses ``tempfile.mkstemp`` with ``dir=`` set to
        the target's parent directory. This guarantees the tempfile
        lives on the same filesystem as the target, which is required
        for the rename to be atomic (POSIX rename is only atomic
        within a single filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tempfile in the same directory ensures the subsequent rename
    # stays within one filesystem, which is what makes it atomic.
    # The dotfile prefix and .tmp suffix make any leftover tempfiles
    # easy to spot and clean up if a process dies mid-write.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        # os.fdopen wraps the low-level file descriptor returned by
        # mkstemp in a normal Python file object, so we can write text
        # to it with the requested encoding.
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()  # push Python's buffer into the OS
            if fsync:
                os.fsync(f.fileno())  # push the OS buffer onto disk

        # os.replace is the atomic rename on both POSIX and Windows.
        # Using this instead of os.rename handles the Windows case
        # where rename-onto-existing-file raises.
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup: if anything above failed, try to remove
        # the tempfile so we don't accumulate ".options.toml.XYZ.tmp"
        # garbage next to the real file. We swallow the cleanup error
        # so the caller sees the original exception, not a secondary
        # one from unlink.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_replace(src: PathLike, dst: PathLike) -> None:
    """Atomically move ``src`` onto ``dst``.

    Thin wrapper around ``os.replace`` whose purpose is to make the
    intent explicit at the call site: we are doing a crash-safe
    rename, not a copy-and-delete. Both paths must be on the same
    filesystem for atomicity.

    Args:
        src: Existing file to move.
        dst: Destination path. Overwritten if it already exists.

    Raises:
        OSError: If ``src`` does not exist or the rename cannot be
            performed (e.g. cross-filesystem rename attempt).

    Example:
        A two-step "write then publish" pattern for a manifest
        snapshot::

            atomic_write_text("snapshot.json.new", serialized)
            atomic_replace("snapshot.json.new", "snapshot.json")
    """
    os.replace(os.fspath(src), os.fspath(dst))


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist (``mkdir -p``).

    Exists so callers do not have to repeat the ``parents=True,
    exist_ok=True`` incantation and so the intent is obvious.

    Args:
        path: Directory path to create.

    Returns:
        The same path as a ``pathlib.Path`` for convenience chaining.

    Example:
        ::

            out_dir = ensure_dir("results/run_42") / "avg_stress.txt"
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
