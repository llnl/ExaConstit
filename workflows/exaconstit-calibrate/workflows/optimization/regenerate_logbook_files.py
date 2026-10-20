"""Regenerate the DEAP-style ``logbook1_stats.log`` and
``logbook2_solutions.log`` text files from a checkpoint pickle.

Useful when those on-disk logs have been lost — truncated by a
misconfigured resume run, accidentally deleted, whatever — but the
checkpoint pickle for the last good generation is still present.
The pickle carries the full DEAP ``Logbook`` history for both
stats and solutions, so the ``.log`` files can be faithfully
reconstructed.

Invocation:

    python -m workflows.optimization.regenerate_logbook_files \\
        ./calibration_run/checkpoint_files/checkpoint_gen_15.pkl

By default the regenerated files land next to the pickle (in the
same directory, with the canonical names ``logbook1_stats.log``
and ``logbook2_solutions.log``). Use ``--output-dir`` to put them
somewhere else:

    python -m workflows.optimization.regenerate_logbook_files \\
        ./calibration_run/checkpoint_files/checkpoint_gen_15.pkl \\
        --output-dir /tmp/recovered_logs

Safe to run repeatedly. Existing ``.log`` files in the output dir
are OVERWRITTEN (that's the whole point), so make a backup first
if you care about whatever's there now.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional

# Reuse the exact writer the driver uses — same formatter, same
# delta mechanics, byte-for-byte compatible with what a live run
# would have produced.
from workflows.optimization.nsga3_driver import _LogbookWriter


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Regenerate logbook1_stats.log and logbook2_solutions.log "
            "from a checkpoint pickle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Regenerate in place (next to the pickle):\n"
            "    python -m workflows.optimization.regenerate_logbook_files \\\n"
            "        ./calibration_run/checkpoint_files/checkpoint_gen_15.pkl\n"
            "\n"
            "  # Write to a different directory:\n"
            "    python -m workflows.optimization.regenerate_logbook_files \\\n"
            "        ./calibration_run/checkpoint_files/checkpoint_gen_15.pkl \\\n"
            "        --output-dir /tmp/recovered\n"
        ),
    )
    p.add_argument(
        "checkpoint",
        type=Path,
        help=(
            "Path to a checkpoint pickle written by run_nsga3, e.g. "
            "``checkpoint_gen_15.pkl``."
        ),
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help=(
            "Directory to write the .log files into. Default: the "
            "directory containing the checkpoint pickle. Created "
            "if it doesn't exist."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.checkpoint.is_file():
        print(
            f"error: checkpoint not found at {args.checkpoint}",
            file=sys.stderr,
        )
        return 1

    # Load the pickle. ``_load_checkpoint`` in the driver does the
    # same thing with a security warning docstring; we do it inline
    # here because we don't need the rest of the driver's state.
    with args.checkpoint.open("rb") as f:
        ckp = pickle.load(f)

    # Sanity-check the pickle shape. Older or custom checkpoint
    # formats might not have both logbooks; we surface a clear
    # error rather than crashing deep inside the writer.
    for key in ("logbook1", "logbook2"):
        if key not in ckp:
            print(
                f"error: checkpoint is missing the {key!r} entry — "
                f"is this a run_nsga3 checkpoint?",
                file=sys.stderr,
            )
            return 1

    logbook1 = ckp["logbook1"]
    logbook2 = ckp["logbook2"]

    # Pick the output directory. Default is "next to the pickle",
    # which matches where a live run would have put them (the
    # driver defaults log_dir to checkpoint_dir).
    out_dir = args.output_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Report what we're about to do before we truncate anything —
    # users should see the action clearly, especially if the
    # output dir already has real data in it.
    stats_path = out_dir / "logbook1_stats.log"
    solutions_path = out_dir / "logbook2_solutions.log"
    print(
        f"regenerating from checkpoint: {args.checkpoint}\n"
        f"  logbook1 records: {len(logbook1)}\n"
        f"  logbook2 records: {len(logbook2)}\n"
        f"  writing: {stats_path}\n"
        f"  writing: {solutions_path}",
        file=sys.stderr,
    )

    writer = _LogbookWriter(out_dir)
    writer.rewrite_from(logbook1, logbook2)

    print("done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
