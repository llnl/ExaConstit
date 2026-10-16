"""
End-to-end smoke test / demo for workflow_common.

Purpose
-------
This script exists to serve three audiences:

1. **Developers testing changes to workflow_common**: the fastest way
   to verify the plumbing still works after a modification is to run
   this script on a desktop. It does not require flux, an HPC
   allocation, or ExaConstit itself.
2. **New users learning the framework**: the code here is a concrete
   minimal example of how the pieces (``TemplatePathResolver``,
   ``SimJobSpec``, ``LocalBackend``, ``Manifest``, sentinels) fit
   together. Read it alongside the module-level docstrings.
3. **CI**: the script's return code is 0 when everything worked, so
   a continuous-integration job can run it as a regression check.

What it does
------------
The script creates a small temporary workspace under ``/tmp/wfc_demo``
(or wherever ``--root`` points), writes a trivial "fake simulation"
Python script, and then runs a tiny 3-gene, 2-objective optimization
against that fake simulation. The fake sim reads its options.toml,
sleeps briefly, and writes an ``avg_stress.txt`` file with a made-up
stress-strain curve. No real physics, just enough plumbing to
exercise every part of the framework.

Modes
-----
The ``--phase`` flag selects which scenario is exercised:

* ``full``: happy path. All cases succeed.
* ``partial``: every case is poisoned to fail via the ``FAKE_FAIL``
  environment variable. Used to verify the failure path and the
  manifest's FAILED state transitions.
* ``restart``: rerun after a ``partial`` phase. The restart logic
  should see failed sentinels and either rerun or skip depending
  on policy. (Current demo just rechecks sentinels; adjust to taste
  when prototyping your own restart policy.)

Usage
-----
Quickest verification::

    python tests/demo_workflow.py --clean --phase full

A restart cycle::

    python tests/demo_workflow.py --clean --phase partial
    python tests/demo_workflow.py --phase restart

The ``--clean`` flag wipes the workspace before starting, so repeated
runs are deterministic. Omit it to inspect leftover state from a
previous run.

Reading guide
-------------
If you are reading this file to learn the framework, start here and
follow the call chain in order:

1. :func:`main` at the bottom - entry point. Shows how the pieces
   are wired together: logging, path resolver, manifest, backend.
2. :func:`run_generation` - the heart of a real driver. Demonstrates
   the sentinel-first-then-manifest ordering, the restart skip
   check, and output validation.
3. :func:`build_specs_and_contexts` - how per-case input files get
   rendered from a template and turned into executable specs.
4. The fake binary ``FAKE_BINARY_SRC`` - a stub "simulation" that
   reads its options and writes a plausible output file. In a real
   workflow this would be replaced by the actual mechanics code.

Everything else in this file is plumbing to make the three functions
above self-contained and runnable on any machine.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import textwrap
import time
from pathlib import Path

# Make workflow_common importable without a package install, by putting
# the repository root on sys.path at runtime. This is the "script
# living next to the package" idiom and is fine for internal tools.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from workflow_common import (  # noqa: E402
    CaseContext,
    CaseState,
    LocalBackend,
    Manifest,
    ManifestEntry,
    SimJobSpec,
    TemplatePathResolver,
    configure_logging,
    get_logger,
    render_template_file,
    write_sentinel,
)
from workflow_common.backends.base import JobOutcome  # noqa: E402
from workflow_common.sentinel import (  # noqa: E402
    is_case_complete,
    read_sentinel,
    Sentinel,
    validate_outputs,
)

logger = get_logger("demo")


# --- Fixtures -------------------------------------------------------------
#
# We generate both the fake simulation binary and the master template on
# the fly so the demo is fully self-contained - the repo does not need to
# carry these as extra data files.

# Source code of the fake simulation. Written out to disk, chmod +x'd,
# and invoked like a real binary. Reads options.toml for a strain_rate,
# sleeps briefly to simulate work, and writes a dummy avg_stress.txt.
# Honors FAKE_FAIL=1 to force a nonzero rc, which is how the --phase
# partial mode exercises the failure path.
FAKE_BINARY_SRC = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    '''Fake simulation. Reads options.toml (we just grep for a value),
    writes an avg_stress.txt, optionally fails depending on env vars.'''
    import os, sys, time, pathlib, re

    opt_path = pathlib.Path("options.toml")
    if not opt_path.exists():
        print("no options.toml", file=sys.stderr)
        sys.exit(2)

    text = opt_path.read_text()
    m = re.search(r'strain_rate\\s*=\\s*([-\\d.eE+]+)', text)
    strain_rate = float(m.group(1)) if m else 1e-3

    # Simulate a bit of work so parallel execution actually looks parallel.
    time.sleep(float(os.environ.get('FAKE_SLEEP', '0.05')))

    # Optional forced failure, controlled by the demo driver.
    if os.environ.get('FAKE_FAIL') == '1':
        print('simulated crash', file=sys.stderr)
        sys.exit(7)

    out_dir = pathlib.Path('results/options')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(10):
        t = 0.1 * i
        s = 100.0 * (1 - 2 ** (-strain_rate * t * 10))
        rows.append(f'{s:.6f} {strain_rate*t:.6f}')
    (out_dir / 'avg_stress.txt').write_text('\\n'.join(rows) + '\\n')
    print('fake sim ok')
    """
)


# Master-template input "file". Uses the %%key%% placeholder syntax so
# the template renderer substitutes per-case values in.
MASTER_TEMPLATE = textwrap.dedent(
    """\
    [Problem]
        name = "case_%%gene%%_%%obj%%"
        strain_rate = %%strain_rate%%
        temperature_k = %%temp_k%%
    """
)


def prepare_fake_binary(root: Path) -> Path:
    """Write the fake simulation script to disk and make it executable.

    Args:
        root: Directory under which to write the script. Written as
            ``<root>/fake_sim.py``.

    Returns:
        Path to the written script, ready to use as a
        :attr:`SimJobSpec.binary`.
    """
    bin_path = root / "fake_sim.py"
    bin_path.write_text(FAKE_BINARY_SRC)
    # chmod +x so the kernel can exec it via shebang. We preserve any
    # existing mode bits and OR in the execute bits for all users; the
    # script lives under /tmp so permissive perms are fine.
    bin_path.chmod(
        bin_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    return bin_path


# --- Core per-generation flow --------------------------------------------


def build_specs_and_contexts(
    genes,
    generation: int,
    resolver: TemplatePathResolver,
    binary: Path,
    master_template_path: Path,
):
    """Render per-case inputs and build :class:`SimJobSpec` instances.

    For each gene and each objective, this function:

    1. Builds a :class:`CaseContext` tagged with per-case parameters.
    2. Resolves the working directory via the supplied
       :class:`TemplatePathResolver`.
    3. Renders the master template into an ``options.toml`` inside
       that working directory.
    4. Constructs a :class:`SimJobSpec` pointing at the fake binary.

    The function returns pairs ``(ctx, spec)`` rather than just specs
    because downstream code needs both: the context to drive the
    manifest transitions, and the spec to hand to the backend.

    Args:
        genes: A list of genes, where each gene is itself a list of
            per-objective parameter dicts. For example,
            ``genes[1][0]`` is the parameter dict for gene 1,
            objective 0.
        generation: The generation number for this batch.
        resolver: Provides per-case working directories.
        binary: Path to the simulation binary.
        master_template_path: Template to render per case.

    Returns:
        A list of ``(CaseContext, SimJobSpec)`` pairs, one per
        objective of each gene.
    """
    out = []
    for igene, gene in enumerate(genes):
        for iobj, params in enumerate(gene):
            ctx = CaseContext(
                generation=generation,
                gene=igene,
                obj=iobj,
                extra=dict(params),
            )
            wd = resolver.working_dir(ctx)
            wd.mkdir(parents=True, exist_ok=True)

            # Render options.toml with per-case parameter values. The
            # values dict picks up the same per-case parameters used
            # in the CaseContext.extra, plus the indices themselves.
            values = dict(gene=igene, obj=iobj, **params)
            render_template_file(
                master_template_path,
                wd / "options.toml",
                values,
            )

            spec = SimJobSpec(
                working_dir=wd,
                binary=binary,
                args=("-opt", "options.toml"),
                num_nodes=1,
                num_tasks=1,
                duration_s=60,
                stdout="stdout.log",
                stderr="stderr.log",
                tag=f"gen{generation}_g{igene}_o{iobj}",
            )
            out.append((ctx, spec))
    return out


def run_generation(
    *,
    generation: int,
    genes,
    resolver: TemplatePathResolver,
    binary: Path,
    master_template_path: Path,
    manifest: Manifest,
    backend: LocalBackend,
    required_outputs,
):
    """Run one generation end-to-end with manifest and sentinel plumbing.

    This is the function that shows how a real driver should wire up
    the machinery. Steps, in order:

    1. Build all case contexts and specs for this generation.
    2. For each case: check if a sentinel already says it is complete.
       If so, skip (and reflect that in the manifest).
    3. For each non-skipped case: record a SUBMITTED manifest entry,
       then hand the spec to the backend.
    4. As results stream back, validate the outputs, write a sentinel
       (sentinel first!), then record the terminal manifest entry.

    The sentinel-before-manifest ordering is deliberate: if a crash
    happens between the two writes, the sentinel is already on disk,
    so restart correctly treats the case as complete. The manifest
    catches up on the next successful record.

    Args:
        generation: Current generation number.
        genes: Per-gene, per-objective parameter dicts.
        resolver: Path resolver for this run.
        binary: Simulation binary.
        master_template_path: Input-file template.
        manifest: The run's manifest (already loaded).
        backend: The job backend to use.
        required_outputs: List of output-file paths (relative to each
            case's working directory) that must exist and be nonempty
            for the case to be considered successful.
    """
    all_items = build_specs_and_contexts(
        genes, generation, resolver, binary, master_template_path
    )

    to_run = []
    skipped = 0
    for ctx, spec in all_items:
        # Restart shortcut: a sentinel means this case was completed
        # (successfully or not) on a previous run. We do not rerun it,
        # but we do reflect the outcome in the manifest so the
        # in-memory state matches disk.
        if is_case_complete(spec.working_dir):
            s = read_sentinel(spec.working_dir)
            logger.info(
                "skipping gen=%d gene=%d obj=%d (sentinel present, rc=%d)",
                ctx.generation, ctx.gene, ctx.obj, s.rc if s else -1,
            )
            manifest.record(
                ManifestEntry(
                    generation=ctx.generation,
                    gene=ctx.gene,
                    obj=ctx.obj,
                    state=(
                        CaseState.COMPLETED
                        if (s and s.rc == 0)
                        else CaseState.FAILED
                    ),
                    rc=s.rc if s else None,
                    case_dir=str(spec.working_dir),
                    message="restored from sentinel",
                )
            )
            skipped += 1
            continue

        # About to hand the case to the backend - record SUBMITTED
        # before the handoff so restart has a record of "I was trying
        # to run this when I died" even if the process is killed
        # mid-submission.
        manifest.record(
            ManifestEntry(
                generation=ctx.generation,
                gene=ctx.gene,
                obj=ctx.obj,
                state=CaseState.SUBMITTED,
                case_dir=str(spec.working_dir),
            )
        )
        to_run.append((ctx, spec))

    logger.info(
        "gen %d: %d cases to run, %d restored from sentinel",
        generation, len(to_run), skipped,
    )

    if not to_run:
        return

    # We need to map results back to their contexts when they stream
    # in from the backend. Using id(spec) is safe here because the
    # spec objects persist for the entire loop and are never copied.
    spec_to_ctx = {id(spec): ctx for ctx, spec in to_run}
    specs = [spec for _, spec in to_run]

    for result in backend.stream_batch(specs):
        ctx = spec_to_ctx[id(result.spec)]

        # Validate outputs regardless of rc. A zero rc with missing
        # outputs is still a workflow-level failure - for example, a
        # process that exited cleanly before writing its results.
        ok, missing = validate_outputs(result.spec.working_dir, required_outputs)
        if result.outcome == JobOutcome.OK and not ok:
            logger.warning(
                "gen=%d gene=%d obj=%d rc=0 but outputs missing: %s",
                ctx.generation, ctx.gene, ctx.obj, missing,
            )

        terminal = (
            CaseState.COMPLETED
            if (result.outcome == JobOutcome.OK and ok)
            else CaseState.FAILED
        )

        # Sentinel first. If we crash between this write and the
        # manifest record below, restart will see the sentinel and
        # correctly treat the case as finished.
        write_sentinel(
            result.spec.working_dir,
            Sentinel(
                rc=result.rc,
                wall_time_s=result.wall_time_s,
                jobid=result.jobid,
                output_files={p: str(p) for p in required_outputs},
                status="ok" if terminal == CaseState.COMPLETED else "bad",
                message=result.error_message,
            ),
        )
        manifest.record(
            ManifestEntry(
                generation=ctx.generation,
                gene=ctx.gene,
                obj=ctx.obj,
                state=terminal,
                rc=result.rc,
                jobid=result.jobid,
                case_dir=str(result.spec.working_dir),
                message=result.error_message,
            )
        )
        logger.info(
            "gen=%d gene=%d obj=%d -> %s rc=%d wall=%.2fs",
            ctx.generation, ctx.gene, ctx.obj,
            terminal.value, result.rc, result.wall_time_s,
        )


def main() -> int:
    """Script entry point.

    Parses CLI arguments, prepares the workspace, runs one
    generation, and exits with status 0 if every case completed
    successfully or 1 if any case failed. Returns, rather than
    directly calling ``sys.exit``, so tests can import and invoke
    ``main()`` programmatically if desired.

    Returns:
        0 on success, 1 if any case failed.
    """
    p = argparse.ArgumentParser(description="workflow_common demo")
    p.add_argument("--root", type=Path, default=Path("/tmp/wfc_demo"))
    p.add_argument("--clean", action="store_true",
                   help="wipe the workspace before starting")
    p.add_argument("--workers", type=int, default=2,
                   help="number of concurrent simulations")
    p.add_argument(
        "--phase",
        choices=("full", "partial", "restart"),
        default="full",
        help="'full' runs all cases normally; 'partial' poisons cases "
             "to fail; 'restart' reruns and uses existing sentinels.",
    )
    args = p.parse_args()

    if args.clean and args.root.exists():
        shutil.rmtree(args.root)
    args.root.mkdir(parents=True, exist_ok=True)

    # stream=True (default) is helpful for demo output so you can see
    # the progression live. On a real HPC run you would typically
    # disable stream and write logs to a file.
    configure_logging(level="info")

    # Layout setup: fake binary, master template, path resolver.
    binary = prepare_fake_binary(args.root)
    master_template_path = args.root / "master.toml"
    master_template_path.write_text(MASTER_TEMPLATE)

    resolver = TemplatePathResolver(
        working_dir_pattern="wf_files/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
        },
        root=args.root,
    )

    # Load the manifest. On a fresh run this is a no-op; on restart
    # it replays history and marks anything stuck in SUBMITTED as
    # INTERRUPTED. The count is logged so the operator can see what
    # the restart detected.
    manifest = Manifest(args.root / "wf_files" / "manifest.jsonl")
    manifest.load()
    n_interrupted = manifest.mark_submitted_as_interrupted()
    if n_interrupted:
        logger.info("marked %d submitted-but-unknown cases as INTERRUPTED",
                    n_interrupted)

    backend = LocalBackend(max_workers=args.workers)

    # Tiny toy optimization: 3 genes, each with 2 objectives at
    # different strain rates. Parameters are passed through the
    # CaseContext.extra mapping and into the template.
    genes = [
        [dict(strain_rate=1e-3, temp_k=298.0), dict(strain_rate=1e-1, temp_k=298.0)],
        [dict(strain_rate=5e-4, temp_k=298.0), dict(strain_rate=5e-2, temp_k=298.0)],
        [dict(strain_rate=2e-3, temp_k=298.0), dict(strain_rate=2e-1, temp_k=298.0)],
    ]

    # 'partial' mode poisons the subprocess environment so every case
    # fails. Use this to exercise the FAILED branch of the driver.
    # The env var goes on os.environ so the child processes inherit it.
    if args.phase == "partial":
        os.environ["FAKE_FAIL"] = "1"
    else:
        os.environ.pop("FAKE_FAIL", None)

    required_outputs = ["results/options/avg_stress.txt"]

    run_generation(
        generation=0,
        genes=genes,
        resolver=resolver,
        binary=binary,
        master_template_path=master_template_path,
        manifest=manifest,
        backend=backend,
        required_outputs=required_outputs,
    )

    # Snapshot at end of generation so a later restart does not have
    # to replay the entire JSONL log from the beginning.
    manifest.snapshot()

    # Final tally, printed for human consumption.
    n_ok = sum(1 for e in manifest.all_entries() if e.state == CaseState.COMPLETED)
    n_bad = sum(1 for e in manifest.all_entries() if e.state == CaseState.FAILED)
    logger.info("done. completed=%d failed=%d", n_ok, n_bad)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
