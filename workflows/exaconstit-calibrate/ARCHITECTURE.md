# workflow_common — Architecture

Read this if you are new to the codebase and want a one-document
overview before diving into individual modules. The goal here is not
to replace the in-source docstrings (those are the source of truth)
but to show how the pieces fit together.

## What problem this package solves

A scientific workflow — a parameter sweep, an optimization, a UQ
campaign — ends up doing the same handful of things no matter what
simulation code sits underneath:

1. Render per-case input files from a master template.
2. Lay out working directories in a predictable way.
3. Launch simulations on some hardware (laptop, cluster, HPC).
4. Collect results, deal with failures, survive an allocation timeout.

Historically these concerns were tangled together inside a single
driver script. `workflow_common` separates them so that each one is
independently reusable and independently testable — and so the same
driver can swap in a different simulation code by supplying a
different template and a different path pattern, rather than being
rewritten.

## One-picture overview

```
                    +-------------------------+
                    |   Driver / user code    |
                    |  (optimizer, sweep, ...)|
                    +-------------------------+
                       |     |        |     |
                       v     v        v     v
           +------------+ +------+ +------+ +--------+
           |  templates | |paths | |manif.| |sentinel|
           +------------+ +------+ +------+ +--------+
                                      |
                                      |  submits SimJobSpec
                                      v
                          +------------------------+
                          |   JobBackend Protocol  |
                          +------------------------+
                              /                  \
                             /                    \
                 +------------------+     +------------------+
                 |   LocalBackend   |     |   FluxBackend    |
                 |  (subprocess +   |     |  (flux.job.      |
                 |   ThreadPool)    |     |   FluxExecutor)  |
                 +------------------+     +------------------+
                           |                       |
                           v                       v
                    local subprocesses        Flux-submitted jobs
```

The driver is the only code that knows the user's optimization
algorithm. Everything below the driver is problem-agnostic.

## Module responsibilities, one line each

| Module | Responsibility |
| --- | --- |
| `_fs` | Directory context manager, atomic text writes |
| `logging_utils` | Stdlib `logging` setup + shims for old call sites |
| `templates` | `%%key%%` substitution for input-file rendering |
| `paths` | Case → working-directory and logical → output-file resolution |
| `manifest` | JSONL event log for crash-safe case state tracking |
| `sentinel` | Per-case `.done` file as authoritative completion marker |
| `platform_detect` | Hostname-based detection of Spectrum-MPI machines |
| `backends.base` | `SimJobSpec` / `JobResult` / `JobBackend` Protocol |
| `backends.local` | Run sims as local subprocesses |
| `backends.flux_backend` | Run sims via `flux.job.FluxExecutor` |

## The two most important interfaces

Almost everything in the package hangs off two small abstractions.
If you internalize these, the rest is mechanical.

### `PathResolver`

```python
class PathResolver(Protocol):
    def working_dir(self, ctx: CaseContext) -> Path: ...
    def output_file(self, logical_name: str, ctx: CaseContext) -> Path: ...
```

Given a `CaseContext` (a (generation, gene, obj) triple plus extras),
a resolver answers:

- "Where does this case live on disk?"
- "Where will its `avg_stress.txt` appear after the sim runs?"

`TemplatePathResolver` is the supplied implementation: layouts are
described by Python format-string patterns. Users with exotic needs
can write their own class satisfying the Protocol.

### `JobBackend`

```python
class JobBackend(Protocol):
    def stream_batch(self, specs: Sequence[SimJobSpec]) -> Iterator[JobResult]: ...
    def submit_batch(self, specs: Sequence[SimJobSpec]) -> List[JobResult]: ...
    def submit_one(self, spec: SimJobSpec) -> JobResult: ...
```

Given a batch of `SimJobSpec`s, a backend runs them and produces
`JobResult`s. Callers do not need to know whether the backend uses
subprocesses, Flux, SLURM, or something not yet written. Concrete
backends inherit from `BaseBackend` and only need to implement
`stream_batch`; the other two methods fall out for free.

## How a case flows through the system

The case lifecycle is the clearest way to see how the modules
cooperate. For each case, in order:

1. **Path resolution.** Driver builds a `CaseContext` and asks the
   `PathResolver` for the working directory. It creates the directory
   and renders `options.toml` (or whatever) from the master template
   using `render_template_file`.
2. **Manifest: SUBMITTED.** Driver records a `ManifestEntry(state=
   SUBMITTED)` with the case key and working dir. If the driver is
   killed between this write and a terminal state, restart logic
   will later promote this entry to `INTERRUPTED`.
3. **Backend handoff.** Driver builds a `SimJobSpec` pointing at the
   simulation binary and hands it to the backend's `stream_batch`.
   The backend launches the job (subprocess or Flux) and, later,
   yields a `JobResult`.
4. **Output validation.** Driver calls `validate_outputs` to confirm
   the required files are present and non-empty. A clean `rc=0` with
   missing files is still a workflow-level failure.
5. **Sentinel write.** Driver calls `write_sentinel` with a `Sentinel`
   object describing what happened. The write is atomic
   (tempfile + rename). After this point, the case is considered
   done regardless of what happens next.
6. **Manifest: COMPLETED / FAILED.** Driver records the terminal
   `ManifestEntry`. This is the last step on purpose: a crash between
   steps 5 and 6 is safely recoverable (sentinel tells the truth),
   but a crash in the opposite order would leave the manifest
   claiming work was done that was never validated.

```
  CaseContext           build SimJobSpec        JobResult
       |                       |                    |
       | render template       | stream_batch       | validate outputs
       v                       v                    v
  options.toml           backend launches      +-----------+
                         flux / subprocess     | Sentinel  |  step 5
                                               | write     |  (FIRST)
                                               +-----------+
                                                     |
                                                     v
                                               +-----------+
                                               | Manifest  |  step 6
                                               | record    |  (SECOND)
                                               +-----------+
```

## Why two state files (manifest and sentinel)?

A common question. Both record case state; why not unify them?

They answer different questions:

- **The manifest** answers "what is the global state of this run?"
  It is append-only (JSONL), holds every transition for every case,
  and is the source of truth for restart planning.
- **The sentinel** answers "is the work in this specific directory
  finished?" It is a single file in the case directory, written
  atomically, and is the source of truth for per-case skip logic.

The sentinel is locally inspectable. An analyst looking at one
output directory knows at a glance whether it is safe to post-process
the files inside. They do not need to parse the manifest, they do
not need to know which run this directory came from — the `.done`
file tells them.

The manifest is globally inspectable. The driver needs to know, on
restart, which cases are done so it can skip them, and which cases
were submitted but never finished so it can retry them. A pile of
sentinels cannot tell you about cases that were planned but never
launched.

Keeping both means each layer is small, simple, and can be reasoned
about in isolation.

## Restart model

Optimization runs routinely outlive their HPC allocations. The
restart story is:

1. On startup, driver calls `Manifest.load()`. This replays the
   snapshot + JSONL log to rebuild the in-memory state.
2. Driver calls `Manifest.mark_submitted_as_interrupted()`. Anything
   stuck in SUBMITTED is assumed to have been killed alongside the
   old allocation.
3. For each case in the new plan, driver checks
   `is_case_complete(working_dir)`. A present sentinel means the
   case is done (regardless of what the manifest says) and can be
   skipped. No sentinel means the case either never ran, was
   interrupted, or failed without producing a sentinel, and should
   be submitted.

There is deliberately no Flux-state recovery. When the allocation
dies, the Flux instance dies with it, and its jobids are
unrecoverable. The restart machinery is entirely filesystem-driven.

## Non-goals

Things `workflow_common` deliberately does not do:

- **Parse simulation input files.** Master templates are text; we
  substitute `%%key%%` placeholders and nothing else. The framework
  never knows whether the file is TOML, XML, INI, or a shell script.
- **Know about physical quantities.** Strain rates, yield stresses,
  stress-strain curves — all of that is user code. The framework
  only carries opaque parameter dicts from the driver into the
  template renderer.
- **Optimization.** No GA, no nearest-neighbor sampling, no
  surrogate models. Those live in the driver. The framework's job
  is to make driving the simulation reproducible and restartable.
- **Auto-starting Flux.** You start a Flux instance yourself (via
  `flux start` inside an allocation); the backend connects to
  whatever is already running.

## Adding a new backend

If you need to support a new execution environment (e.g. a SLURM
step backend, or a remote SSH backend for tiny shared clusters):

1. Subclass `BaseBackend` (from `backends.base`).
2. Implement `stream_batch(specs) -> Iterator[JobResult]`. The
   `submit_batch` and `submit_one` methods come for free.
3. Use `SimJobSpec` fields for all inputs; do not introduce new
   fields unless absolutely necessary. If you do need new fields,
   put them inside `SimJobSpec.extra` so core callers remain
   backend-agnostic.
4. Return `JobResult` with a meaningful `JobOutcome`.

No Protocol inheritance declaration required — Python's structural
typing means any class with the right method signatures is accepted
wherever `JobBackend` is expected.

## Adding a new path layout

If `TemplatePathResolver`'s format-string approach does not fit your
needs (e.g. you want paths looked up from a database, or computed
from some hash of the parameter values):

1. Write a class with `working_dir(self, ctx)` and
   `output_file(self, logical_name, ctx)` methods.
2. Hand an instance to the driver where it expects a `PathResolver`.

That's it. Again, no inheritance declaration needed.

## Where to look next

- `tests/demo_workflow.py` — runnable end-to-end example, no HPC
  required. Read it top to bottom and you will see every module in
  use.
- Each module's header docstring — the "why" for that module.
- The function-level docstrings — the "what" and the "how".
