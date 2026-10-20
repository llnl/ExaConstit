"""
Reading simulation outputs back into structured, analysis-ready data.

Why this module exists
----------------------
After a case finishes and its sentinel file is written, the next thing
the driver wants to do is read what the simulation produced and
compute an objective value from it. In the legacy ``ExaProb`` class,
that logic sat inline: the same method that organized directories
also parsed ``avg_stress.txt``, also loaded experimental data, also
computed error metrics. The result was a monolith that was hard to
test in isolation and impossible to reuse on a different simulation
code.

This module separates the "read files back into arrays" step from
everything else. Two abstractions do the work:

* :class:`CaseLayout` — a read-oriented view onto a single case on
  disk. Given a :class:`CaseContext` and a :class:`PathResolver`, it
  answers "where is this output?", "does it exist?", "delete all
  outputs for this case so I can re-run it".

* :class:`ResultReader` (Protocol) — turns one or more files inside
  a case layout into structured data. This module ships a concrete
  :class:`TextTableReader` for the common case of whitespace- or
  comma-delimited columnar files, plus a small
  :func:`load_experimental_csv` helper for the experimental-side
  data that these results get compared against. Users who have
  unusual formats (HDF5, VTK, binary dumps) write their own class
  satisfying the Protocol.

Design principle: readers stop at "read". Smoothing, resampling,
nondimensionalization, error-metric computation — all of those are
the next layer up. Keeping readers focused makes them easier to
reuse across codes.

What a typical ExaConstit output layout looks like
--------------------------------------------------
For a case whose working directory is ``wf/gen_0/gene_3_obj_1`` and
whose basename option is ``options``, a successful run leaves::

    wf/gen_0/gene_3_obj_1/
        options.toml                           # rendered input
        stdout.log                             # captured stdout
        stderr.log                             # captured stderr
        .done                                  # sentinel (workflow_common)
        results/
            options/
                avg_stress.txt                 # 6 cols: s11..s13
                avg_def_grad.txt               # 9 cols of F components
                avg_plastic_work.txt           # scalar per step
                visualizations/                # ParaView output
                    ...

``CaseLayout`` and ``TextTableReader`` together turn that directory
tree into a dict of pandas DataFrames, one per logical output name.

Dependencies
------------
This module uses numpy and pandas. Both are already hard dependencies
of ExaConstit-adjacent workflows, so we do not guard the imports. If
you are reading this on a stdlib-only machine, you will need to
install them or write your own ResultReader implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .paths import CaseContext, PathResolver

logger = get_logger(__name__)


# --- CaseLayout -----------------------------------------------------------


@dataclass
class CaseLayout:
    """A read-oriented view onto one case's directory on disk.

    A :class:`CaseLayout` bundles a :class:`CaseContext` (which
    identifies the case) with a :class:`PathResolver` (which knows
    the layout rules). The result is a lightweight object whose
    methods are all scoped to this one case, so driver code does not
    have to pass ``(ctx, resolver)`` pairs everywhere.

    Two ways to think about the relationship:

    * ``PathResolver`` is the *map*: it knows how the directory
      layout works in general.
    * ``CaseLayout`` is the *pin on the map*: it knows where one
      specific case is and what is inside it.

    Fields:
        ctx: The :class:`CaseContext` identifying which case this is.
            Immutable once the layout is constructed - if you need
            to switch cases, make a new layout.
        resolver: The :class:`PathResolver` used to translate logical
            output names to physical paths. Typically shared across
            every layout in a run.

    Example:
        Construct a layout and inspect it::

            from workflow_common import (
                CaseContext, TemplatePathResolver, CaseLayout,
            )
            resolver = TemplatePathResolver(
                working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
                output_file_patterns={
                    "avg_stress":
                        "{working_dir}/results/options/avg_stress.txt",
                },
            )
            layout = CaseLayout(
                ctx=CaseContext(generation=0, gene=3, obj=1),
                resolver=resolver,
            )
            print(layout.working_dir)           # .../gene_3_obj_1
            print(layout.output_file("avg_stress"))
            print(layout.exists("avg_stress"))  # True if the file is there
    """

    ctx: CaseContext
    resolver: PathResolver

    # ---- path queries ----

    @property
    def working_dir(self) -> Path:
        """The case's working directory.

        Computed lazily by delegating to the resolver. The directory
        is NOT created by this accessor - only resolved. Callers
        that need the directory to exist on disk should call
        ``.mkdir(parents=True, exist_ok=True)`` themselves.

        Returns:
            The case's working directory as a :class:`pathlib.Path`.
        """
        return self.resolver.working_dir(self.ctx)

    def output_file(self, logical_name: str) -> Path:
        """Return the resolved path of a named output file.

        Args:
            logical_name: The framework-internal label for the
                output (e.g. ``"avg_stress"``). Must be a name the
                resolver knows about.

        Returns:
            The output's resolved path.

        Raises:
            KeyError: If ``logical_name`` is not configured on the
                resolver.
        """
        return self.resolver.output_file(logical_name, self.ctx)

    def known_outputs(self) -> List[str]:
        """Return the list of logical output names the resolver knows.

        Useful for introspection, logging, and generating validators
        that check all known outputs at once.

        Returns:
            Logical names in registration order. Empty list if the
            resolver does not support introspection.
        """
        # Not every PathResolver implementation is required to expose
        # this. TemplatePathResolver has a known_outputs() method; a
        # user-written resolver may not. Fall back gracefully.
        getter = getattr(self.resolver, "known_outputs", None)
        return list(getter()) if callable(getter) else []

    # ---- existence and cleanup ----

    def exists(self, logical_name: str) -> bool:
        """Return True if the named output exists on disk.

        The check is a simple ``Path.exists()``; it does not verify
        the file is nonempty or well-formed. For stronger checks
        see :func:`workflow_common.sentinel.validate_outputs`.

        Args:
            logical_name: Logical output name.

        Returns:
            True if the file exists, False otherwise.

        Raises:
            KeyError: If ``logical_name`` is not a configured output.
        """
        return self.output_file(logical_name).exists()

    def present_outputs(self) -> List[str]:
        """Return the subset of known outputs that currently exist on disk.

        Handy for logging what a partially-completed case produced
        before it died. Depends on :meth:`known_outputs` working,
        which in turn depends on the resolver supporting
        introspection.

        Returns:
            Logical names, in the same order as :meth:`known_outputs`.
        """
        return [n for n in self.known_outputs() if self.exists(n)]

    def clear_outputs(
        self, logical_names: Optional[Iterable[str]] = None
    ) -> List[Path]:
        """Delete output files for this case.

        Use before re-running a case when you want to ensure no stale
        data contaminates the new run. Files that are not present are
        silently ignored, so this is safe to call on a fresh case too.

        Does NOT remove the working directory itself or the rendered
        input files (``options.toml`` and the like); only the
        simulation's output artifacts. If you need a fully clean
        directory, ``shutil.rmtree`` it instead.

        Args:
            logical_names: Names to clear. ``None`` (default) means
                clear every known output on the resolver. Unknown
                names raise ``KeyError``.

        Returns:
            List of paths that were actually deleted (files that
            existed at call time). Missing files are not included.

        Example:
            ::

                # Before retrying a failed case
                layout.clear_outputs()
                run_again(layout)
        """
        names = list(logical_names) if logical_names is not None else self.known_outputs()
        wd = self.working_dir.resolve()
        removed: List[Path] = []
        for name in names:
            p = self.output_file(name)
            # Safety gate: refuse to delete any file that isn't
            # inside the case's working directory. A
            # misconfigured resolver (absolute output_file_pattern,
            # or a pattern that escapes via ``..``) could otherwise
            # have us unlink arbitrary files on disk when
            # ``clear_outputs_on_rerun=True`` fires during a
            # restart. Resolve both sides first so symlink trickery
            # can't defeat the check.
            try:
                p_resolved = p.resolve(strict=False)
            except OSError:
                # Typical cause: a path component is a broken
                # symlink. Treat as "not safe to touch" rather than
                # pursuing the unlink; the next Ok-path case will
                # have run its own renders and nothing is lost.
                logger.warning(
                    "clear_outputs: could not resolve %s; skipping", p,
                )
                continue
            try:
                p_resolved.relative_to(wd)
            except ValueError:
                # Path is outside the case working dir. Log loudly
                # and refuse — this was the bug class where an
                # absolute entry in output_file_patterns caused
                # ``unlink`` on a user file elsewhere on disk.
                logger.warning(
                    "clear_outputs: refusing to delete %s — "
                    "it is outside the case working dir %s. "
                    "This usually indicates a misconfigured "
                    "output_file_pattern (absolute path, or "
                    "one that escapes via '..').",
                    p_resolved, wd,
                )
                continue
            try:
                p.unlink()
                removed.append(p)
            except FileNotFoundError:
                # Expected: the file did not exist. Not an error.
                pass
            except OSError as e:
                # Something weirder - permission denied, file is a
                # directory, etc. Log but keep going so one bad entry
                # does not prevent cleanup of the rest.
                logger.warning("could not delete %s: %s", p, e)
        return removed


# --- Reader Protocol and Result containers --------------------------------


@dataclass
class TabularResult:
    """A single text table read from disk, as a pandas DataFrame plus metadata.

    Fields:
        name: Logical output name (e.g. ``"avg_stress"``).
        df: The parsed contents as a pandas DataFrame. Columns are
            named per the reader's configuration.
        source_path: Path the data was read from. Preserved for
            logging and for error messages when downstream code
            encounters a malformed value.

    Example:
        A ``TabularResult`` for ``avg_stress`` might look like::

            TabularResult(
                name="avg_stress",
                df=DataFrame with columns [time, s11, s22, s33, s12, s23, s13],
                source_path=Path("wf/gen_0/gene_3/.../avg_stress.txt"),
            )
    """

    name: str
    df: pd.DataFrame
    source_path: Path


@dataclass
class CaseResultSet:
    """All result files successfully read for one case.

    A :class:`ResultReader`'s output. Bundles the read tables and
    records which requested outputs were missing so the caller can
    distinguish "file was absent" from "file was empty" from "read
    succeeded".

    Fields:
        ctx: The case this result set describes. Carried along so
            downstream code can correlate back to the optimizer's
            plan without extra bookkeeping.
        tables: Mapping from logical name to :class:`TabularResult`
            for every output that was successfully read.
        missing: Logical names that were requested but not found on
            disk (and not marked required). Empty for a fully
            successful read.

    Convenience:
        A result set is truthy iff every requested output was read.
        This lets callers write::

            rs = reader.read(layout)
            if rs:
                compute_error(rs)
            else:
                log.warning("missing outputs: %s", rs.missing)
    """

    ctx: CaseContext
    tables: Dict[str, TabularResult] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return not self.missing

    def __getitem__(self, name: str) -> TabularResult:
        """Return the ``TabularResult`` for ``name``; raises KeyError if absent."""
        return self.tables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tables

    def df(self, name: str) -> pd.DataFrame:
        """Shorthand for ``self[name].df``.

        The DataFrame is the thing most downstream code actually
        wants; this accessor saves a ``.df`` chase at every call
        site.

        Args:
            name: Logical output name.

        Returns:
            The corresponding pandas DataFrame.

        Raises:
            KeyError: If ``name`` was not read.
        """
        return self.tables[name].df


@runtime_checkable
class ResultReader(Protocol):
    """Protocol: anything that can turn a :class:`CaseLayout` into results.

    A concrete reader implementation needs only one method. Users
    who need to read binary dumps, HDF5 files, VTK collections, or
    any format :class:`TextTableReader` does not handle can
    implement their own class satisfying this Protocol and the
    rest of the framework accepts it transparently.

    Methods:
        read(layout):
            Read whatever outputs this reader cares about from
            ``layout``. Return a :class:`CaseResultSet`.
    """

    def read(self, layout: CaseLayout) -> CaseResultSet: ...


# --- TextTableReader: the workhorse --------------------------------------


@dataclass
class TextTableSpec:
    """Configuration for reading one whitespace/delimited text table.

    Describes how :class:`TextTableReader` should parse one output
    file. One spec per file.

    Fields:
        columns: Column names to assign to the parsed DataFrame.
            The number of names must match the number of columns
            actually present in the file, unless ``usecols`` is set.
        delimiter: Column separator. ``None`` (default) splits on
            any whitespace, which matches ExaConstit's
            space-delimited text outputs.
        skip_rows: Number of lines to skip at the top of the file,
            e.g. for a banner or a non-header comment block. The
            ``comment`` character handles per-line ``#`` comments
            separately from this.
        comment: Character that marks the start of a comment. Lines
            (or line suffixes, for pandas) that begin with this
            character are skipped. Defaults to ``"#"``.
        required: If True, the reader raises when this file is
            missing. If False (default), the name is recorded in
            ``CaseResultSet.missing`` and reading continues.
        usecols: Optional subset of columns (0-indexed) to keep. If
            given, ``columns`` should have one entry per kept column.
            Handy when a simulation writes more columns than we care
            about.
        dtypes: Optional mapping from column name to dtype. If
            omitted, pandas infers per-column dtype from the data.

    Example:
        Spec for an ExaConstit-style ``avg_stress.txt`` with 8
        columns (Time + Volume + 6 Cauchy-stress components), all
        floats. ExaConstit's volume-averaged output files share a
        canonical ``# Time  Volume  ...`` header prefix; see
        ``ExaConstit/src/postprocessing/postprocessing_file_manager.hpp``
        ``GetVolumeAverageHeader`` for the authoritative per-calc-type
        column lists::

            TextTableSpec(
                columns=[
                    "Time", "Volume",
                    "Sxx", "Syy", "Szz",
                    "Sxy", "Sxz", "Syz",
                ],
                required=True,
            )
    """

    columns: Sequence[str]
    delimiter: Optional[str] = None
    skip_rows: int = 0
    comment: str = "#"
    required: bool = False
    usecols: Optional[Sequence[int]] = None
    dtypes: Optional[Mapping[str, object]] = None


class TextTableReader:
    """A :class:`ResultReader` for whitespace- or delimiter-separated text files.

    Handles the common case: a simulation writes one or more plain
    text output files, each with a fixed number of numeric columns.
    Rows are time steps; columns are quantities. Each output is
    parsed into a pandas DataFrame using the schema in the matching
    :class:`TextTableSpec`.

    The reader's configuration is a mapping from logical name to
    :class:`TextTableSpec`. At read time, the logical name is looked
    up on the :class:`CaseLayout` to get the actual file path, the
    file is parsed, and the resulting DataFrame is stored in the
    :class:`CaseResultSet` under the same name.

    Args:
        specs: Mapping of logical output name to :class:`TextTableSpec`.
            The names must match names that the :class:`PathResolver`
            on the layout knows about; otherwise you will get a
            KeyError when reading.

    Example:
        Configure for ExaConstit's default output set. All
        volume-averaged files share the ``Time`` + ``Volume`` prefix;
        the per-calc-type columns that follow come from
        ``ExaConstit/src/postprocessing/postprocessing_file_manager.hpp``
        :: ``GetVolumeAverageHeader``::

            reader = TextTableReader({
                "avg_stress": TextTableSpec(
                    columns=[
                        "Time", "Volume",
                        "Sxx", "Syy", "Szz",
                        "Sxy", "Sxz", "Syz",
                    ],
                    required=True,
                ),
                "avg_def_grad": TextTableSpec(
                    columns=[
                        "Time", "Volume",
                        "F11", "F12", "F13",
                        "F21", "F22", "F23",
                        "F31", "F32", "F33",
                    ],
                    required=True,
                ),
                "avg_plastic_work": TextTableSpec(
                    columns=["Time", "Volume", "Plastic_Work"],
                    required=False,
                ),
            })

            rs = reader.read(layout)
            stress_df = rs.df("avg_stress")
            # stress_df["Szz"] is a pandas Series of the z-axis
            # Cauchy stress over time (note capital S, x/y/z indexing
            # — ExaConstit uses Sxx/Syy/Szz/Sxy/Sxz/Syz, not
            # s11/s22/s33/...).
    """

    def __init__(self, specs: Mapping[str, TextTableSpec]):
        # Defensive copy so later mutations on the caller's dict do
        # not change what the reader reads. We store as a plain dict
        # for predictable iteration order in error messages.
        self._specs: Dict[str, TextTableSpec] = dict(specs)

    @property
    def specs(self) -> Mapping[str, TextTableSpec]:
        """Read-only view of the configured specs. Useful for debugging."""
        return dict(self._specs)

    def read(self, layout: CaseLayout) -> CaseResultSet:
        """Read every configured output present on the layout's disk.

        Missing required outputs cause a :class:`FileNotFoundError`;
        missing optional outputs are recorded in
        :attr:`CaseResultSet.missing` and the read continues. This
        lets workflows treat "simulation did not produce avg_plastic_work
        for this load case" differently from "simulation did not
        produce avg_stress at all".

        Args:
            layout: The :class:`CaseLayout` to read from.

        Returns:
            A :class:`CaseResultSet` populated with one
            :class:`TabularResult` per successfully-read output.

        Raises:
            FileNotFoundError: If a ``required=True`` output is
                absent or unreadable.
            ValueError: If a file's column count does not match its
                spec, or if ``pandas`` fails to parse a numeric
                column.
        """
        out = CaseResultSet(ctx=layout.ctx)

        for name, spec in self._specs.items():
            path = layout.output_file(name)

            if not path.exists():
                # Missing file. Required files are hard errors;
                # optional files are recorded and we move on.
                if spec.required:
                    raise FileNotFoundError(
                        f"required output {name!r} not found at {path}"
                    )
                out.missing.append(name)
                logger.debug(
                    "TextTableReader: %s missing at %s (optional)", name, path
                )
                continue

            df = self._read_one(path, spec)
            out.tables[name] = TabularResult(
                name=name, df=df, source_path=path
            )

        return out

    # ---- internals ----

    def _read_one(self, path: Path, spec: TextTableSpec) -> pd.DataFrame:
        """Parse a single file according to ``spec``.

        Private helper; use :meth:`read` for normal access.

        The parsing strategy is deliberately strict:

        * Column count is verified against ``spec.columns``. A
          mismatch raises, rather than silently padding with NaN,
          because a column-count drift in simulation output almost
          always means an upstream format change that the caller
          needs to notice.
        * Numeric parsing uses pandas default inference when
          ``spec.dtypes`` is not supplied. Non-numeric tokens in a
          purely-numeric column raise; this catches partial writes
          from a killed simulation that left a half-formed line at
          the end of the file.

        Args:
            path: File to read.
            spec: How to parse it.

        Returns:
            A pandas DataFrame with columns per ``spec.columns``.

        Raises:
            ValueError: On column-count mismatch or non-numeric
                data where numerics were expected.
            FileNotFoundError: If the file is missing (caller should
                have checked).
        """
        # Pandas engine choice:
        #
        # - No comment character: ``engine="c"`` is 2-3x faster on
        #   large files and tokenizes whitespace cleanly.
        # - ``comment`` set: the C engine mishandles the common
        #   shape where a file's first line is a commented header
        #   with leading whitespace (``"   # col1 col2 ..."``) AND
        #   the subsequent data rows also have leading whitespace.
        #   It raises ``EmptyDataError: No columns to parse from
        #   file``, which is a misleading error for a file that
        #   obviously has data. The python engine handles the
        #   shape correctly. ExaConstit's ``avg_stress_global.txt``
        #   and friends use exactly this layout, so any user who
        #   sets ``comment="#"`` gets the python engine
        #   automatically.
        #
        # ExaConstit output sizes are typically < 10k rows, so the
        # speed difference is negligible. Callers who know their
        # files don't have comments can still opt into the C engine
        # by setting ``comment=None`` on the spec.
        if spec.comment is None:
            engine = "c"
        else:
            engine = "python"

        if spec.delimiter is None:
            read_kwargs = dict(sep=r"\s+", engine=engine)
        else:
            read_kwargs = dict(sep=spec.delimiter, engine=engine)

        # header=None because we're providing our own column names;
        # otherwise pandas would try to interpret the first row as
        # a header and silently lose data.
        try:
            df = pd.read_csv(
                path,
                header=None,
                skiprows=spec.skip_rows,
                comment=spec.comment,
                usecols=list(spec.usecols) if spec.usecols is not None else None,
                dtype=dict(spec.dtypes) if spec.dtypes is not None else None,
                **read_kwargs,
            )
        except pd.errors.EmptyDataError as e:
            # Make the error message actionable. EmptyDataError from
            # the C engine in particular is frequently NOT "file is
            # empty" but "your comment shape tripped up the
            # tokenizer." Include the first few bytes of the file
            # to let the user see what pandas actually saw.
            try:
                with path.open("r") as f:
                    head = f.read(512)
            except OSError:
                head = "<could not read file for diagnostic>"
            raise ValueError(
                f"{path}: pandas reports the file has no parseable "
                f"columns. First 512 bytes:\n{head}\n"
                f"Common causes: (a) the file really is empty — "
                f"check the simulation stdout/stderr; (b) the spec's "
                f"``comment`` character matches every line; "
                f"(c) the delimiter is wrong (spec has "
                f"{spec.delimiter!r})."
            ) from e

        # Explicitly verify the column count matches what the spec
        # promised. A mismatch here is almost always a sign that the
        # output format has drifted upstream - catch it now rather
        # than let wrong-shaped arrays propagate into error metrics.
        expected = len(spec.columns)
        actual = df.shape[1]
        if actual != expected:
            raise ValueError(
                f"{path}: expected {expected} columns ({list(spec.columns)}), "
                f"got {actual}"
            )

        df.columns = list(spec.columns)
        return df


# --- Experimental-data loader --------------------------------------------


def load_experimental_csv(
    path: Union[str, Path],
    *,
    delimiter: Optional[str] = None,
    skip_rows: int = 0,
    comment: str = "#",
    columns: Optional[Sequence[str]] = None,
    dtypes: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Load experimental reference data from a CSV / TSV / whitespace file.

    A thin, stdlib-flavored convenience over ``pandas.read_csv`` for
    the common case of "I have a text file of experimental
    stress-strain pairs and I want a DataFrame back". Nothing magical
    - if the defaults do not suit your data, reach for pandas
    directly.

    There is intentionally no framework-enforced schema. Experimental
    data comes from many sources (tensile-test rigs, DIC software,
    other people's papers) with no consistent column names or units.
    The user decides on the schema; this loader just gets the bytes
    off disk into a DataFrame.

    Args:
        path: Path to the experimental data file.
        delimiter: Column separator. ``None`` (default) splits on any
            whitespace, which handles both ``.txt`` with spaces and
            tab-delimited files. Pass ``","`` for CSVs.
        skip_rows: Number of leading lines to skip (banner, units,
            etc.). Per-line ``#`` comments are handled separately via
            the ``comment`` parameter.
        comment: Line-start character marking comments. Defaults to
            ``"#"``.
        columns: Optional column names. If given, the file is assumed
            to have no header row and these names are applied. If
            omitted, the first data row is treated as a header per
            pandas defaults.
        dtypes: Optional per-column dtype overrides.

    Returns:
        A pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If pandas cannot parse the file as configured.

    Example:
        Load a whitespace-delimited file of (strain, stress) pairs
        with no header row::

            exp = load_experimental_csv(
                "data/exp_tension_298K.txt",
                columns=["strain", "stress"],
            )
            # exp["strain"] and exp["stress"] are float Series
    """
    # Figure out the sep kwarg for pandas. Same trick as in
    # TextTableReader: whitespace needs the regex pattern + python
    # engine to avoid an ambiguous-separator warning on some pandas
    # versions.
    if delimiter is None:
        sep_kwargs = dict(sep=r"\s+", engine="c")
    else:
        sep_kwargs = dict(sep=delimiter, engine="c")

    return pd.read_csv(
        path,
        header=None if columns is not None else "infer",
        names=list(columns) if columns is not None else None,
        skiprows=skip_rows,
        comment=comment,
        dtype=dict(dtypes) if dtypes is not None else None,
        **sep_kwargs,
    )


# --- Utility helpers often needed right after reading --------------------


def common_time_range(
    *dfs: pd.DataFrame, time_column: str = "Time"
) -> Tuple[float, float]:
    """Return the (tmin, tmax) interval common to every DataFrame.

    A read-adjacent helper that comes up often enough to belong here:
    simulation and experimental curves seldom start and end at the
    same time, and most comparison logic starts by clipping both
    series to their overlap.

    Args:
        *dfs: One or more DataFrames. Each must contain a column
            named ``time_column``.
        time_column: Name of the time-like column. Defaults to
            ``"Time"`` — ExaConstit's capital-T convention, matching
            the ``# Time  Volume  ...`` header its volume-averaged
            output writes.

    Returns:
        A tuple ``(tmin, tmax)`` describing the intersection of all
        input time ranges. Raises if the intersection is empty, so
        the caller does not silently compare non-overlapping curves.

    Raises:
        ValueError: If any DataFrame lacks ``time_column``, or if
            the intersection is empty.
        TypeError: If no DataFrames are given.

    Example:
        ExaConstit-produced DataFrames (the default)::

            tmin, tmax = common_time_range(sim_df, exp_df)

        Non-ExaConstit DataFrames using a lowercase ``"time"``
        column::

            tmin, tmax = common_time_range(
                sim_df, exp_df, time_column="time",
            )
    """
    if not dfs:
        raise TypeError("common_time_range requires at least one DataFrame")

    ranges = []
    for i, df in enumerate(dfs):
        if time_column not in df.columns:
            raise ValueError(
                f"DataFrame #{i} has no column {time_column!r}; "
                f"available: {list(df.columns)}"
            )
        t = df[time_column]
        ranges.append((float(t.min()), float(t.max())))

    tmin = max(r[0] for r in ranges)
    tmax = min(r[1] for r in ranges)
    if tmin >= tmax:
        raise ValueError(
            f"empty time intersection: tmin={tmin} >= tmax={tmax}"
        )
    return tmin, tmax


def interpolate_to(
    df: pd.DataFrame,
    target_times: np.ndarray,
    *,
    time_column: str = "Time",
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Linearly interpolate a time-series DataFrame onto new time points.

    Comes up constantly when comparing simulation to experiment: the
    two series are rarely sampled at the same time points, and any
    error metric needs them aligned first. Linear interpolation is
    the right default for stress-strain type data; for signals with
    known smoothness properties, use something like PCHIP instead
    (step 5 of the refactor plan covers this).

    Args:
        df: Source DataFrame. Must contain ``time_column``.
        target_times: 1-D array-like of times to interpolate onto.
            Should be monotonically non-decreasing.
        time_column: Name of the time column. Defaults to ``"Time"``
            — ExaConstit's capital-T convention.
        columns: Optional subset of columns to interpolate. ``None``
            (default) interpolates every non-time column.

    Returns:
        A new DataFrame whose ``time_column`` matches ``target_times``
        exactly and whose other columns are linearly interpolated.

    Raises:
        ValueError: If ``df`` lacks ``time_column``, or if
            ``target_times`` extends outside the source time range
            (we do not extrapolate silently).

    Example:
        Align a simulation curve onto experimental sample times
        (ExaConstit default)::

            sim_aligned = interpolate_to(
                sim_df,
                target_times=exp_df["Time"].to_numpy(),
            )

        Non-ExaConstit DataFrames with a lowercase ``"time"``
        column — override the keyword::

            sim_aligned = interpolate_to(
                sim_df,
                target_times=exp_df["time"].to_numpy(),
                time_column="time",
            )
    """
    if time_column not in df.columns:
        raise ValueError(
            f"DataFrame has no column {time_column!r}; "
            f"available: {list(df.columns)}"
        )
    t_src = df[time_column].to_numpy()
    target_times = np.asarray(target_times, dtype=float)

    # Refuse to extrapolate. Quietly extrapolating linear-interp
    # beyond the source data is a classic source of subtle bugs;
    # loudly rejecting it forces the caller to clip first (via
    # common_time_range) or explicitly widen the source.
    if target_times.min() < t_src.min() - 1e-12 or target_times.max() > t_src.max() + 1e-12:
        raise ValueError(
            f"target_times range [{target_times.min()}, {target_times.max()}] "
            f"is outside source range [{t_src.min()}, {t_src.max()}]; "
            f"clip with common_time_range() first"
        )

    cols = list(columns) if columns is not None else [
        c for c in df.columns if c != time_column
    ]
    out = {time_column: target_times}
    for c in cols:
        # numpy.interp is the lightweight linear interpolator. For
        # anything fancier (monotone cubic etc.) the next layer up
        # in the framework - smoothing - will intervene.
        out[c] = np.interp(target_times, t_src, df[c].to_numpy())
    return pd.DataFrame(out)
