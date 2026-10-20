"""
Per-case file preparation: rendering input templates and writing
material-property files.

Why this module exists
----------------------
Before a simulation can run, its working directory needs a few files:

1. **Input files** (e.g. ``options.toml``) — render the master
   template with per-case substitutions (gene index, strain rate,
   temperature, and so on). This is the :class:`CaseTemplater`
   layer.

2. **Property files** (e.g. ``material_props.toml``) — encode the
   optimizer's current gene vector into whatever on-disk format the
   simulation code expects. ExaConstit reads its Voce hardening
   parameters, elastic constants, etc. from a file that differs from
   the main options file. This is the :class:`PropertyWriter` layer.

Splitting these into two modules matters because input files are
mostly static per objective - they change when the strain rate or
temperature changes but not per gene - while property files change
on every gene evaluation. Keeping them separate lets callers cache
or skip the more expensive work when appropriate.

Relationship to :mod:`workflow_common.templates`
------------------------------------------------
The low-level :mod:`~workflow_common.templates` module renders one
text file at a time with ``%%key%%`` substitution. This module wraps
it in a higher-level abstraction that knows about case layouts and
can render several files in one call. If you only need single-file
rendering, use ``render_template_file`` directly; if you have a
per-case "render these 3 templates into the case directory" step,
use :class:`CaseTemplater`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from ._fs import atomic_write_text
from .logging_utils import get_logger
from .results import CaseLayout
from .templates import render_template_file

logger = get_logger(__name__)


def _validate_case_relative_dest(dest: str, field_name: str) -> None:
    """Validate that a destination filename is safe to join with a
    case's working directory.

    Rejects the shapes that would silently escape the case directory:

    * absolute paths — ``Path("/abs") / "rel/..."`` on POSIX drops
      the working_dir prefix entirely, so an absolute ``dest`` would
      land at ``/abs`` regardless of which case's layout we thought
      we were writing to
    * ``..`` parent-directory references — escape the per-case
      isolation the framework depends on (one case's writes
      clobbering another's or escaping the workspace entirely)
    * ``{...}`` placeholder syntax — this is not a
      :class:`PathResolver` template; the resolver has its own
      richer mapping for per-case paths. Placeholders here would be
      written verbatim, producing files with literal ``{gene}`` in
      their names

    Called from the ``__post_init__`` of every dataclass that has a
    ``dest`` field so the failure surfaces at construction time
    rather than after the first case runs.

    Args:
        dest: The user-supplied destination filename / relative path.
        field_name: Human-readable field name for the error message
            (``"TemplateTarget.dest"`` etc.). Helps a user back
            from error to the offending source line.

    Raises:
        ValueError: If ``dest`` matches any of the rejected shapes
            above. The message names ``field_name`` and explains
            what shapes are allowed.
    """
    if not isinstance(dest, str):
        raise TypeError(
            f"{field_name}: expected str, got {type(dest).__name__}"
        )
    if not dest:
        raise ValueError(f"{field_name}: must not be empty")
    p = Path(dest)
    if p.is_absolute():
        raise ValueError(
            f"{field_name}={dest!r}: must be a relative path, not "
            f"absolute. The framework joins each entry with the "
            f"case's working directory; an absolute path would "
            f"land outside the case dir and quietly clobber files "
            f"elsewhere."
        )
    if ".." in p.parts:
        raise ValueError(
            f"{field_name}={dest!r}: parent-directory references "
            f"('..') are not supported. Files must land inside the "
            f"case's working directory."
        )
    if "{" in dest or "}" in dest:
        raise ValueError(
            f"{field_name}={dest!r}: contains '{{' or '}}'. "
            f"Placeholder syntax belongs in the resolver's "
            f"output_file_patterns, not here. Use a plain relative "
            f"filename like 'options.toml'."
        )


# --- CaseTemplater ------------------------------------------------------


@dataclass(frozen=True)
class TemplateTarget:
    """One (source template, destination) pair for :class:`CaseTemplater`.

    Each target tells the templater to read a master template from
    ``source``, substitute ``%%key%%`` placeholders, and write the
    result to ``dest`` inside a case's working directory. One
    :class:`CaseTemplater` can process many targets in a single call.

    Fields:
        source: Absolute or relative path to the master template
            file. Read once per :meth:`CaseTemplater.render` call.
            Must exist at render time.
        dest: Destination filename, relative to the case's working
            directory. A typical value is ``"options.toml"``. Nested
            paths (``"cfg/options.toml"``) are supported; parent
            directories are created as needed.
        strict: If True (default), unresolved placeholders raise
            :class:`~workflow_common.templates.UnresolvedPlaceholderError`.
            If False, unknown placeholders pass through verbatim,
            which is occasionally useful for multi-pass templating.
        encoding: Text encoding for reading and writing. Defaults
            to UTF-8 which handles every real simulation input
            format.
        substitute: If True (default), the source file is read and
            ``%%key%%`` placeholders are substituted from the
            templater's ``values`` mapping. If False, the source
            is byte-copied verbatim to ``dest`` with no parsing,
            no substitution, and no encoding assumption. Use
            ``substitute=False`` to stage binary or binary-safe
            files alongside rendered text: grain-orientation
            quaternion files, HDF5 state dumps, mesh binaries,
            reference images, etc. ``strict`` and ``encoding`` are
            ignored when ``substitute=False``.

    Example:
        Two targets for an ExaConstit case: the main options file and
        an auxiliary mesh-configuration file::

            targets = [
                TemplateTarget(
                    source=Path("templates/master_options.toml"),
                    dest="options.toml",
                ),
                TemplateTarget(
                    source=Path("templates/mesh_config.toml"),
                    dest="mesh.toml",
                ),
            ]

        Staging a binary orientation file alongside the rendered
        options.toml — the ori file is not text, so tell the
        templater to copy it verbatim::

            targets = [
                TemplateTarget(
                    source=Path("templates/master_options.toml"),
                    dest="options.toml",
                ),
                TemplateTarget(
                    source=Path("orientations/voce_quats.ori"),
                    dest="voce_quats.ori",
                    substitute=False,
                ),
            ]
    """

    source: Path
    dest: str
    strict: bool = True
    encoding: str = "utf-8"
    substitute: bool = True

    def __post_init__(self):
        _validate_case_relative_dest(self.dest, "TemplateTarget.dest")


class CaseTemplater:
    """Renders one or more per-case input files from master templates.

    A :class:`CaseTemplater` is a bundle of :class:`TemplateTarget`
    records plus a render method that applies them all to a given
    :class:`CaseLayout` + value mapping. Typical usage: one
    templater instance per problem, reused across every case.

    Args:
        targets: Sequence of :class:`TemplateTarget` defining the
            files to render. Order does not matter for correctness
            but is preserved for logging.

    Raises:
        ValueError: If ``targets`` is empty. A templater with no
            targets is almost always a configuration mistake -
            either the caller meant to pass targets and forgot, or
            they do not actually need a templater at all. Either
            way we would rather raise loudly than silently no-op.

    Example:
        Render two files into each case directory. ``values`` holds
        the per-case substitutions (gene index, physical parameters,
        etc.)::

            templater = CaseTemplater([
                TemplateTarget(Path("master.toml"), "options.toml"),
                TemplateTarget(Path("mesh.toml"),  "mesh.toml"),
            ])

            for ctx in contexts:
                layout = CaseLayout(ctx, resolver)
                templater.render(layout, values={
                    "strain_rate": 1e-3,
                    "temp_k": 298.0,
                    "gene": ctx.gene,
                })
    """

    def __init__(self, targets: Sequence[TemplateTarget]):
        # Empty targets is valid: users whose per-case setup is
        # entirely handled by the PropertyWriter don't need to
        # render any templates. render() becomes a no-op in that
        # case. This avoids forcing users to invent a dummy
        # placeholder target just to satisfy the constructor.
        self._targets: tuple = tuple(targets)

    @property
    def targets(self) -> Sequence[TemplateTarget]:
        """Read-only view of the configured targets."""
        return self._targets

    def render(
        self,
        layout: CaseLayout,
        values: Mapping[str, Any],
    ) -> List[Path]:
        """Render every configured template into ``layout``'s working dir.

        The working directory is created if missing. For each target,
        the source file is read, placeholders are substituted from
        ``values``, and the result is written atomically (tempfile
        + rename) so a crash mid-write cannot leave a truncated file.

        Args:
            layout: The :class:`CaseLayout` describing where to write
                the rendered files. Its working directory is used
                as the base for relative destination paths.
            values: Mapping of placeholder identifier to replacement
                value. Values that are not strings are stringified
                via ``str()``. The same mapping is used for every
                target, so values that only some templates need are
                fine - extras do not cause errors.

        Returns:
            List of rendered-file paths, in the order the targets
            were configured. Useful for logging or for downstream
            tools that need to inspect the rendered output.

        Raises:
            FileNotFoundError: If any source template is missing.
            UnresolvedPlaceholderError: If a strict, substituting
                target has a placeholder with no value in ``values``.
                Non-strict targets do not raise;
                ``substitute=False`` targets do not parse for
                placeholders at all, so this error is impossible
                for them.
            OSError: On any write failure.
        """
        # Create the case's working directory up front. It may already
        # exist (the caller often creates it before calling the
        # templater); mkdir with exist_ok handles both cases.
        layout.working_dir.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for target in self._targets:
            dest_path = layout.working_dir / target.dest
            if target.substitute:
                logger.debug(
                    "rendering template %s -> %s", target.source, dest_path
                )
                render_template_file(
                    target.source,
                    dest_path,
                    values,
                    strict=target.strict,
                    encoding=target.encoding,
                )
            else:
                # Byte-copy path. Binary-safe (no decode / encode round
                # trip) so it's suitable for .ori files, HDF5 dumps,
                # mesh binaries, etc. The parent dir is already created
                # above via layout.working_dir.mkdir; nested dest paths
                # need their own subdir created here.
                logger.debug(
                    "copying (no substitution) %s -> %s",
                    target.source, dest_path,
                )
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # shutil.copyfile preserves bytes exactly and does not
                # copy permission bits (we don't want to propagate
                # source-file permissions into the case dir).
                import shutil as _shutil
                _shutil.copyfile(target.source, dest_path)
            written.append(dest_path)
        return written


# --- PropertyWriter Protocol + implementations -------------------------


@runtime_checkable
class PropertyWriter(Protocol):
    """Protocol: anything that writes per-gene material properties.

    A property writer takes the optimizer's current gene vector
    (a sequence of floating-point numbers) and encodes it into
    a file the simulation code can read. The file format is
    code-specific - text lines, TOML, binary, HDF5 - so there is
    one Protocol and several concrete implementations.

    Methods:
        write(layout, gene, param_names):
            Write the properties for this case.
            ``gene`` is the numerical parameter vector.
            ``param_names`` gives human-readable names aligned to
            ``gene`` by index. Return the path of the written file,
            which the caller may log or pass to downstream tools.
    """

    def write(
        self,
        layout: CaseLayout,
        gene: Sequence[float],
        param_names: Sequence[str],
    ) -> Path: ...


@dataclass
class TemplatePropertyWriter:
    """Writes properties by substituting into a master template.

    This is the most flexible writer: the user supplies a template
    file using ``%%name%%`` placeholders for each parameter, and the
    writer fills them in from the gene vector at call time. Works
    for any text-based property format (TOML, JSON, XML, INI).

    Args:
        template_path: Master template with ``%%name%%`` placeholders.
            The placeholder names must match ``param_names`` given
            at ``write()`` time.
        dest: Destination filename relative to the case working
            directory. Default ``"properties.toml"``.
        extra_values: Optional extra substitutions applied on every
            write - for things that vary per case but are not part
            of the gene (e.g. a case-specific seed). These are
            merged INTO the gene-derived values with the gene
            values taking precedence on name collision.
        strict: If True (default), unresolved placeholders raise.

    Example:
        Suppose ``master_props.toml`` contains::

            [Voce]
                yield_stress = %%yield_stress%%
                hardening    = %%hardening%%
                saturation   = %%saturation%%

        and the optimizer is evolving a 3-vector aligned to those
        three parameter names::

            writer = TemplatePropertyWriter(
                template_path=Path("master_props.toml"),
                dest="properties.toml",
            )
            writer.write(
                layout,
                gene=[200.0, 2000.0, 5.0],
                param_names=["yield_stress", "hardening", "saturation"],
            )

        The resulting ``properties.toml`` contains the substituted
        values.
    """

    template_path: Path
    dest: str = "properties.toml"
    extra_values: Mapping[str, Any] = field(default_factory=dict)
    strict: bool = True

    def __post_init__(self):
        _validate_case_relative_dest(self.dest, "TemplatePropertyWriter.dest")

    def write(
        self,
        layout: CaseLayout,
        gene: Sequence[float],
        param_names: Sequence[str],
    ) -> Path:
        """Render the template with per-gene substitutions.

        Args:
            layout: The case's :class:`CaseLayout`.
            gene: The numerical parameter vector. Must be the same
                length as ``param_names``.
            param_names: Placeholder names aligned to ``gene``.

        Returns:
            The path of the written file (inside the case's
            working directory).

        Raises:
            ValueError: If ``gene`` and ``param_names`` differ in
                length. A length mismatch is a logic error in the
                caller, not a user-data problem, so we fail fast.
        """
        if len(gene) != len(param_names):
            raise ValueError(
                f"gene has {len(gene)} entries but param_names has "
                f"{len(param_names)}: {list(param_names)}"
            )

        # Build the substitution mapping. Start with extra_values
        # (the static, per-case additions), then overlay the gene-
        # derived values so gene values win on name collision.
        values: Dict[str, Any] = dict(self.extra_values)
        for name, val in zip(param_names, gene):
            values[name] = val

        dest_path = layout.working_dir / self.dest
        logger.debug(
            "TemplatePropertyWriter writing %d params to %s",
            len(gene), dest_path,
        )
        render_template_file(
            self.template_path, dest_path, values, strict=self.strict
        )
        return dest_path


@dataclass
class CallablePropertyWriter:
    """Delegates property-file writing to a user-supplied callable.

    The most flexible PropertyWriter. Use this when the simulation
    code needs a property file whose format isn't a clean match for
    :class:`TemplatePropertyWriter` (template-substitution) or
    :class:`DelimitedPropertyWriter` (one-value-per-line). Crystal
    plasticity codes often want a structured property block with
    group headers, model-specific keys, derived quantities (e.g.
    ratios of gene values), or values that depend on SimCase
    metadata — things none of the canned writers handle well.

    The callable gets four inputs:

    * ``case_dir`` — absolute path to the case's working directory.
      Write the property file INSIDE this directory.
    * ``gene`` — the numerical parameter vector (same order as
      ``param_names``).
    * ``param_names`` — human-readable gene names for logging or
      for writing a header row.
    * ``sim_case`` — the :class:`~workflow_common.problem.SimCase`
      for this evaluation. Lets the callable produce different
      property files per experiment (e.g. picking an orientation
      file, adjusting derived quantities for temperature).

    The callable must return the path of the file it wrote, so the
    framework can log it and surface it in the manifest. Writing
    multiple files from a single callable is fine — just return
    the "main" one (the one the sim binary opens directly).

    This class deliberately does NOT care what format you write:
    text, binary, HDF5, multiple files, whatever. It's a shim
    between the framework's PropertyWriter protocol and a plain
    Python function.

    Args:
        func: The user callable. Signature:
            ``(case_dir: Path, gene: Sequence[float], param_names: Sequence[str], sim_case: SimCase) -> Path``.
        dest_hint: Logical "main file" name for logging only. Does
            not constrain what the callable writes. Default
            ``"properties.txt"``. The callable's return value wins
            for the actual path.

    Example:
        Write an ExaCMech-style properties block with named groups
        and per-experiment orientation file reference::

            def write_exacmech_properties(case_dir, gene, names, sim_case):
                # Unpack named gene positions.
                g0_1, g0_2, g0_3, sat, *rate_params, exp = gene
                # Pull the orientation filename from SimCase metadata.
                ori = sim_case.case_data["ori_file"]
                path = case_dir / "properties.txt"
                with path.open("w") as f:
                    f.write("[material]\\n")
                    f.write(f"orientation_file = \"{ori}\"\\n")
                    f.write("[slip_systems]\\n")
                    f.write(f"g0 = [{g0_1}, {g0_2}, {g0_3}]\\n")
                    f.write(f"saturation = {sat}\\n")
                    f.write("[rate_kinetics]\\n")
                    for i, v in enumerate(rate_params):
                        f.write(f"gdot_0_{i + 1} = {v}\\n")
                    f.write(f"exponent = {exp}\\n")
                return path

            writer = CallablePropertyWriter(func=write_exacmech_properties)
    """

    func: Callable[..., Path]
    dest_hint: str = "properties.txt"

    def write(
        self,
        layout: CaseLayout,
        gene: Sequence[float],
        param_names: Sequence[str],
    ) -> Path:
        """Invoke the user callable and return the path it reports.

        Passes through any exception raised by the callable so that
        configuration errors surface loudly at the first case rather
        than being swallowed as "this case failed." Callables should
        raise immediately on malformed gene vectors; the framework's
        failure handler only catches sim-binary-level failures.
        """
        # ctx.sim_case is populated by Problem when it dispatches
        # work. None means this writer was called outside a Problem
        # (typically a unit test); the callable may still run if it
        # doesn't need the SimCase reference.
        sim_case = layout.ctx.sim_case
        path = self.func(
            layout.working_dir, gene, list(param_names), sim_case,
        )
        if not isinstance(path, Path):
            raise TypeError(
                f"{self.func.__name__}() must return a pathlib.Path; "
                f"got {type(path).__name__}"
            )
        return path


@dataclass
class DelimitedPropertyWriter:
    """Writes gene values as a simple delimited text file.

    For simulation codes that read a bare list of numbers from disk
    (one per line, or tab/space-separated) rather than a structured
    format. No template file required; the format is entirely
    determined by the writer's configuration.

    Args:
        dest: Destination filename relative to the case working
            directory. Default ``"properties.txt"``.
        delimiter: String placed between values. Default ``"\\n"``
            (one value per line). Use ``" "`` for space-separated,
            ``","`` for CSV, ``"\\t"`` for TSV.
        include_names: If True, the file starts with a header line
            listing ``param_names`` (delimiter-joined). Default
            False - most simulation codes expect a bare number list
            and would fail to parse a header line.
        fmt: ``%``-style format string for each value. Default
            ``"%g"`` uses Python/C's shortest-round-trip formatting,
            which is the safest choice for scientific codes that
            do their own parsing. Use ``"%.17g"`` if you need
            guaranteed bit-exact round-trip for IEEE 754 doubles.

    Example:
        A plain text file with one value per line::

            writer = DelimitedPropertyWriter(dest="props.txt")
            writer.write(
                layout,
                gene=[200.0, 2000.0, 5.0],
                param_names=["yield", "hardening", "sat"],
            )

        produces a props.txt containing three lines: ``200``,
        ``2000``, ``5``.

        A CSV with a header::

            writer = DelimitedPropertyWriter(
                dest="props.csv", delimiter=",", include_names=True,
            )
            # -> "yield,hardening,sat\\n200,2000,5\\n"
    """

    dest: str = "properties.txt"
    delimiter: str = "\n"
    include_names: bool = False
    fmt: str = "%g"

    def __post_init__(self):
        _validate_case_relative_dest(self.dest, "DelimitedPropertyWriter.dest")

    def write(
        self,
        layout: CaseLayout,
        gene: Sequence[float],
        param_names: Sequence[str],
    ) -> Path:
        """Format gene values and write them atomically.

        Args:
            layout: The case's :class:`CaseLayout`.
            gene: The numerical parameter vector.
            param_names: Names aligned to ``gene``. Used only if
                ``include_names`` is True.

        Returns:
            The path of the written file.

        Raises:
            ValueError: If ``include_names=True`` and the lengths
                of ``gene`` and ``param_names`` do not match.
        """
        if self.include_names and len(gene) != len(param_names):
            raise ValueError(
                f"gene has {len(gene)} entries but param_names has "
                f"{len(param_names)} and include_names=True"
            )

        # Format each value with the configured fmt string. Using
        # a list + join is the idiomatic and fastest way to build
        # a delimited string in Python.
        lines: List[str] = []
        if self.include_names:
            lines.append(self.delimiter.join(param_names))
        lines.append(self.delimiter.join(self.fmt % v for v in gene))

        # Rows are joined by newlines regardless of the per-value
        # delimiter. Using the delimiter here would mangle multi-row
        # output formats like CSV where columns are comma-separated
        # but rows are newline-separated. A trailing newline is tacked
        # on because virtually every downstream parser tolerates it,
        # and many require it for the final line to be read.
        text = "\n".join(lines) + "\n"

        dest_path = layout.working_dir / self.dest
        atomic_write_text(dest_path, text)
        return dest_path
