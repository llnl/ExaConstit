"""
Path resolution: mapping workflow cases to working directories and
output file locations.

What this module does
---------------------
Given a "case" (a specific combination of generation, gene, and
objective in an optimization run), we need to know two things:

1. Where does the case live on disk? - its working directory
2. Where will a given output file appear after the simulation finishes?

Historically these paths were built inline with ad hoc ``os.path.join``
calls sprinkled across the workflow driver. That made them hard to
change when the simulation code's output layout evolved, and it made
it impossible to reuse the same driver with a different simulation
code that used a different layout.

This module replaces that ad-hoc approach with a small abstraction:
``PathResolver`` is a Protocol (see below) that describes what any path
layout implementation must provide, and ``TemplatePathResolver`` is a
general-purpose implementation driven by user-supplied format strings.

Protocols for readers unfamiliar with the term
----------------------------------------------
A Protocol is Python's way of expressing "any class that has these
methods is acceptable here, regardless of its inheritance". It is like
a C++ concept or a Rust trait. Users can write their own PathResolver
implementation (e.g. one that queries a database or runs a completely
custom path algorithm) and the framework will accept it as long as it
has a ``working_dir`` method and an ``output_file`` method with the
right signatures. No inheritance boilerplate required.

Two different placeholder schemes
---------------------------------
You will notice that path patterns in this module use Python's
standard ``{key}`` format-string syntax, whereas input-file templates
in :mod:`workflow_common.templates` use ``%%key%%``. This is
deliberate:

* Path patterns live inside Python source or config dicts supplied by
  the user. They never end up inside a simulation input file, so the
  fact that ``{`` and ``}`` are special in Python is fine - in fact
  it is convenient because ``str.format`` does the substitution for
  us and errors are caught immediately.
* Input-file templates live inside files that may themselves contain
  literal ``{`` and ``}`` characters (especially TOML arrays and XML
  attributes). A dedicated ``%%key%%`` syntax avoids those collisions
  and keeps the template scheme format-agnostic.

Do not try to unify them. The different contexts genuinely warrant
different tools.

Common directory layouts
------------------------
A few layouts that have worked well in practice, shown as
``working_dir_pattern`` values. Pick the one closest to your needs
or use them as starting points for your own.

Flat, indexed by gene/objective::

    working_dir_pattern = "cases/case_{gene:04d}_{obj}"
    # -> cases/case_0003_0

Hierarchical by generation (good for GA runs)::

    working_dir_pattern = "wf/gen_{generation}/gene_{gene}_obj_{obj}"
    # -> wf/gen_0/gene_3_obj_1

Per-RVE with generations nested inside (good for multi-RVE studies)::

    working_dir_pattern = (
        "by_rve/{rve_name}/gen_{generation}/gene_{gene}_obj_{obj}"
    )
    # -> by_rve/grain_32/gen_0/gene_3_obj_1
    # (Requires CaseContext(..., extra={"rve_name": "grain_32"}))

Flat mirror of a parameter sweep (no generations)::

    working_dir_pattern = "sweep/{rve_name}/T{temp_k:.0f}K_E{strain_rate:.0e}"
    # -> sweep/grain_32/T298K_E1e-03
    # (Requires the relevant fields in extra, and is fine passing
    #  generation=0, gene=0 unused.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Union, runtime_checkable


@dataclass(frozen=True)
class CaseContext:
    """A record identifying one simulation case within a workflow run.

    A "case" in this framework is a specific
    ``(generation, gene, obj)`` triple, matching the structure of a
    multi-objective genetic-algorithm optimization:

    * ``generation`` - the GA iteration number (0 for initial
      population, increments with each evolutionary step).
    * ``gene`` - the individual within that generation.
    * ``obj`` - the objective index for multi-objective runs (set to
      0 if the workflow is single-objective).

    Code whose workflow does not have one of these dimensions should
    still pass a value (commonly 0) - the framework treats them
    uniformly. The extra dimensions are cheap and they give uniform
    directory layouts across single-objective, multi-objective, and
    parameter-sweep runs.

    The ``extra`` mapping is for any additional data that a user's
    path pattern needs: RVE names, temperature values, loading
    direction labels, etc. Its contents are merged into the format
    mapping when resolving paths.

    Fields:
        generation: GA generation index.
        gene: Individual index within the generation.
        obj: Objective index for multi-objective runs; default 0.
        extra: Free-form mapping of additional names to values,
            available for use in path patterns.
        sim_case: Optional reference to the :class:`SimCase` this
            case was built from. Populated by :class:`Problem` when
            it dispatches work, so downstream components
            (property writers, templaters) can reach experiment-
            specific metadata without a separate plumbing path.
            ``None`` for contexts built outside a Problem (e.g.
            unit tests). Typed as ``Any`` to avoid an import cycle
            between :mod:`paths` and :mod:`problem`.

    Example:
        ::

            ctx = CaseContext(
                generation=0,
                gene=3,
                obj=1,
                extra={"rve_name": "grain_32", "temp_k": 298},
            )
    """

    generation: int
    gene: int
    obj: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)
    sim_case: Any = None

    def as_format_mapping(self) -> Dict[str, Any]:
        """Return a flat dict suitable for use with ``str.format_map``.

        Top-level fields and ``extra`` entries are merged into a
        single dict so that a format string like
        ``"gen_{generation}/{rve_name}/gene_{gene}"`` resolves
        correctly in one pass. Top-level fields take precedence on
        name collisions (so a user's ``extra={"generation": 999}``
        does not override the real generation number).

        A convenience alias ``gen`` is included as a synonym for
        ``generation``, because the existing ExaConstit directory
        conventions use ``gen_N`` in several places.

        Returns:
            A flat ``dict`` with all context fields and ``extra``
            entries ready for ``str.format_map``.
        """
        out: Dict[str, Any] = dict(self.extra)
        # Top-level keys overwrite any colliding keys from extra so
        # a user accidentally shadowing "generation" with an extra
        # value cannot produce wrong paths.
        out.update(
            generation=self.generation,
            gen=self.generation,  # alias matching legacy "gen_0/" layout
            gene=self.gene,
            obj=self.obj,
        )
        return out


@runtime_checkable
class PathResolver(Protocol):
    """Protocol describing how cases map to filesystem paths.

    A class that implements these two methods can be used as the
    framework's path resolver, regardless of its inheritance chain or
    module of origin. The framework uses only these two methods; any
    additional methods are visible to user code but ignored here.

    This is deliberately the *minimum* useful contract. Users with
    bespoke layouts (e.g. "all runs of the same RVE share a common
    directory that must be created once and linked into") can layer
    their own helpers on top of this core.

    Methods:
        working_dir(ctx):
            Returns the working directory for ``ctx``. Callers may
            create this directory and run the simulation inside it.
        output_file(logical_name, ctx):
            Returns the expected path of a named output artifact
            (e.g. ``"avg_stress"``). The caller decides what logical
            names are supported; the framework only requires that
            the mapping be stable across a run.

    Note on ``@runtime_checkable``: this decorator lets
    ``isinstance(obj, PathResolver)`` succeed for any object with
    matching methods, not just for explicit subclasses. Useful in
    defensive code that wants to accept duck-typed resolvers.
    """

    def working_dir(self, ctx: CaseContext) -> Path: ...

    def output_file(self, logical_name: str, ctx: CaseContext) -> Path: ...


class TemplatePathResolver:
    """Default :class:`PathResolver` implementation driven by format strings.

    This covers the common case where directory layouts can be
    described by simple Python ``str.format`` patterns with ``{key}``
    placeholders drawn from the :class:`CaseContext`. If you need
    something more elaborate (e.g. "look the directory up from a
    database"), write your own implementation of the Protocol.

    Path patterns can reference:

    * Any field on the :class:`CaseContext`: ``{generation}``,
      ``{gen}`` (alias), ``{gene}``, ``{obj}``.
    * Any key in ``ctx.extra`` or in the resolver's ``defaults``.
    * The special key ``{working_dir}`` when used in an output-file
      pattern - it expands to the result of ``working_dir(ctx)`` so
      you do not have to repeat the directory pattern.

    Args:
        working_dir_pattern: Format string describing the per-case
            working directory. Example:
            ``"wf/gen_{generation}/gene_{gene}_obj_{obj}"``.
        output_file_patterns: Mapping from logical output name to the
            format string that produces its full path. Example:
            ``{"avg_stress": "{working_dir}/avg_stress.txt"}``.
        defaults: Optional mapping of fixed name-to-value substitutions
            that apply to all cases. Useful for things like the
            simulation's basename or a root prefix that does not vary
            per case. Overridden by anything in ``ctx.extra`` with the
            same key.
        root: Optional root directory. If set, a relative working
            directory pattern is resolved against this root. Absolute
            patterns are passed through unchanged. Useful when the
            framework is invoked from a different ``cwd`` than the
            one it should write into.

    Example:
        ExaConstit-style layout where each case has its own directory
        with a ``results/<basename>/`` subtree holding averaged
        quantities::

            resolver = TemplatePathResolver(
                working_dir_pattern=(
                    "wf_files/gen_{generation}/gene_{gene}_obj_{obj}"
                ),
                output_file_patterns={
                    "avg_stress":
                        "{working_dir}/results/{basename}/avg_stress.txt",
                    "avg_def_grad":
                        "{working_dir}/results/{basename}/avg_def_grad.txt",
                },
                defaults={"basename": "options"},
                root="/scratch/run_42",
            )

            ctx = CaseContext(generation=0, gene=3, obj=1)
            resolver.working_dir(ctx)
            # -> PosixPath("/scratch/run_42/wf_files/gen_0/gene_3_obj_1")
            resolver.output_file("avg_stress", ctx)
            # -> PosixPath("/scratch/run_42/.../results/options/avg_stress.txt")
    """

    def __init__(
        self,
        working_dir_pattern: str,
        output_file_patterns: Mapping[str, str],
        *,
        defaults: Mapping[str, Any] = None,
        root: Union[str, Path, None] = None,
    ):
        self._working_dir_pattern = working_dir_pattern
        # Defensive copy so later mutations on the caller's dict cannot
        # surprise us with changed paths mid-run.
        self._output_file_patterns = dict(output_file_patterns)
        self._defaults = dict(defaults or {})
        self._root = Path(root) if root is not None else None

    def _format_mapping(self, ctx: CaseContext) -> Dict[str, Any]:
        """Build the full key/value mapping used for substitution.

        Precedence rules:

        1. Start with ``self._defaults`` (static, lowest priority).
        2. Overlay ``ctx.extra`` and the top-level fields from ``ctx``
           via :meth:`CaseContext.as_format_mapping`.

        This means context fields win over defaults, and top-level
        context fields win over ``extra`` entries with the same name.

        Args:
            ctx: The case context to resolve.

        Returns:
            A merged dict ready for ``str.format_map``.
        """
        m = dict(self._defaults)
        m.update(ctx.as_format_mapping())
        return m

    def working_dir(self, ctx: CaseContext) -> Path:
        """Resolve the per-case working directory.

        Substitutes the ``working_dir_pattern`` with values drawn from
        ``ctx`` (top-level fields and ``extra``) and from the static
        ``defaults``. If the pattern produces a relative path and a
        ``root`` was supplied to the constructor, the root is
        prepended.

        Args:
            ctx: The case context.

        Returns:
            The resolved working directory as a :class:`pathlib.Path`.
            The path is NOT created on disk here; callers that need
            the directory to exist should call ``.mkdir(parents=True,
            exist_ok=True)`` themselves. Keeping creation separate
            from resolution makes this method side-effect free and
            easier to reason about.

        Raises:
            KeyError: If the pattern references a placeholder that is
                not available in ``ctx`` or ``defaults``. The error
                message names the missing key and lists what was
                available, so typos are obvious.
        """
        m = self._format_mapping(ctx)
        try:
            rendered = self._working_dir_pattern.format_map(m)
        except KeyError as e:
            # str.format_map raises KeyError with just the missing key
            # name as its argument. Reraise with a more helpful message
            # that includes what was available, so users do not have
            # to guess.
            raise KeyError(
                f"working_dir_pattern {self._working_dir_pattern!r} "
                f"references {{{e.args[0]}}} but it is not in the context "
                f"(available: {sorted(m)})"
            ) from None
        p = Path(rendered)
        if self._root is not None and not p.is_absolute():
            p = self._root / p
        return p

    def output_file(self, logical_name: str, ctx: CaseContext) -> Path:
        """Resolve the path of a named output file for ``ctx``.

        Logical names are framework-internal labels that the caller
        chose when configuring the resolver (``"avg_stress"``,
        ``"avg_def_grad"``, ``"plastic_work"``, etc.). Keeping the
        logical-to-physical mapping in the resolver rather than
        scattered across the driver lets the same driver work with
        different simulation codes that name their outputs differently.

        Args:
            logical_name: The framework-internal name of the output
                file. Must match one of the keys given to the
                constructor as ``output_file_patterns``.
            ctx: The case context.

        Returns:
            The resolved path of the output file.

        Raises:
            KeyError: If ``logical_name`` is not a known output, or
                if the pattern references an unavailable placeholder.
                In the latter case the error message lists what was
                available.
        """
        if logical_name not in self._output_file_patterns:
            raise KeyError(
                f"No output_file pattern registered for {logical_name!r}. "
                f"Known: {sorted(self._output_file_patterns)}"
            )
        pattern = self._output_file_patterns[logical_name]
        m = self._format_mapping(ctx)
        # Expose the working dir as a format key so output patterns can
        # compose against it without having to duplicate the working_dir
        # pattern in every entry. Adds it only for this call, does not
        # mutate the resolver.
        m["working_dir"] = str(self.working_dir(ctx))
        try:
            rendered = pattern.format_map(m)
        except KeyError as e:
            raise KeyError(
                f"output_file pattern for {logical_name!r} references "
                f"{{{e.args[0]}}} but it is not in the context"
            ) from None
        p = Path(rendered)
        # Symmetry fix: if the rendered path is relative AND the
        # pattern didn't use {working_dir} (so the user wasn't
        # explicitly building from it), treat it as relative to the
        # case's working directory. Matches how working_dir_pattern
        # auto-prepends root for relative results. Without this,
        # patterns like ``"results/avg_stress.txt"`` resolve against
        # the driver's cwd, silently miss the case dir, and cases
        # look empty at read time — a silent-wrong-path bug class
        # that has burned real runs. An absolute pattern still wins
        # (users who wanted an absolute path get what they asked for).
        if not p.is_absolute() and "{working_dir}" not in pattern:
            p = self.working_dir(ctx) / p
        return p

    def known_outputs(self) -> "list[str]":
        """Return the list of logical output names this resolver knows about.

        Useful for logging, for introspection tools, and for validating
        that a driver is referencing only outputs that have been
        configured.

        Returns:
            A list of logical name strings, in registration order.
        """
        return list(self._output_file_patterns)
