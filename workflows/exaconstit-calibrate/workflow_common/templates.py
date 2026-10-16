"""
Template rendering with ``%%key%%`` placeholder substitution.

What this is for
----------------
Optimization and parameter-sweep workflows typically start from a
"master" input file for the simulation code, with a small number of
values that vary per case (strain rate, temperature, output filenames,
and so on). This module handles rendering a per-case input file by
substituting placeholder tokens in the master file with actual values.

The substitution is deliberately format-agnostic. The master file can
be TOML, XML, JSON, INI, a Bash script, or plain text - we operate on
the text content as a string, never parse it into a structured form.
That means the same machinery renders input files for ExaConstit, for
a completely different FEM code, or for a pre-processing utility, with
no changes.

Placeholder syntax
------------------
Placeholders look like ``%%identifier%%`` where the identifier is any
Python-style name (letters, digits, underscores; cannot start with a
digit). A few design choices worth noting:

* The leading and trailing ``%%`` are distinctive enough that they do
  not collide with content that appears naturally in the hosted file
  formats (TOML, XML, Bash, etc.). We considered Jinja-style ``{{ }}``
  but those appear in several real config formats.
* Values that are not already strings get converted via ``str(value)``.
  This is the simplest rule that works. For types whose string form is
  format-dependent (numpy arrays, nested lists, very small floats), the
  caller is expected to produce the exact desired string upstream - a
  different file format may want ``1.0e-3`` vs ``0.001`` vs ``1e-3``.
* Unknown placeholders that remain after substitution raise an error
  by default (``strict=True``). A silently-unreplaced ``%%foo%%`` in a
  generated simulation input file is almost always a bug that shows up
  far downstream as a parse error or a wrong value, so we fail loudly
  at render time instead. Set ``strict=False`` to allow them through
  unchanged.
* Substitution is single-pass, not recursive. If a value happens to
  contain a literal ``%%...%%``, that text is NOT re-examined for
  further substitution. This avoids injection surprises and is easy
  to reason about.

End-to-end example
------------------
Given a master file ``master_options.toml`` containing::

    [Problem]
        name = "sim_%%gene%%_%%obj%%"
        strain_rate = %%strain_rate%%
        temperature_k = %%temp_k%%
    [Properties]
        yield_stress = %%sigma_y%%
        hardening = %%H%%

and a per-case substitution mapping::

    {
        "gene":        3,
        "obj":         1,
        "strain_rate": 1.0e-3,
        "temp_k":      298.0,
        "sigma_y":     250.0,
        "H":           2500.0,
    }

the result of :func:`render_template_file` writing to
``wf/gen_0/gene_3_obj_1/options.toml`` is::

    [Problem]
        name = "sim_3_1"
        strain_rate = 0.001
        temperature_k = 298.0
    [Properties]
        yield_stress = 250.0
        hardening = 2500.0

Nothing else about the file - section headers, spacing, comments,
blank lines - is touched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Union

from ._fs import atomic_write_text

# Match %%key%% where key is a standard Python identifier. Compiled
# once at module load. The captured group is the identifier itself,
# without the surrounding percent signs.
_PLACEHOLDER_RE = re.compile(r"%%([A-Za-z_][A-Za-z0-9_]*)%%")


class UnresolvedPlaceholderError(KeyError):
    """Raised when ``render_template(..., strict=True)`` encounters a
    placeholder with no matching key in the substitution mapping.

    Inherits from ``KeyError`` so code that catches ``KeyError`` still
    catches this, but carries a richer message pointing to the
    offending key name and the set of keys that were available.
    """


def render_template(
    template: str,
    values: Mapping[str, object],
    *,
    strict: bool = True,
) -> str:
    """Substitute ``%%key%%`` placeholders in ``template`` with values.

    Args:
        template: Template string. May contain zero or more
            ``%%identifier%%`` placeholders. All other text is passed
            through unchanged. Multiple occurrences of the same
            placeholder are all replaced with the same value.
        values: Mapping from identifier to replacement value. Values
            that are not strings are stringified via ``str()``. The
            mapping does NOT need to contain every key present in
            the template if ``strict=False``.
        strict: If True (default), any placeholder left unreplaced at
            the end raises :class:`UnresolvedPlaceholderError`. If
            False, unknown placeholders are left verbatim in the
            output, which can be useful for multi-pass rendering or
            intentionally incomplete configurations.

    Returns:
        The rendered template as a new string. The input is not
        modified.

    Raises:
        UnresolvedPlaceholderError: In strict mode, if any
            ``%%key%%`` in the template is missing from ``values``.
            The error message includes the missing key name and the
            sorted list of keys that were available, to make typos
            easy to spot.

    Example:
        Render a tiny TOML fragment::

            from workflow_common import render_template
            text = render_template(
                "strain_rate = %%rate%%\\ntemperature = %%temp%%",
                {"rate": 1e-3, "temp": 298.0},
            )
            # text == "strain_rate = 0.001\\ntemperature = 298.0"

        Non-strict mode leaves unknown placeholders alone::

            render_template(
                "%%a%% %%b%%", {"a": "ok"}, strict=False,
            )
            # -> "ok %%b%%"
    """

    def repl(match: "re.Match[str]") -> str:
        """Inner callback invoked by ``re.sub`` for each placeholder.
        Looks up the captured key in ``values`` and returns either the
        stringified replacement or, in non-strict mode, the original
        ``%%key%%`` text unchanged.
        """
        key = match.group(1)
        if key in values:
            return str(values[key])
        if strict:
            raise UnresolvedPlaceholderError(
                f"Template placeholder %%{key}%% has no value in the "
                f"substitution mapping (available keys: {sorted(values)})"
            )
        # Non-strict: pass the original text through unchanged. We
        # return match.group(0) (the whole match) rather than
        # reconstructing "%%key%%" so odd Unicode inputs round-trip
        # exactly.
        return match.group(0)

    # Single regex pass over the entire template. Because we use
    # re.sub rather than a loop that repeatedly scans the output,
    # substituted text is not itself rescanned for placeholders -
    # there is no accidental recursion or infinite loop possible even
    # if a replacement value happens to contain "%%foo%%".
    return _PLACEHOLDER_RE.sub(repl, template)


def render_template_file(
    template_path: Union[str, Path],
    output_path: Union[str, Path],
    values: Mapping[str, object],
    *,
    strict: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Render a template read from disk and atomically write the result.

    Convenience wrapper that combines three common steps: reading a
    template file, substituting placeholders, and writing the rendered
    output so that readers never see a partial file.

    Args:
        template_path: Path to the master / template file to read.
        output_path: Path where the rendered file should be written.
            Parent directories are created as needed. Writing uses
            :func:`atomic_write_text` so a crash mid-write cannot
            leave a truncated output behind.
        values: Mapping of placeholder identifier to value. See
            :func:`render_template` for details.
        strict: Whether to raise on unresolved placeholders. Defaults
            to True.
        encoding: Text encoding for both read and write. Defaults to
            UTF-8.

    Raises:
        FileNotFoundError: If the template file does not exist.
        UnresolvedPlaceholderError: In strict mode, if any placeholder
            in the template is missing from ``values``.
        OSError: If the output cannot be written.

    Example:
        Render a per-case options file for one run::

            render_template_file(
                "master_options.toml",
                "wf/gen_0/gene_3/options.toml",
                {
                    "strain_rate": 1e-3,
                    "temp_k": 298.0,
                    "avg_stress_ext": "gene_3",
                },
            )
    """
    template_path = Path(template_path)
    output_path = Path(output_path)
    text = template_path.read_text(encoding=encoding)
    rendered = render_template(text, values, strict=strict)
    atomic_write_text(output_path, rendered, encoding=encoding)


def extract_placeholders(template: str) -> "set[str]":
    """Return the set of placeholder identifiers that appear in ``template``.

    Useful for validating a substitution mapping before rendering, for
    generating documentation of what a given master file expects, or
    for building interactive tooling (e.g. "these are the variables
    your template needs values for").

    Args:
        template: Template string to scan.

    Returns:
        A set of identifier strings. Each identifier appears once in
        the result regardless of how many times it occurs in the
        template.

    Example:
        ::

            extract_placeholders("rate=%%r%%, temp=%%t%%, rate again=%%r%%")
            # -> {"r", "t"}
    """
    return {m.group(1) for m in _PLACEHOLDER_RE.finditer(template)}
