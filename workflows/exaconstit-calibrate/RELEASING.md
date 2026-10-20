# Releasing exaconstit-calibrate

This is a maintainer guide for cutting new versions of the package.
End users installing the package do not need to read this.

## TL;DR

```bash
# 0. One-time: install the build front-end (see Prerequisites)
pip install --upgrade build

# 1. Bump the version in pyproject.toml
# 2. Run the full test suite
python -m pytest tests/ -q

# 3. Build the artifacts
rm -rf dist build *.egg-info
python -m build

# 4. Sanity-check the wheel in a clean venv
python -m venv /tmp/releasecheck
/tmp/releasecheck/bin/pip install "dist/exaconstit_calibrate-<VERSION>-py3-none-any.whl[test]"
cp -r tests /tmp/releasecheck/
/tmp/releasecheck/bin/python -m pytest /tmp/releasecheck/tests -q

# 5. (Optional) tag and publish
git tag -a v<VERSION> -m "Release v<VERSION>"
git push origin v<VERSION>
# For PyPI: python -m twine upload dist/*
```

If any step fails, do not publish. Fix the failure and start over.

## Prerequisites

A one-time setup on your development machine:

```bash
pip install --upgrade build    # PEP 517 front-end
pip install --upgrade twine    # only if you publish to a package index
```

Both are stdlib-external but they are the standard Python Packaging
Authority tools and should be considered mandatory for any Python
release work. `build` replaces the old `python setup.py bdist_wheel`
invocation that PEP 517 deprecated.

You also need Python 3.12 or newer (same floor as the package itself
— building a wheel that requires `>=3.12` on an older Python would
silently produce metadata that's wrong for the interpreter running
the tests).

## Step 1 — Decide on a version number

The version lives in exactly one place: the `version = "X.Y.Z"` line
near the top of `pyproject.toml`. Everything else (wheel filename,
sdist filename, metadata, `importlib.metadata.version(...)`) derives
from that single string.

We follow [semantic versioning](https://semver.org/):

- **PATCH (0.1.0 → 0.1.1)** — bug fixes, no API changes. Users
  can pip-upgrade without reading release notes.
- **MINOR (0.1.0 → 0.2.0)** — new features, backward-compatible.
  Users can pip-upgrade but might want to read release notes.
- **MAJOR (0.1.0 → 1.0.0)** — breaking changes. Users need to
  update their code. Reserve this for genuine API rewrites.

Pre-1.0 versions have a softer contract: 0.X minor bumps can break
API if you call it out in the release notes. After 1.0, minor bumps
must not break API.

Example edit:

```toml
# pyproject.toml
[project]
name = "exaconstit-calibrate"
version = "0.2.0"              # was 0.1.0
```

Do NOT hardcode the version anywhere else. No `__version__` string
in `workflow_common/__init__.py`, no VERSION file. If you need
the version at runtime, use `importlib.metadata`:

```python
from importlib.metadata import version
v = version("exaconstit-calibrate")
```

## Step 2 — Run the tests

Before building anything, prove the code works. From the project
root:

```bash
python -m pytest tests/ -q
```

Expected output: `254 passed` (as of v0.1.0; the number grows as
tests are added). If any test fails, do not build a release —
fix the failure and re-run.

The test suite uses the fake simulation binary in
`tests/conftest.py`; you do not need ExaConstit or any HPC tool
installed to validate a release candidate.

## Step 3 — Clean the build directory

Stale artifacts from a previous build can cause subtle wrong-file
inclusions in the new build. Always wipe before building:

```bash
rm -rf dist build *.egg-info
```

`dist/` holds the final artifacts (wheel and sdist). `build/` is
setuptools' scratch area. `*.egg-info` is editable-install metadata
that should regenerate cleanly each time.

Also confirm the `build` front-end is installed in the environment
you plan to invoke it from:

```bash
python -c "import build; print('build available')" || \
    pip install --upgrade build
```

`build` is a maintainer tool, not a runtime dependency of the
package, so it is intentionally absent from
`[project.optional-dependencies]`. If you see "No module named
'build'", install it explicitly — it will not come in via
`pip install .[test]`.

## Step 4 — Build the wheel and sdist

```bash
python -m build
```

This produces two files in `dist/`:

- **`exaconstit_calibrate-<VERSION>-py3-none-any.whl`** — the wheel.
  Pure-Python, universal (py3, no platform tag). This is what
  `pip install` uses by default.
- **`exaconstit_calibrate-<VERSION>.tar.gz`** — the sdist (source
  distribution). Pip falls back to this if no wheel is available;
  PyPI requires both.

The names use underscores (`exaconstit_calibrate`) not hyphens
even though the distribution name in `pyproject.toml` uses hyphens
(`exaconstit-calibrate`). That's a packaging convention, not a
mistake: distribution names are normalized to underscores for
filenames.

Expect to see a lot of "adding 'workflow_common/...'" lines scroll
past. The build ends with:

```
Successfully built exaconstit_calibrate-<VERSION>.tar.gz and
exaconstit_calibrate-<VERSION>-py3-none-any.whl
```

If the build fails, the most common causes (in order of how often
you will hit them):

- **setuptools too old.** We require `>=77.0` for PEP 639 license
  expressions. Upgrade: `pip install --upgrade setuptools`.
- **Python too old.** `>=3.12` per the `requires-python` field.
- **Syntax error in pyproject.toml.** TOML is picky; a misplaced
  comma or missing quote will produce a cryptic error from
  setuptools. Validate with `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`.

## Step 5 — Sanity-check the wheel

**Never publish a wheel you have not test-installed.** The build
succeeding only proves the build tool was happy; it doesn't prove
the wheel is importable, has the right files, or passes tests.

Do this in a fresh venv, not your development environment:

```bash
python -m venv /tmp/releasecheck
/tmp/releasecheck/bin/pip install \
    "dist/exaconstit_calibrate-<VERSION>-py3-none-any.whl[test]"

# The wheel doesn't include the tests directory, so copy it over:
cp -r tests /tmp/releasecheck/

# Run the test suite against the installed package:
cd /tmp/releasecheck
./bin/python -m pytest tests -q
```

Expected: `254 passed` (or however many tests exist in this
version). If any test fails, the wheel is broken — do not publish.

Common things this test catches that the source-tree test did not:

- A Markdown file was forgotten in `[tool.setuptools.package-data]`
  and doesn't ship in the wheel.
- A helper module was added but its import path is wrong for an
  installed layout.
- The lazy `__getattr__` in `workflows/optimization/__init__.py`
  was accidentally replaced with an eager import, breaking the
  base-install path.

While you're here, also verify the documentation files made it
into the wheel:

```bash
./bin/python -c "
import workflow_common, pathlib
d = pathlib.Path(workflow_common.__file__).parent
for name in ('ARCHITECTURE.md', 'MIGRATION.md'):
    p = d / name
    print(name, 'present:', p.is_file(), 'size:', p.stat().st_size if p.is_file() else '(missing!)')
"
```

Both files should report sizes > 0.

## Step 6 — (Optional) Git tag the release

If the project is under version control (it is — this is in the
ExaConstit repo), mark the release with an annotated tag:

```bash
git add pyproject.toml
git commit -m "Release v<VERSION>"
git tag -a v<VERSION> -m "Release v<VERSION>"
git push origin main
git push origin v<VERSION>
```

The `-a` makes it an annotated tag (has its own commit-like object
and a message) rather than a lightweight tag. Annotated tags are
what GitHub uses to create Releases.

## Step 7 — (Optional) Publish to a package index

### To a GitHub Release

The low-friction option. Go to
https://github.com/LLNL/ExaConstit/releases, click "Draft a new
release", pick the tag from step 6, and attach both artifacts from
`dist/` as release assets. Users install with:

```bash
pip install https://github.com/LLNL/ExaConstit/releases/download/v<VERSION>/exaconstit_calibrate-<VERSION>-py3-none-any.whl
```

### To PyPI (public)

If LLNL decides to publish on PyPI:

```bash
python -m twine check dist/*      # metadata sanity check
python -m twine upload dist/*     # prompts for credentials
```

You will need a PyPI account with ownership of the
`exaconstit-calibrate` name. The first upload reserves the name;
subsequent uploads append new versions. PyPI does not allow
deleting or overwriting a published version — if you upload a
broken 0.2.0, you must release 0.2.1 to fix it.

### To an internal / HPC index

LLNL and other HPC sites often run an internal devpi or Nexus index.
`twine upload` accepts a `--repository-url` to target a non-PyPI
index; credentials go in `~/.pypirc`. Ask your site admin for the
URL and auth method.

## Step 8 — Announce

If there's a changelog (`CHANGELOG.md`, release notes section in
README, etc.), update it. Post-release is also a good moment to
bump the version in `pyproject.toml` to `0.X.Y+dev` or `0.X.(Y+1).dev0`
so main-branch installs are obviously not release builds. That's a
convention, not a requirement.

## Troubleshooting

### "No module named 'deap'" during `pytest` in the release check

The `[test]` extra should pull DEAP from the rcarson3 fork. If it
didn't, check `pyproject.toml` under `[project.optional-dependencies]`
— the `test` list must include
`"deap @ git+https://github.com/rcarson3/deap.git"`.

### "error in egg-info command" during `python -m build`

Setuptools is too old. Upgrade it: `pip install --upgrade setuptools`.
Version `>=77.0` is needed for the PEP 639 SPDX license expression.

### Wheel contains unexpected files / missing expected files

`[tool.setuptools.packages.find]` controls which packages ship.
`[tool.setuptools.package-data]` controls non-`.py` files (Markdown
docs, config files, etc.). Edit both and rebuild.

To inspect the wheel contents without installing:

```bash
python -c "
import zipfile
with zipfile.ZipFile('dist/exaconstit_calibrate-<VERSION>-py3-none-any.whl') as z:
    for n in z.namelist():
        print(n)
"
```

### `pip install` falls back to building from sdist instead of using the wheel

Your Python version or platform doesn't match the wheel's tags.
Since our wheel is `py3-none-any` (pure Python, any platform), this
usually means the user's Python is older than `requires-python`.
Check the `Requires-Python: >=3.12` line in the wheel's METADATA.

### A published version has a critical bug

You cannot overwrite or delete a published version on PyPI. Release
a patch version (`0.2.0` → `0.2.1`) with the fix. Document the known
bug in the GitHub Release notes for `0.2.0` so users know to skip
that version.

## Reference

- [Python Packaging User Guide](https://packaging.python.org/)
  — the canonical docs. Start here for anything not covered above.
- [PEP 517](https://peps.python.org/pep-0517/) — the build-system
  standard that `python -m build` implements.
- [PEP 639](https://peps.python.org/pep-0639/) — SPDX license
  expressions in pyproject.toml.
- [Semantic Versioning](https://semver.org/) — the version-number
  contract.
