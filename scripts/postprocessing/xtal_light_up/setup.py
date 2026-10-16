"""
Builds the xtal_light_up Python package, whose xtal_light_up.xtal_light_up
extension module is compiled from the Rust code in src/ using setuptools-rust.

Requirements: a Rust toolchain (1.88 or newer, e.g. from https://rustup.rs),
Python 3.9 or newer, and ``pip install setuptools-rust setuptools wheel``.

Install from this directory with either of:
    pip install .
    python setup.py develop --user    # editable, user site-packages

See README.md for how to use the module with ExaConstit results.
"""
from setuptools import setup
from setuptools_rust import Binding, RustExtension

import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="xtal_light_up",
    version="0.1",
    rust_extensions=[RustExtension("xtal_light_up.xtal_light_up", binding=Binding.PyO3,debug=False)],
    packages=["xtal_light_up"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)