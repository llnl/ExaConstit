//! xtal_light_up: fast lattice strain / "light-up" post-processing for ExaConstit.
//!
//! Built as a Python extension module (see setup.py); the Python facing functions
//! live in [`pyxtal_light_up`].

extern crate anyhow;
extern crate num_traits as libnum;
extern crate numpy;
extern crate pyo3;

// pub mod xtal_light_up;
pub mod pyxtal_light_up;
