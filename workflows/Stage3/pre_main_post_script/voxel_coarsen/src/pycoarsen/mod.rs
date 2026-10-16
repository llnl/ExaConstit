use numpy::PyArray1;
use pyo3::prelude::PyModuleMethods;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult, Python};

use crate::coarsen::voxel_coarsen;

#[pyfunction]
#[pyo3(name = "voxel_coarsen")]
fn voxel_coarsen_py<'py>(
    py: Python<'py>,
    file: &str,
    coarsen_size: usize,
) -> anyhow::Result<((usize, usize, usize), Bound<'py, PyArray1<i32>>)> {
    let result = voxel_coarsen(file, coarsen_size)?;
    Ok((result.0, PyArray1::from_vec(py, result.1)))
}

#[pymodule]
fn rust_voxel_coarsen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voxel_coarsen_py, m)?)?;
    Ok(())
}
