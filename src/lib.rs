use pyo3::prelude::*;

mod py_file;
mod py_getvar;

/// The native Rust module exposed as wrf._wrf
#[pymodule]
fn _wrf(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_file::WrfFile>()?;
    py_getvar::register(py, m)?;
    Ok(())
}
