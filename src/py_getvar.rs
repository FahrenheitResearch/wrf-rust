use pyo3::prelude::*;

use crate::py_file::{self, WrfFile};

/// Compute a diagnostic variable from a WRF file.
///
/// Args:
///     wrffile: An open WrfFile handle.
///     name: Variable name (e.g. "temp", "slp", "sbcape").
///     timeidx: Time index (default 0).
///     units: Desired output units.
///     parcel_type: Parcel type for CAPE ("sb", "ml", "mu").
///     storm_motion: Custom storm motion (u, v) in m/s.
///     top_m: Integration top in meters AGL.
///     depth_m: Layer depth in meters AGL.
///
/// Returns:
///     numpy.ndarray with shape [ny, nx] or [nz, ny, nx].
#[pyfunction]
#[pyo3(signature = (wrffile, name, timeidx=None, units=None, parcel_type=None, storm_motion=None, top_m=None, depth_m=None))]
fn getvar<'py>(
    py: Python<'py>,
    wrffile: &WrfFile,
    name: &str,
    timeidx: Option<usize>,
    units: Option<String>,
    parcel_type: Option<String>,
    storm_motion: Option<(f64, f64)>,
    top_m: Option<f64>,
    depth_m: Option<f64>,
) -> PyResult<PyObject> {
    let opts = wrf_core::ComputeOpts {
        units,
        parcel_type,
        storm_motion,
        top_m,
        depth_m,
    };

    let result = wrf_core::getvar(wrffile.inner(), name, timeidx, &opts)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    py_file::to_numpy(py, result)
}

/// List all supported variable names.
#[pyfunction]
fn list_variables() -> Vec<(String, String, String)> {
    wrf_core::variables::VARS
        .iter()
        .map(|v| {
            (
                v.name.to_string(),
                v.description.to_string(),
                v.default_units.to_string(),
            )
        })
        .collect()
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(getvar, m)?)?;
    m.add_function(wrap_pyfunction!(list_variables, m)?)?;
    Ok(())
}
