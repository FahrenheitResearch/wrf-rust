use pyo3::prelude::*;

use crate::py_file::{self, WrfFile};
use crate::py_opts;

/// Compute a diagnostic variable from a WRF file.
#[pyfunction]
#[pyo3(signature = (wrffile, name, timeidx=None, units=None, parcel_type=None, storm_motion=None, top_m=None, bottom_m=None, depth_m=None, parcel_pressure=None, parcel_temperature=None, parcel_dewpoint=None, bottom_p=None, top_p=None, layer_type=None, use_virtual=None, lake_interp=None, use_varint=None, use_liqskin=None))]
fn getvar<'py>(
    py: Python<'py>,
    wrffile: &WrfFile,
    name: &str,
    timeidx: Option<usize>,
    units: Option<String>,
    parcel_type: Option<String>,
    storm_motion: Option<Py<PyAny>>,
    top_m: Option<f64>,
    bottom_m: Option<f64>,
    depth_m: Option<f64>,
    parcel_pressure: Option<f64>,
    parcel_temperature: Option<f64>,
    parcel_dewpoint: Option<f64>,
    bottom_p: Option<f64>,
    top_p: Option<f64>,
    layer_type: Option<String>,
    use_virtual: Option<bool>,
    lake_interp: Option<f64>,
    use_varint: Option<bool>,
    use_liqskin: Option<bool>,
) -> PyResult<PyObject> {
    let opts = py_opts::build_compute_opts(
        py,
        wrffile.inner().ny,
        wrffile.inner().nx,
        units,
        parcel_type,
        storm_motion,
        top_m,
        bottom_m,
        depth_m,
        parcel_pressure,
        parcel_temperature,
        parcel_dewpoint,
        bottom_p,
        top_p,
        layer_type,
        use_virtual,
        lake_interp,
        use_varint,
        use_liqskin,
    )?;

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
