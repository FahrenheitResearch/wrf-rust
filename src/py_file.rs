use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::py_opts;

/// A WRF output file handle.
///
/// Opens a NetCDF WRF output file and provides access to grid metadata
/// and derived diagnostic variables.
#[pyclass(name = "WrfFile")]
pub struct WrfFile {
    inner: wrf_core::WrfFile,
}

#[pymethods]
impl WrfFile {
    /// Open a WRF output file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = wrf_core::WrfFile::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of grid points in the x (west-east) direction.
    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx
    }

    /// Number of grid points in the y (south-north) direction.
    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny
    }

    /// Number of vertical levels.
    #[getter]
    fn nz(&self) -> usize {
        self.inner.nz
    }

    /// Number of time steps.
    #[getter]
    fn nt(&self) -> usize {
        self.inner.nt
    }

    /// Grid spacing in x (meters).
    #[getter]
    fn dx(&self) -> f64 {
        self.inner.dx
    }

    /// Grid spacing in y (meters).
    #[getter]
    fn dy(&self) -> f64 {
        self.inner.dy
    }

    /// File path.
    #[getter]
    fn path(&self) -> String {
        self.inner.path.display().to_string()
    }

    /// Time strings from the file.
    fn times(&self) -> PyResult<Vec<String>> {
        self.inner
            .times()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Compute a diagnostic variable.
    #[pyo3(signature = (name, timeidx=None, units=None, parcel_type=None, storm_motion=None, top_m=None, bottom_m=None, depth_m=None, parcel_pressure=None, parcel_temperature=None, parcel_dewpoint=None, bottom_p=None, top_p=None, layer_type=None, use_virtual=None, lake_interp=None, use_varint=None, use_liqskin=None))]
    fn getvar<'py>(
        &self,
        py: Python<'py>,
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
            self.inner.ny,
            self.inner.nx,
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

        let result = wrf_core::getvar(&self.inner, name, timeidx, &opts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        to_numpy(py, result)
    }

    fn __repr__(&self) -> String {
        format!(
            "WrfFile('{}', nx={}, ny={}, nz={}, nt={})",
            self.inner.path.display(),
            self.inner.nx,
            self.inner.ny,
            self.inner.nz,
            self.inner.nt,
        )
    }
}

impl WrfFile {
    /// Borrow the inner wrf_core::WrfFile (for use from py_getvar).
    pub fn inner(&self) -> &wrf_core::WrfFile {
        &self.inner
    }
}

/// Convert VarOutput to a numpy array with the correct shape.
pub fn to_numpy(py: Python<'_>, result: wrf_core::VarOutput) -> PyResult<PyObject> {
    let err = |e: ndarray::ShapeError| -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    };
    match result.shape.as_slice() {
        [ny, nx] => {
            let arr = ndarray::Array2::from_shape_vec((*ny, *nx), result.data).map_err(err)?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        [nz, ny, nx] => {
            let arr = ndarray::Array3::from_shape_vec((*nz, *ny, *nx), result.data).map_err(err)?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        [nf, nz, ny, nx] => {
            let arr =
                ndarray::Array4::from_shape_vec((*nf, *nz, *ny, *nx), result.data).map_err(err)?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        _ => {
            let arr = ndarray::Array1::from_vec(result.data);
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
    }
}
