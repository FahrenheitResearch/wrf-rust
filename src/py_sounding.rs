use pyo3::prelude::*;

use crate::py_file::WrfFile;
use wrf_sounding::{extract_and_write_sounding_png, BoxSoundingMethod, SoundingSelection};

#[pyfunction]
#[pyo3(signature = (wrffile, output, lat, lon, timeidx=None))]
fn render_sounding_latlon(
    wrffile: &WrfFile,
    output: &str,
    lat: f64,
    lon: f64,
    timeidx: Option<usize>,
) -> PyResult<()> {
    render_sounding(
        wrffile,
        output,
        SoundingSelection::PointLatLon { lat, lon },
        timeidx,
    )
}

#[pyfunction]
#[pyo3(signature = (wrffile, output, i, j, timeidx=None))]
fn render_sounding_ij(
    wrffile: &WrfFile,
    output: &str,
    i: usize,
    j: usize,
    timeidx: Option<usize>,
) -> PyResult<()> {
    render_sounding(
        wrffile,
        output,
        SoundingSelection::PointIj { i, j },
        timeidx,
    )
}

#[pyfunction]
#[pyo3(signature = (wrffile, output, south, west, north, east, method="mean", timeidx=None))]
fn render_sounding_box(
    wrffile: &WrfFile,
    output: &str,
    south: f64,
    west: f64,
    north: f64,
    east: f64,
    method: &str,
    timeidx: Option<usize>,
) -> PyResult<()> {
    let method = parse_box_method(method)?;
    render_sounding(
        wrffile,
        output,
        SoundingSelection::BoxLatLon {
            south,
            west,
            north,
            east,
            method,
        },
        timeidx,
    )
}

fn render_sounding(
    wrffile: &WrfFile,
    output: &str,
    selection: SoundingSelection,
    timeidx: Option<usize>,
) -> PyResult<()> {
    extract_and_write_sounding_png(wrffile.inner(), selection, timeidx, output)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))
}

fn parse_box_method(method: &str) -> PyResult<BoxSoundingMethod> {
    match method {
        "mean" | "mean_profile" => Ok(BoxSoundingMethod::MeanProfile),
        "median" | "median_profile" => Ok(BoxSoundingMethod::MedianProfile),
        "most_unstable" | "most_unstable_column" | "mu" => {
            Ok(BoxSoundingMethod::MostUnstableColumn)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown box sounding method `{method}`"
        ))),
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_sounding_latlon, m)?)?;
    m.add_function(wrap_pyfunction!(render_sounding_ij, m)?)?;
    m.add_function(wrap_pyfunction!(render_sounding_box, m)?)?;
    Ok(())
}
