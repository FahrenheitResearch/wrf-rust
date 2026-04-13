use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use wrf_core::StormMotion;

fn flatten_component_grid(
    grid: Vec<Vec<f64>>,
    ny: usize,
    nx: usize,
    label: &str,
) -> PyResult<Vec<f64>> {
    if grid.len() != ny {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{label} has shape ({}, ...), expected ({ny}, {nx})",
            grid.len()
        )));
    }

    let mut flat = Vec::with_capacity(ny * nx);
    for (j, row) in grid.into_iter().enumerate() {
        if row.len() != nx {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{label} row {j} has length {}, expected {nx}",
                row.len()
            )));
        }
        flat.extend(row);
    }

    Ok(flat)
}

fn parse_grid_storm_motion(
    storm_motion: Vec<Vec<Vec<f64>>>,
    ny: usize,
    nx: usize,
) -> PyResult<StormMotion> {
    if storm_motion.len() == 2 {
        let mut components = storm_motion.into_iter();
        let u = flatten_component_grid(components.next().unwrap(), ny, nx, "storm_motion[0]")?;
        let v = flatten_component_grid(components.next().unwrap(), ny, nx, "storm_motion[1]")?;
        return Ok(StormMotion::Grid { u, v });
    }

    Err(PyErr::new::<PyValueError, _>(format!(
        "storm_motion must be a scalar (u, v), a pair of 2D arrays with shape ({ny}, {nx}), \
or a stacked array with shape (2, {ny}, {nx})"
    )))
}

fn parse_storm_motion(
    storm_motion: Option<&Bound<'_, PyAny>>,
    ny: usize,
    nx: usize,
) -> PyResult<Option<StormMotion>> {
    let Some(storm_motion) = storm_motion else {
        return Ok(None);
    };

    if let Ok((u, v)) = storm_motion.extract::<(f64, f64)>() {
        return Ok(Some(StormMotion::Uniform { u, v }));
    }

    if let Ok(grid) = storm_motion.extract::<Vec<Vec<Vec<f64>>>>() {
        return Ok(Some(parse_grid_storm_motion(grid, ny, nx)?));
    }

    Err(PyErr::new::<PyValueError, _>(format!(
        "storm_motion must be a scalar (u, v), a pair of 2D arrays with shape ({ny}, {nx}), \
or a stacked array with shape (2, {ny}, {nx})"
    )))
}

#[allow(clippy::too_many_arguments)]
pub fn build_compute_opts(
    py: Python<'_>,
    ny: usize,
    nx: usize,
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
) -> PyResult<wrf_core::ComputeOpts> {
    let storm_motion = parse_storm_motion(storm_motion.as_ref().map(|obj| obj.bind(py)), ny, nx)?;

    Ok(wrf_core::ComputeOpts {
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
    })
}

#[cfg(test)]
mod tests {
    use super::parse_storm_motion;
    use pyo3::prelude::*;
    use wrf_core::StormMotion;

    #[test]
    fn parses_uniform_storm_motion_tuple() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let storm_motion = (12.0, 8.0).into_pyobject(py).unwrap().into_any();
            let parsed = parse_storm_motion(Some(&storm_motion), 2, 2).unwrap();

            assert_eq!(parsed, Some(StormMotion::Uniform { u: 12.0, v: 8.0 }));
        });
    }

    #[test]
    fn parses_leading_component_axis_grid() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let storm_motion = vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            ]
            .into_pyobject(py)
            .unwrap()
            .into_any();

            let parsed = parse_storm_motion(Some(&storm_motion), 2, 2).unwrap();

            assert_eq!(
                parsed,
                Some(StormMotion::Grid {
                    u: vec![1.0, 2.0, 3.0, 4.0],
                    v: vec![5.0, 6.0, 7.0, 8.0],
                })
            );
        });
    }
}
