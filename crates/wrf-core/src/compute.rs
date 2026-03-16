use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;
use crate::units::{convert_array, parse_units};
use crate::variables::{get_var_def, VarDim};

/// Options controlling variable computation.
#[derive(Debug, Clone, Default)]
pub struct ComputeOpts {
    /// Requested output units (overrides default).
    pub units: Option<String>,
    /// Parcel type for CAPE: "sb", "ml", "mu".
    pub parcel_type: Option<String>,
    /// Custom storm motion (u, v) in m/s for SRH.
    pub storm_motion: Option<(f64, f64)>,
    /// Integration top (meters AGL) for CAPE, lapse rates, updraft helicity, etc.
    pub top_m: Option<f64>,
    /// Bottom of layer (meters AGL) for shear, mean wind, lapse rates, UH, etc.
    pub bottom_m: Option<f64>,
    /// Depth for SRH (meters AGL). Default varies by variable.
    pub depth_m: Option<f64>,
    /// Custom parcel starting pressure (hPa).
    pub parcel_pressure: Option<f64>,
    /// Custom parcel starting temperature (deg C).
    pub parcel_temperature: Option<f64>,
    /// Custom parcel starting dewpoint (deg C).
    pub parcel_dewpoint: Option<f64>,
    /// Layer type: "fixed" (default) or "effective" (for STP, SRH).
    pub layer_type: Option<String>,
    /// Use virtual temperature instead of absolute temperature (for lapse rates).
    pub use_virtual: Option<bool>,
}

/// The result of computing a WRF variable.
#[derive(Debug, Clone)]
pub struct VarOutput {
    /// Flat data array.
    pub data: Vec<f64>,
    /// Shape: `[ny, nx]` for 2-D, `[nz, ny, nx]` for 3-D.
    pub shape: Vec<usize>,
    /// Unit string of the returned data.
    pub units: String,
    /// Human-readable description.
    pub description: String,
}

/// Top-level variable retrieval: look up name, compute, apply unit conversion.
pub fn getvar(
    file: &WrfFile,
    name: &str,
    timeidx: Option<usize>,
    opts: &ComputeOpts,
) -> WrfResult<VarOutput> {
    let vardef = get_var_def(name)
        .ok_or_else(|| WrfError::UnknownVar(name.to_string()))?;

    let t = timeidx.unwrap_or(0);
    if t >= file.nt {
        return Err(WrfError::InvalidParam(format!(
            "timeidx {t} out of range (file has {} times)",
            file.nt
        )));
    }

    let mut data = (vardef.compute)(file, t, opts)?;

    let shape = match vardef.dim {
        VarDim::TwoD => vec![file.ny, file.nx],
        VarDim::ThreeD => vec![file.nz, file.ny, file.nx],
    };

    // Validate output size
    let expected = shape.iter().product::<usize>();
    // Some variables return multi-field (e.g., cape2d returns 4*nxy, uvmet returns 2*nxyz).
    // Only validate when sizes match the declared dim.
    let actual_units;

    if let Some(ref req_units) = opts.units {
        let from = parse_units(vardef.default_units)?;
        let to = parse_units(req_units)?;
        convert_array(&mut data, from, to)?;
        actual_units = req_units.clone();
    } else {
        actual_units = vardef.default_units.to_string();
    }

    // If data length doesn't match the standard shape, adjust shape to reflect
    // the actual output (multi-field variables).
    let final_shape = if data.len() == expected {
        shape
    } else if data.len() % expected == 0 {
        let nfields = data.len() / expected;
        match vardef.dim {
            VarDim::TwoD => vec![nfields, file.ny, file.nx],
            VarDim::ThreeD => vec![nfields, file.nz, file.ny, file.nx],
        }
    } else {
        shape
    };

    Ok(VarOutput {
        data,
        shape: final_shape,
        units: actual_units,
        description: vardef.description.to_string(),
    })
}
