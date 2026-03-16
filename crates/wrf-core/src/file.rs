use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Axis;

use crate::error::{WrfError, WrfResult};
use crate::grid;

// ── Physical constants ──
const G: f64 = 9.80665;
const P0: f64 = 100_000.0; // Pa
const KAPPA: f64 = 0.2857142857; // Rd / Cp

/// A handle to a WRF output file with dimension caching and memoized fields.
pub struct WrfFile {
    pub path: PathBuf,
    nc: netcdf::File,
    /// Unstaggered grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nt: usize,
    /// Staggered dimensions.
    pub nx_stag: usize,
    pub ny_stag: usize,
    pub nz_stag: usize,
    /// Grid spacing (m).
    pub dx: f64,
    pub dy: f64,
    /// Memoization cache keyed by `"{field}_{timeidx}"`.
    cache: Mutex<HashMap<String, Vec<f64>>>,
}

// ── NetCDF helpers ──

fn nc_dim_len(file: &netcdf::File, name: &str) -> WrfResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| WrfError::DimMismatch(format!("dimension '{name}' not found")))
}

fn _nc_read_f64(var: &netcdf::Variable) -> WrfResult<Vec<f64>> {
    let arr: ndarray::ArrayD<f64> = var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
    Ok(arr.iter().copied().collect())
}

fn nc_read_f64_time(var: &netcdf::Variable, t: usize) -> WrfResult<Vec<f64>> {
    let ndim = var.dimensions().len();
    // Read full array then slice the requested time index.
    let arr: ndarray::ArrayD<f64> = var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
    if ndim >= 2 {
        let slice = arr.index_axis(Axis(0), t);
        Ok(slice.iter().copied().collect())
    } else {
        Ok(arr.iter().copied().collect())
    }
}

fn nc_get_global_f64(file: &netcdf::File, name: &str) -> WrfResult<f64> {
    let attr = file
        .attribute(name)
        .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
    // netcdf attribute values - try to extract as f64
    let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
    attr_value_to_f64(&val, name)
}

fn nc_get_global_i32(file: &netcdf::File, name: &str) -> WrfResult<i32> {
    let attr = file
        .attribute(name)
        .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
    let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
    attr_value_to_i32(&val, name)
}

fn nc_get_global_str(file: &netcdf::File, name: &str) -> WrfResult<String> {
    let attr = file
        .attribute(name)
        .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
    let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
    match val {
        netcdf::AttributeValue::Str(s) => Ok(s),
        other => Err(WrfError::AttrType(format!("{name}: expected string, got {other:?}"))),
    }
}

fn attr_value_to_f64(val: &netcdf::AttributeValue, name: &str) -> WrfResult<f64> {
    use netcdf::AttributeValue::*;
    match val {
        Double(d) => Ok(*d),
        Float(f) => Ok(*f as f64),
        Int(i) => Ok(*i as f64),
        Short(s) => Ok(*s as f64),
        Ushort(u) => Ok(*u as f64),
        Uint(u) => Ok(*u as f64),
        Uchar(u) => Ok(*u as f64),
        Schar(s) => Ok(*s as f64),
        Longlong(l) => Ok(*l as f64),
        Ulonglong(u) => Ok(*u as f64),
        Doubles(d) if !d.is_empty() => Ok(d[0]),
        Floats(f) if !f.is_empty() => Ok(f[0] as f64),
        Ints(i) if !i.is_empty() => Ok(i[0] as f64),
        _ => Err(WrfError::AttrType(format!("{name}: cannot convert to f64"))),
    }
}

fn attr_value_to_i32(val: &netcdf::AttributeValue, name: &str) -> WrfResult<i32> {
    use netcdf::AttributeValue::*;
    match val {
        Int(i) => Ok(*i),
        Short(s) => Ok(*s as i32),
        Uchar(u) => Ok(*u as i32),
        Schar(s) => Ok(*s as i32),
        Longlong(l) => Ok(*l as i32),
        Float(f) => Ok(*f as i32),
        Double(d) => Ok(*d as i32),
        Ints(i) if !i.is_empty() => Ok(i[0]),
        _ => Err(WrfError::AttrType(format!("{name}: cannot convert to i32"))),
    }
}

impl WrfFile {
    /// Open a WRF output file and read grid dimensions.
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let path = path.as_ref().to_path_buf();
        let nc = netcdf::open(&path)?;

        let nx = nc_dim_len(&nc, "west_east")?;
        let ny = nc_dim_len(&nc, "south_north")?;
        let nz = nc_dim_len(&nc, "bottom_top")?;
        let nt = nc_dim_len(&nc, "Time")?;

        let nx_stag = nc_dim_len(&nc, "west_east_stag").unwrap_or(nx + 1);
        let ny_stag = nc_dim_len(&nc, "south_north_stag").unwrap_or(ny + 1);
        let nz_stag = nc_dim_len(&nc, "bottom_top_stag").unwrap_or(nz + 1);

        let dx = nc_get_global_f64(&nc, "DX").unwrap_or(1000.0);
        let dy = nc_get_global_f64(&nc, "DY").unwrap_or(1000.0);

        Ok(Self {
            path,
            nc,
            nx,
            ny,
            nz,
            nt,
            nx_stag,
            ny_stag,
            nz_stag,
            dx,
            dy,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Number of grid cells in a 2D plane.
    pub fn nxy(&self) -> usize {
        self.nx * self.ny
    }

    /// Number of grid cells in a 3D volume.
    pub fn nxyz(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    // ── Raw variable reading ──

    /// Read a raw variable for a single time step.
    /// Returns data with the Time dimension removed.
    pub fn read_var(&self, name: &str, t: usize) -> WrfResult<Vec<f64>> {
        let var = self
            .nc
            .variable(name)
            .ok_or_else(|| WrfError::VarNotFound(name.to_string()))?;
        nc_read_f64_time(&var, t)
    }

    /// Check if a variable exists in the file.
    pub fn has_var(&self, name: &str) -> bool {
        self.nc.variable(name).is_some()
    }

    /// Read a global attribute as f64.
    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        nc_get_global_f64(&self.nc, name)
    }

    /// Read a global attribute as i32.
    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        nc_get_global_i32(&self.nc, name)
    }

    /// Read a global attribute as String.
    pub fn global_attr_str(&self, name: &str) -> WrfResult<String> {
        nc_get_global_str(&self.nc, name)
    }

    // ── Cached derived fields ──

    fn cached_or_compute(&self, key: &str, f: impl FnOnce() -> WrfResult<Vec<f64>>) -> WrfResult<Vec<f64>> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(v) = cache.get(key) {
                return Ok(v.clone());
            }
        }
        let result = f()?;
        self.cache.lock().unwrap().insert(key.to_string(), result.clone());
        Ok(result)
    }

    /// Full pressure = P + PB (Pa). Shape: `[nz, ny, nx]`.
    pub fn full_pressure(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("pressure_{t}");
        self.cached_or_compute(&key, || {
            let p = self.read_var("P", t)?;
            let pb = self.read_var("PB", t)?;
            Ok(p.iter().zip(pb.iter()).map(|(a, b)| a + b).collect())
        })
    }

    /// Full potential temperature = T + 300 (K). Shape: `[nz, ny, nx]`.
    pub fn full_theta(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("theta_{t}");
        self.cached_or_compute(&key, || {
            let th = self.read_var("T", t)?;
            Ok(th.iter().map(|v| v + 300.0).collect())
        })
    }

    /// Full geopotential = PH + PHB (m^2/s^2), destaggered in Z.
    /// Shape: `[nz, ny, nx]`.
    pub fn full_geopotential(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("geopt_{t}");
        self.cached_or_compute(&key, || {
            let ph = self.read_var("PH", t)?;
            let phb = self.read_var("PHB", t)?;
            let stag: Vec<f64> = ph.iter().zip(phb.iter()).map(|(a, b)| a + b).collect();
            Ok(grid::destagger_z(&stag, self.nz_stag, self.ny, self.nx))
        })
    }

    /// Geopotential on staggered Z levels (not destaggered).
    /// Shape: `[nz_stag, ny, nx]`.
    pub fn geopotential_stag(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("geopt_stag_{t}");
        self.cached_or_compute(&key, || {
            let ph = self.read_var("PH", t)?;
            let phb = self.read_var("PHB", t)?;
            Ok(ph.iter().zip(phb.iter()).map(|(a, b)| a + b).collect())
        })
    }

    /// Temperature = theta * (p / p0)^kappa (K). Shape: `[nz, ny, nx]`.
    pub fn temperature(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("temp_{t}");
        self.cached_or_compute(&key, || {
            let theta = self.full_theta(t)?;
            let pres = self.full_pressure(t)?;
            Ok(theta
                .iter()
                .zip(pres.iter())
                .map(|(th, p)| th * (p / P0).powf(KAPPA))
                .collect())
        })
    }

    /// Height MSL = geopotential / g (m). Shape: `[nz, ny, nx]`.
    pub fn height_msl(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("height_msl_{t}");
        self.cached_or_compute(&key, || {
            let geopt = self.full_geopotential(t)?;
            Ok(geopt.iter().map(|g| g / G).collect())
        })
    }

    /// Terrain height (m). Shape: `[ny, nx]`.
    pub fn terrain(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("terrain_{t}");
        self.cached_or_compute(&key, || self.read_var("HGT", t))
    }

    /// Height AGL = height_msl - terrain (m). Shape: `[nz, ny, nx]`.
    pub fn height_agl(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("height_agl_{t}");
        self.cached_or_compute(&key, || {
            let h = self.height_msl(t)?;
            let ter = self.terrain(t)?;
            let nxy = self.nxy();
            Ok(h.iter()
                .enumerate()
                .map(|(idx, hv)| hv - ter[idx % nxy])
                .collect())
        })
    }

    /// Water vapour mixing ratio (kg/kg). Shape: `[nz, ny, nx]`.
    pub fn qvapor(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("QVAPOR", t)
    }

    /// Surface pressure (Pa). Shape: `[ny, nx]`.
    pub fn psfc(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("PSFC", t)
    }

    /// 2-m temperature (K). Shape: `[ny, nx]`.
    pub fn t2(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("T2", t)
    }

    /// 2-m mixing ratio (kg/kg). Shape: `[ny, nx]`.
    pub fn q2(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("Q2", t)
    }

    /// 10-m U wind (m/s). Shape: `[ny, nx]`.
    pub fn u10(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("U10", t)
    }

    /// 10-m V wind (m/s). Shape: `[ny, nx]`.
    pub fn v10(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("V10", t)
    }

    /// Destaggered U wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn u_destag(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("u_destag_{t}");
        self.cached_or_compute(&key, || {
            let u_stag = self.read_var("U", t)?;
            Ok(grid::destagger_x(&u_stag, self.nz, self.ny, self.nx_stag))
        })
    }

    /// Destaggered V wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn v_destag(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("v_destag_{t}");
        self.cached_or_compute(&key, || {
            let v_stag = self.read_var("V", t)?;
            Ok(grid::destagger_y(&v_stag, self.nz, self.ny_stag, self.nx))
        })
    }

    /// Destaggered W wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn w_destag(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("w_destag_{t}");
        self.cached_or_compute(&key, || {
            let w_stag = self.read_var("W", t)?;
            Ok(grid::destagger_z(&w_stag, self.nz_stag, self.ny, self.nx))
        })
    }

    /// Latitude (degrees). Shape: `[ny, nx]`.
    pub fn xlat(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("XLAT", t)
    }

    /// Longitude (degrees). Shape: `[ny, nx]`.
    pub fn xlong(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("XLONG", t)
    }

    /// Sin of map-rotation angle. Shape: `[ny, nx]`.
    pub fn sinalpha(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("SINALPHA", t)
    }

    /// Cos of map-rotation angle. Shape: `[ny, nx]`.
    pub fn cosalpha(&self, t: usize) -> WrfResult<Vec<f64>> {
        self.read_var("COSALPHA", t)
    }

    /// Pressure in hPa. Shape: `[nz, ny, nx]`.
    pub fn pressure_hpa(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("pressure_hpa_{t}");
        self.cached_or_compute(&key, || {
            let p = self.full_pressure(t)?;
            Ok(p.iter().map(|v| v / 100.0).collect())
        })
    }

    /// Temperature in Celsius. Shape: `[nz, ny, nx]`.
    pub fn temperature_c(&self, t: usize) -> WrfResult<Vec<f64>> {
        let key = format!("temp_c_{t}");
        self.cached_or_compute(&key, || {
            let tk = self.temperature(t)?;
            Ok(tk.iter().map(|v| v - 273.15).collect())
        })
    }

    /// Read WRF Times variable as strings.
    pub fn times(&self) -> WrfResult<Vec<String>> {
        let var = self
            .nc
            .variable("Times")
            .ok_or_else(|| WrfError::VarNotFound("Times".to_string()))?;
        let arr: ndarray::ArrayD<u8> = var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        let shape = arr.shape();
        if shape.len() == 2 {
            let nt = shape[0];
            let slen = shape[1];
            let flat: Vec<u8> = arr.iter().copied().collect();
            let mut times = Vec::with_capacity(nt);
            for i in 0..nt {
                let start = i * slen;
                let end = start + slen;
                let s = String::from_utf8_lossy(&flat[start..end])
                    .trim_end_matches('\0')
                    .to_string();
                times.push(s);
            }
            Ok(times)
        } else {
            Err(WrfError::DimMismatch("Times variable has unexpected shape".into()))
        }
    }

    /// Expose the netcdf file reference for advanced use.
    pub fn nc(&self) -> &netcdf::File {
        &self.nc
    }
}
