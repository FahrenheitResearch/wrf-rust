//! NCAR RIP CAPE/CIN compatibility used by wrf-python 1.3.4.1.
//!
//! This is deliberately separate from [`super::thermo::cape_cin_core`].  The
//! latter is wrf-rust's configurable parcel diagnostic; this module preserves
//! the historical parcel selection, 500 m averaging, pseudoadiabat lookup,
//! crossing insertion, and missing-value conventions of NCAR's `rip_cape.f90`.

use std::sync::OnceLock;

const LOOKUP_SIZE: usize = 150;
const LOOKUP_SCALE: f64 = 100_000.0;
const LOOKUP_MISSING: f64 = 1.0e9;

pub(crate) const G: f64 = crate::WRF_GRAVITY_M_S2;
pub(crate) const RD: f64 = 287.0;
pub(crate) const CP: f64 = 1004.5;
pub(crate) const GAMMA: f64 = RD / CP;

const CELKEL: f64 = 273.15;
const EPS: f64 = 0.622;
const CPMD: f64 = 0.887;
const GAMMAMD: f64 = -0.279;
const EZERO: f64 = 6.112;
const ESLCON1: f64 = 17.67;
const ESLCON2: f64 = 29.65;
const TLCLC1: f64 = 2840.0;
const TLCLC2: f64 = 3.5;
const TLCLC3: f64 = 4.805;
const TLCLC4: f64 = 55.0;
const THTECON1: f64 = 3376.0;
const THTECON2: f64 = 2.54;
const THTECON3: f64 = 0.81;

pub(crate) type RipResult<T> = Result<T, String>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Cape2dColumn {
    pub cape: f64,
    pub cin: f64,
    pub lcl_agl: f64,
    pub lfc_agl: f64,
}

#[derive(Debug, Clone, Copy)]
struct ParcelAverage {
    temperature: f64,
    mixing_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct LiftResult {
    cape: f64,
    cin: f64,
    lcl_zrel: f64,
    lfc_zrel: f64,
    #[cfg(test)]
    lfc_index: usize,
}

#[derive(Default)]
pub(crate) struct CapeWorkspace {
    buoyancy: Vec<f64>,
    relative_height: Vec<f64>,
    accumulated_energy: Vec<f64>,
}

impl CapeWorkspace {
    pub(crate) fn with_levels(levels: usize) -> Self {
        let capacity = 2 * levels + 1;
        Self {
            buoyancy: Vec::with_capacity(capacity),
            relative_height: Vec::with_capacity(capacity),
            accumulated_energy: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.buoyancy.clear();
        self.relative_height.clear();
        self.accumulated_energy.clear();
    }
}

struct PseudoadiabatTable {
    theta_e: Vec<f64>,
    pressure: Vec<f64>,
    temperature: Vec<f64>,
}

static PSEUDOADIABAT_TABLE: OnceLock<RipResult<PseudoadiabatTable>> = OnceLock::new();

fn lookup_table() -> RipResult<&'static PseudoadiabatTable> {
    PSEUDOADIABAT_TABLE
        .get_or_init(decode_lookup_table)
        .as_ref()
        .map_err(|error| error.clone())
}

pub(crate) fn prepare_lookup_table() -> RipResult<()> {
    lookup_table().map(|_| ())
}

fn decode_lookup_table() -> RipResult<PseudoadiabatTable> {
    // Losslessly delta-encoded at 1e-5 K/hPa from NCAR wrf-python tag
    // v1.3.4.1, src/wrf/data/psadilookup.dat (SHA-256
    // 021b6a2de2724f3cd91050d29d3fc4ecb209688ae5a05bc3b2cfb7b09e590ee1).
    // Every decimal value in the formatted Fortran source is exactly
    // representable at this scale; tests pin representative grid/table nodes.
    let encoded = include_str!("psadilookup-v1.3.4.1.delta64");
    let bytes = decode_base64(encoded)?;
    let magic = b"RIPCAPE1";
    if !bytes.starts_with(magic) {
        return Err("vendored pseudoadiabat table has an invalid header".into());
    }

    let mut cursor = VarintCursor::new(&bytes[magic.len()..]);
    let theta_e = decode_delta_series(&mut cursor, LOOKUP_SIZE)?;
    let pressure = decode_delta_series(&mut cursor, LOOKUP_SIZE)?;
    let mut temperature = Vec::with_capacity(LOOKUP_SIZE * LOOKUP_SIZE);

    for _ in 0..LOOKUP_SIZE {
        let mut value = cursor.read_uvar()? as i64;
        temperature.push(value as f64 / LOOKUP_SCALE);

        let mut first_difference = decode_zigzag(cursor.read_uvar()?);
        value = value
            .checked_add(first_difference)
            .ok_or_else(|| "pseudoadiabat first-difference overflow".to_string())?;
        temperature.push(value as f64 / LOOKUP_SCALE);

        let mut second_difference = decode_zigzag(cursor.read_uvar()?);
        first_difference = first_difference
            .checked_add(second_difference)
            .ok_or_else(|| "pseudoadiabat second-difference overflow".to_string())?;
        value = value
            .checked_add(first_difference)
            .ok_or_else(|| "pseudoadiabat value overflow".to_string())?;
        temperature.push(value as f64 / LOOKUP_SCALE);

        for _ in 3..LOOKUP_SIZE {
            let third_difference = decode_zigzag(cursor.read_uvar()?);
            second_difference = second_difference
                .checked_add(third_difference)
                .ok_or_else(|| "pseudoadiabat third-difference overflow".to_string())?;
            first_difference = first_difference
                .checked_add(second_difference)
                .ok_or_else(|| "pseudoadiabat first-difference overflow".to_string())?;
            value = value
                .checked_add(first_difference)
                .ok_or_else(|| "pseudoadiabat value overflow".to_string())?;
            temperature.push(value as f64 / LOOKUP_SCALE);
        }
    }

    if !cursor.is_empty() {
        return Err("vendored pseudoadiabat table has trailing bytes".into());
    }
    if theta_e.len() != LOOKUP_SIZE
        || pressure.len() != LOOKUP_SIZE
        || temperature.len() != LOOKUP_SIZE * LOOKUP_SIZE
    {
        return Err("vendored pseudoadiabat table has an invalid shape".into());
    }

    Ok(PseudoadiabatTable {
        theta_e,
        pressure,
        temperature,
    })
}

fn decode_delta_series(cursor: &mut VarintCursor<'_>, len: usize) -> RipResult<Vec<f64>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut values = Vec::with_capacity(len);
    let mut value = cursor.read_uvar()? as i64;
    values.push(value as f64 / LOOKUP_SCALE);
    for _ in 1..len {
        value = value
            .checked_add(decode_zigzag(cursor.read_uvar()?))
            .ok_or_else(|| "pseudoadiabat grid delta overflow".to_string())?;
        values.push(value as f64 / LOOKUP_SCALE);
    }
    Ok(values)
}

fn decode_base64(encoded: &str) -> RipResult<Vec<u8>> {
    let mut output = Vec::with_capacity(encoded.len() * 3 / 4);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    let mut saw_padding = false;

    for byte in encoded.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            saw_padding = true;
            continue;
        }
        if saw_padding {
            return Err("invalid data after pseudoadiabat base64 padding".into());
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err("invalid character in pseudoadiabat base64".into()),
        } as u32;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((accumulator >> bits) & 0xff) as u8);
            accumulator &= if bits == 0 { 0 } else { (1 << bits) - 1 };
        }
    }

    if accumulator != 0 {
        return Err("non-zero trailing bits in pseudoadiabat base64".into());
    }
    Ok(output)
}

struct VarintCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> VarintCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_uvar(&mut self) -> RipResult<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| "truncated pseudoadiabat varint".to_string())?;
            self.position += 1;
            if shift == 63 && byte > 1 {
                return Err("pseudoadiabat varint overflow".into());
            }
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("pseudoadiabat varint is too long".into())
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

fn decode_zigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn temperature_on_pseudoadiabat(theta_e: f64, pressure_hpa: f64) -> RipResult<f64> {
    let table = lookup_table()?;
    if pressure_hpa <= table.pressure[LOOKUP_SIZE - 1] {
        return Ok(theta_e * (pressure_hpa / 1000.0).powf(GAMMA));
    }

    // These bounds intentionally stop at element 148 (Fortran index 149).
    // The RIP binary search can extrapolate beyond the grid endpoints and
    // never selects the final theta-e or pressure interval.
    let theta_index = rip_binary_search(0, LOOKUP_SIZE - 2, |mid| theta_e >= table.theta_e[mid]);
    let pressure_index = rip_binary_search(0, LOOKUP_SIZE - 2, |mid| {
        pressure_hpa <= table.pressure[mid]
    });

    let frac_theta = (theta_e - table.theta_e[theta_index])
        / (table.theta_e[theta_index + 1] - table.theta_e[theta_index]);
    let frac_pressure = (table.pressure[pressure_index] - pressure_hpa)
        / (table.pressure[pressure_index] - table.pressure[pressure_index + 1]);
    let frac_theta_2 = 1.0 - frac_theta;
    let frac_pressure_2 = 1.0 - frac_pressure;

    let at = |ip: usize, jt: usize| table.temperature[jt * LOOKUP_SIZE + ip];
    let t00 = at(pressure_index, theta_index);
    let t10 = at(pressure_index + 1, theta_index);
    let t01 = at(pressure_index, theta_index + 1);
    let t11 = at(pressure_index + 1, theta_index + 1);
    if [t00, t10, t01, t11]
        .iter()
        .any(|value| *value > LOOKUP_MISSING)
    {
        return Err(format!(
            "pseudoadiabat lookup accessed missing data at {pressure_hpa} hPa, theta-e {theta_e} K"
        ));
    }

    let result = frac_pressure_2 * frac_theta_2 * t00
        + frac_pressure * frac_theta_2 * t10
        + frac_pressure_2 * frac_theta * t01
        + frac_pressure * frac_theta * t11;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!(
            "non-finite pseudoadiabat temperature at {pressure_hpa} hPa, theta-e {theta_e} K"
        ))
    }
}

fn rip_binary_search<F>(mut low: usize, mut high: usize, predicate: F) -> usize
where
    F: Fn(usize) -> bool,
{
    while high - low > 1 {
        let middle = (high + low) / 2;
        if predicate(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn virtual_temperature(temperature: f64, mixing_ratio: f64) -> f64 {
    temperature * (EPS + mixing_ratio) / (EPS * (1.0 + mixing_ratio))
}

fn cape2d_lcl_lapse_rate(mixing_ratio: f64) -> f64 {
    G / (CP * (1.0 + CPMD * mixing_ratio))
}

fn cape3d_lcl_lapse_rate(mixing_ratio: f64) -> f64 {
    G / CP * (1.0 + CPMD * mixing_ratio)
}

fn lcl_temperature(temperature: f64, vapor_pressure: f64) -> f64 {
    TLCLC1 / ((temperature.powf(TLCLC2) / vapor_pressure).ln() - TLCLC3) + TLCLC4
}

fn equivalent_potential_temperature(
    temperature: f64,
    pressure_hpa: f64,
    mixing_ratio: f64,
    lcl_temperature: f64,
) -> f64 {
    temperature
        * (1000.0 / pressure_hpa).powf(GAMMA * (1.0 + GAMMAMD * mixing_ratio))
        * ((THTECON1 / lcl_temperature - THTECON2) * mixing_ratio * (1.0 + THTECON3 * mixing_ratio))
            .exp()
}

fn validate_profile(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
) -> RipResult<()> {
    let levels = pressure_hpa.len();
    if levels < 2 {
        return Err("RIP CAPE requires at least two vertical levels".into());
    }
    if temperature_k.len() != levels || mixing_ratio.len() != levels || height_msl.len() != levels {
        return Err("RIP CAPE profile arrays have different lengths".into());
    }
    for level in 0..levels {
        if !pressure_hpa[level].is_finite()
            || pressure_hpa[level] <= 0.0
            || !temperature_k[level].is_finite()
            || temperature_k[level] <= 0.0
            || !mixing_ratio[level].is_finite()
            || mixing_ratio[level] <= -EPS
            || mixing_ratio[level] >= 1.0
            || !height_msl[level].is_finite()
        {
            return Err(format!("invalid RIP CAPE input at vertical level {level}"));
        }
    }
    if pressure_hpa.windows(2).any(|values| values[0] <= values[1]) {
        return Err("RIP CAPE pressure must decrease from surface to model top".into());
    }
    if height_msl.windows(2).any(|values| values[0] >= values[1]) {
        return Err("RIP CAPE height must increase from surface to model top".into());
    }
    Ok(())
}

fn select_max_theta_e_level(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    terrain_m: f64,
) -> usize {
    let mut selected = 0;
    let mut maximum = -1.0;

    // RIP scans top-to-surface and updates only on a strict increase.  Ties
    // therefore select the uppermost eligible level.
    for level in (0..pressure_hpa.len()).rev() {
        if height_msl[level] - terrain_m < 3000.0 {
            let q = mixing_ratio[level].max(1.0e-15);
            let vapor_pressure = q * pressure_hpa[level] / (EPS + q);
            let tlcl = lcl_temperature(temperature_k[level], vapor_pressure);
            let theta_e = equivalent_potential_temperature(
                temperature_k[level],
                pressure_hpa[level],
                q,
                tlcl,
            );
            if theta_e > maximum {
                selected = level;
                maximum = theta_e;
            }
        }
    }
    selected
}

fn average_500m_parcel(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    surface_pressure_hpa: f64,
    selected: usize,
) -> RipResult<ParcelAverage> {
    let parcel_virtual_temperature =
        virtual_temperature(temperature_k[selected], mixing_ratio[selected]);
    let pressure_depth = 500.0 * pressure_hpa[selected] * G / (RD * parcel_virtual_temperature);
    let lower_pressure = (pressure_hpa[selected] + 0.5 * pressure_depth).min(surface_pressure_hpa);
    let upper_pressure = lower_pressure - pressure_depth;
    if !pressure_depth.is_finite() || pressure_depth <= 0.0 || upper_pressure <= 0.0 {
        return Err("invalid pressure depth for RIP 500 m parcel average".into());
    }

    let mut theta_total = 0.0;
    let mut moisture_total = 0.0;
    let mut integrated_pressure = 0.0;
    for level in 0..pressure_hpa.len() - 1 {
        // pf[s] is the high-pressure (lower) boundary of surface-first layer
        // s. Compute it in place to avoid a per-column allocation.
        let layer_lower = if level == 0 {
            surface_pressure_hpa
        } else {
            0.5 * (pressure_hpa[level - 1] + pressure_hpa[level])
        };
        let layer_upper = 0.5 * (pressure_hpa[level] + pressure_hpa[level + 1]);
        if layer_lower <= upper_pressure {
            break;
        }
        if layer_upper >= lower_pressure {
            continue;
        }
        let q = mixing_ratio[level].max(1.0e-15);
        let theta =
            temperature_k[level] * (1000.0 / pressure_hpa[level]).powf(GAMMA * (1.0 + GAMMAMD * q));
        let overlap_upper = upper_pressure.max(layer_upper);
        let overlap_lower = lower_pressure.min(layer_lower);
        let overlap = overlap_lower - overlap_upper;
        if overlap > 0.0 {
            moisture_total += q * overlap;
            theta_total += theta * overlap;
            integrated_pressure += overlap;
        }
    }

    if !integrated_pressure.is_finite() || integrated_pressure <= 0.0 {
        return Err("RIP 500 m parcel did not intersect a model layer".into());
    }
    let parcel_q = moisture_total / integrated_pressure;
    // This selected-level environmental q in the exponent is an intentional
    // compatibility quirk in rip_cape.f90.
    let parcel_temperature = (theta_total / integrated_pressure)
        * (pressure_hpa[selected] / 1000.0).powf(GAMMA * (1.0 + GAMMAMD * mixing_ratio[selected]));
    if parcel_temperature.is_finite() && parcel_q.is_finite() {
        Ok(ParcelAverage {
            temperature: parcel_temperature,
            mixing_ratio: parcel_q,
        })
    } else {
        Err("RIP 500 m parcel average produced non-finite properties".into())
    }
}

fn append_buoyancy_sample(
    scratch: &mut CapeWorkspace,
    buoyancy: f64,
    relative_height: f64,
) -> usize {
    if let (Some(&previous_buoyancy), Some(&previous_height)) =
        (scratch.buoyancy.last(), scratch.relative_height.last())
    {
        if buoyancy * previous_buoyancy < 0.0 {
            let crossing_height = previous_height
                + previous_buoyancy / (previous_buoyancy - buoyancy)
                    * (relative_height - previous_height);
            scratch.buoyancy.push(0.0);
            scratch.relative_height.push(crossing_height);
        }
    }
    scratch.buoyancy.push(buoyancy);
    scratch.relative_height.push(relative_height);
    scratch.buoyancy.len() - 1
}

fn finish_buoyancy_profile(scratch: &mut CapeWorkspace, lcl_index: usize) -> LiftResult {
    scratch.accumulated_energy.clear();
    scratch.accumulated_energy.push(0.0);
    for level in 1..scratch.buoyancy.len() {
        let dz = scratch.relative_height[level] - scratch.relative_height[level - 1];
        let energy = scratch.accumulated_energy[level - 1]
            + 0.5 * dz * (scratch.buoyancy[level - 1] + scratch.buoyancy[level]);
        scratch.accumulated_energy.push(energy);
    }

    let equilibrium_level = (lcl_index..scratch.buoyancy.len())
        .rev()
        .find(|&level| scratch.buoyancy[level] >= 0.0);
    let Some(equilibrium_level) = equilibrium_level else {
        let top = scratch.buoyancy.len() - 1;
        return LiftResult {
            cape: f64::NAN,
            cin: f64::NAN,
            lcl_zrel: scratch.relative_height[lcl_index],
            lfc_zrel: scratch.relative_height[top],
            #[cfg(test)]
            lfc_index: top,
        };
    };

    let mut minimum_energy = 9.0e9;
    let mut lfc_index = scratch.buoyancy.len() - 1;
    for level in lcl_index..=equilibrium_level {
        if scratch.accumulated_energy[level] < minimum_energy {
            minimum_energy = scratch.accumulated_energy[level];
            lfc_index = level;
        }
    }

    let cape = (scratch.accumulated_energy[equilibrium_level] - minimum_energy).max(0.1);
    let cin = if cape < 100.0 {
        f64::NAN
    } else {
        (-minimum_energy).max(0.1)
    };
    LiftResult {
        cape,
        cin,
        lcl_zrel: scratch.relative_height[lcl_index],
        lfc_zrel: scratch.relative_height[lfc_index],
        #[cfg(test)]
        lfc_index,
    }
}

// Keep the four parallel profile slices explicit: this mirrors the pinned RIP
// kernel and avoids allocating a temporary profile-of-structs per parcel.
#[allow(clippy::too_many_arguments)]
fn lift_parcel(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    start: usize,
    parcel_temperature: f64,
    parcel_mixing_ratio: f64,
    lcl_lapse_rate: f64,
    scratch: &mut CapeWorkspace,
) -> RipResult<LiftResult> {
    scratch.clear();
    let vapor_pressure =
        (parcel_mixing_ratio * pressure_hpa[start] / (EPS + parcel_mixing_ratio)).max(1.0e-20);
    let tlcl = lcl_temperature(parcel_temperature, vapor_pressure);
    let theta_e = equivalent_potential_temperature(
        parcel_temperature,
        pressure_hpa[start],
        parcel_mixing_ratio,
        tlcl,
    );
    let lcl_height = height_msl[start] + (parcel_temperature - tlcl) / lcl_lapse_rate;
    if !tlcl.is_finite() || !theta_e.is_finite() || !lcl_height.is_finite() {
        return Err(format!(
            "invalid RIP parcel thermodynamics at vertical level {start}"
        ));
    }

    let dry_temperature = temperature_k[start];
    let dry_mixing_ratio = mixing_ratio[start];
    let mut lcl_inserted = height_msl[start] >= lcl_height;
    let mut lcl_index = 0usize;

    for level in start..pressure_hpa.len() {
        let (lifted_virtual_temperature, environmental_virtual_temperature, lifted_height, is_lcl) =
            if height_msl[level] < lcl_height {
                let lifted_temperature = dry_temperature
                    - G / (CP * (1.0 + CPMD * dry_mixing_ratio))
                        * (height_msl[level] - height_msl[start]);
                (
                    virtual_temperature(lifted_temperature, dry_mixing_ratio),
                    virtual_temperature(temperature_k[level], mixing_ratio[level]),
                    height_msl[level],
                    false,
                )
            } else if !lcl_inserted {
                let below = level
                    .checked_sub(1)
                    .ok_or_else(|| "RIP LCL interpolation has no lower model level".to_string())?;
                let denominator = height_msl[level] - height_msl[below];
                let lower_weight = (height_msl[level] - lcl_height) / denominator;
                let upper_weight = (lcl_height - height_msl[below]) / denominator;
                let environmental_temperature =
                    temperature_k[below] * lower_weight + temperature_k[level] * upper_weight;
                let environmental_q =
                    mixing_ratio[below] * lower_weight + mixing_ratio[level] * upper_weight;
                (
                    virtual_temperature(tlcl, dry_mixing_ratio),
                    virtual_temperature(environmental_temperature, environmental_q),
                    lcl_height,
                    true,
                )
            } else {
                let lifted_temperature =
                    temperature_on_pseudoadiabat(theta_e, pressure_hpa[level])?;
                let saturation_vapor_pressure = EZERO
                    * (ESLCON1 * (lifted_temperature - CELKEL) / (lifted_temperature - ESLCON2))
                        .exp();
                let lifted_q = EPS * saturation_vapor_pressure
                    / (pressure_hpa[level] - saturation_vapor_pressure);
                (
                    virtual_temperature(lifted_temperature, lifted_q),
                    virtual_temperature(temperature_k[level], mixing_ratio[level]),
                    height_msl[level],
                    false,
                )
            };

        if !lifted_virtual_temperature.is_finite()
            || !environmental_virtual_temperature.is_finite()
            || environmental_virtual_temperature == 0.0
        {
            return Err(format!("non-finite RIP buoyancy at vertical level {level}"));
        }
        let buoyancy = G * (lifted_virtual_temperature - environmental_virtual_temperature)
            / environmental_virtual_temperature;
        let current_index =
            append_buoyancy_sample(scratch, buoyancy, lifted_height - height_msl[start]);
        if is_lcl {
            lcl_index = current_index;
            lcl_inserted = true;
        }
    }

    if scratch.buoyancy.is_empty() {
        return Err("RIP parcel lift produced no buoyancy samples".into());
    }
    if !lcl_inserted {
        lcl_index = scratch.buoyancy.len() - 1;
    }
    Ok(finish_buoyancy_profile(scratch, lcl_index))
}

/// Strict wrf-python/RIP two-dimensional maximum-parcel CAPE for one column.
///
/// Inputs are surface-first; pressure is hPa, temperature is K, mixing ratio
/// is kg/kg, and heights are m MSL. Missing outputs use NaN.
#[cfg(test)]
pub(crate) fn cape2d_column(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    terrain_m: f64,
    surface_pressure_hpa: f64,
) -> RipResult<Cape2dColumn> {
    let mut workspace = CapeWorkspace::with_levels(pressure_hpa.len());
    cape2d_column_with_workspace(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        height_msl,
        terrain_m,
        surface_pressure_hpa,
        &mut workspace,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cape2d_column_with_workspace(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    terrain_m: f64,
    surface_pressure_hpa: f64,
    workspace: &mut CapeWorkspace,
) -> RipResult<Cape2dColumn> {
    validate_profile(pressure_hpa, temperature_k, mixing_ratio, height_msl)?;
    if !terrain_m.is_finite() || !surface_pressure_hpa.is_finite() || surface_pressure_hpa <= 0.0 {
        return Err("invalid terrain or surface pressure for RIP CAPE".into());
    }

    let selected = select_max_theta_e_level(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        height_msl,
        terrain_m,
    );
    let parcel = average_500m_parcel(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        surface_pressure_hpa,
        selected,
    )?;
    let lifted = lift_parcel(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        height_msl,
        selected,
        parcel.temperature,
        parcel.mixing_ratio,
        // CAPECALC2D uses cpm = CP * (1 + CPMD*q) and G/cpm.
        cape2d_lcl_lapse_rate(parcel.mixing_ratio),
        workspace,
    )?;
    Ok(Cape2dColumn {
        cape: lifted.cape,
        cin: lifted.cin,
        lcl_agl: lifted.lcl_zrel + height_msl[selected] - terrain_m,
        lfc_agl: lifted.lfc_zrel + height_msl[selected] - terrain_m,
    })
}

/// Strict wrf-python/RIP three-dimensional levelwise CAPE/CIN for one column.
///
/// The returned arrays are surface-first. As in `DCAPECALC3D`, the topmost
/// model level is exactly zero for both CAPE and CIN.
#[cfg(test)]
pub(crate) fn cape3d_column(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
) -> RipResult<(Vec<f64>, Vec<f64>)> {
    let mut workspace = CapeWorkspace::with_levels(pressure_hpa.len());
    cape3d_column_with_workspace(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        height_msl,
        &mut workspace,
    )
}

#[cfg(test)]
pub(crate) fn cape3d_column_with_workspace(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    workspace: &mut CapeWorkspace,
) -> RipResult<(Vec<f64>, Vec<f64>)> {
    let levels = pressure_hpa.len();
    let mut cape = vec![0.0; levels];
    let mut cin = vec![0.0; levels];
    cape3d_column_into(
        pressure_hpa,
        temperature_k,
        mixing_ratio,
        height_msl,
        &mut cape,
        &mut cin,
        workspace,
    )?;
    Ok((cape, cin))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cape3d_column_into(
    pressure_hpa: &[f64],
    temperature_k: &[f64],
    mixing_ratio: &[f64],
    height_msl: &[f64],
    cape: &mut [f64],
    cin: &mut [f64],
    workspace: &mut CapeWorkspace,
) -> RipResult<()> {
    validate_profile(pressure_hpa, temperature_k, mixing_ratio, height_msl)?;
    let levels = pressure_hpa.len();
    if cape.len() != levels || cin.len() != levels {
        return Err("RIP CAPE3D output arrays have the wrong length".into());
    }
    cape.fill(0.0);
    cin.fill(0.0);
    for start in 0..levels - 1 {
        let lifted = lift_parcel(
            pressure_hpa,
            temperature_k,
            mixing_ratio,
            height_msl,
            start,
            temperature_k[start],
            mixing_ratio[start],
            // Preserve CAPECALC3D's distinct parenthesization:
            // (G / CP) * (1 + CPMD*q), rather than CAPECALC2D's G/cpm.
            cape3d_lcl_lapse_rate(mixing_ratio[start]),
            workspace,
        )?;
        cape[start] = lifted.cape;
        cin[start] = lifted.cin;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    fn saturated_theta_e_at_1000_hpa(theta_w_c: f64) -> f64 {
        let temperature = theta_w_c + CELKEL;
        let vapor_pressure =
            EZERO * (ESLCON1 * (temperature - CELKEL) / (temperature - ESLCON2)).exp();
        let mixing_ratio = EPS * vapor_pressure / (1000.0 - vapor_pressure);
        // A parcel saturated at 1000 hPa is already at its LCL.
        equivalent_potential_temperature(temperature, 1000.0, mixing_ratio, temperature)
    }

    #[test]
    fn wobus_satlift_matches_documented_ncar_table_accuracy() {
        // theta-w (C), pressure (hPa), Wobus temperature (K), NCAR table (K).
        // The 700-, 300-, and 190-hPa rows are the largest differences found
        // in those pressure bands on the representative -40..40 C by 10-hPa grid.
        let cases = [
            (-20.0, 850.0, 242.756_654_241_680, 242.711_632_398_885),
            (0.0, 700.0, 252.588_141_714_542, 253.078_140_155_707),
            (20.0, 500.0, 264.678_529_908_059, 264.366_496_623_889),
            (16.0, 300.0, 227.977_774_363_201, 227.300_805_020_826),
            (40.0, 190.0, 266.643_743_001_737, 265.463_497_199_297),
        ];

        for (theta_w_c, pressure_hpa, expected_wobus, expected_ncar) in cases {
            let theta_e = saturated_theta_e_at_1000_hpa(theta_w_c);
            let ncar = temperature_on_pseudoadiabat(theta_e, pressure_hpa).unwrap();
            let wobus = crate::met::thermo::satlift(pressure_hpa, theta_w_c) + CELKEL;

            assert_close(wobus, expected_wobus, 1.0e-9);
            assert_close(ncar, expected_ncar, 1.0e-9);
            assert!((wobus - ncar).abs() <= 1.2);
        }
    }

    #[test]
    fn vendored_lookup_decodes_exact_pinned_grid_and_nodes() {
        let table = lookup_table().unwrap();
        assert_eq!(table.theta_e[49], 300.2493);
        assert_eq!(table.pressure[35], 505.0036);
        assert_eq!(table.temperature[49 * LOOKUP_SIZE + 35], 244.9291);
        assert_eq!(table.temperature.len(), 22_500);
    }

    #[test]
    fn pseudoadiabat_matches_pinned_bilinear_interpolation() {
        let temperature = temperature_on_pseudoadiabat(301.4993, 499.4492).unwrap();
        assert_close(temperature, 245.10595, 1.0e-10);
    }

    #[test]
    fn unreasonable_lookup_request_returns_error_instead_of_panicking() {
        let error = temperature_on_pseudoadiabat(682.3188, 1100.0).unwrap_err();
        assert!(error.contains("missing data"));
    }

    #[test]
    fn cape3d_preserves_distinct_rip_lcl_parenthesization() {
        let q = 0.02;
        assert_close(
            cape2d_lcl_lapse_rate(q),
            G / (CP * (1.0 + CPMD * q)),
            1.0e-15,
        );
        assert_close(cape3d_lcl_lapse_rate(q), G / CP * (1.0 + CPMD * q), 1.0e-15);
        assert_ne!(cape2d_lcl_lapse_rate(q), cape3d_lcl_lapse_rate(q));
    }

    #[test]
    fn max_theta_e_tie_selects_upper_level_and_excludes_exactly_three_km() {
        let pressure = [1000.0, 1000.0, 1000.0];
        let temperature = [300.0, 300.0, 310.0];
        let mixing_ratio = [0.012, 0.012, 0.020];
        let height = [0.0, 1000.0, 3000.0];
        assert_eq!(
            select_max_theta_e_level(&pressure, &temperature, &mixing_ratio, &height, 0.0,),
            1
        );
    }

    #[test]
    fn malformed_profile_returns_an_error_without_indexing_past_inputs() {
        let pressure = [900.0, 1000.0];
        let temperature = [290.0, 300.0];
        let mixing_ratio = [0.008, 0.012];
        let height = [1000.0, 100.0];
        let error = cape3d_column(&pressure, &temperature, &mixing_ratio, &height).unwrap_err();
        assert!(error.contains("pressure must decrease"));
    }

    #[test]
    fn parcel_average_uses_pressure_layer_intersections() {
        let pressure = [980.0, 930.0, 800.0];
        let temperature = [306.0, 294.0, 278.0];
        let mixing_ratio = [0.016, 0.006, 0.002];
        let parcel =
            average_500m_parcel(&pressure, &temperature, &mixing_ratio, 1000.0, 1).unwrap();

        let depth =
            500.0 * pressure[1] * G / (RD * virtual_temperature(temperature[1], mixing_ratio[1]));
        let p2 = (pressure[1] + 0.5 * depth).min(1000.0);
        let p1 = p2 - depth;
        let boundary = 0.5 * (pressure[0] + pressure[1]);
        let lower_overlap = (p2 - boundary).max(0.0);
        let selected_overlap = boundary.min(p2) - p1.max(0.5 * (pressure[1] + pressure[2]));
        let expected_q = (mixing_ratio[0] * lower_overlap + mixing_ratio[1] * selected_overlap)
            / (lower_overlap + selected_overlap);
        assert_close(parcel.mixing_ratio, expected_q, 1.0e-12);
        assert!((parcel.mixing_ratio - mixing_ratio[1]).abs() > 1.0e-5);
    }

    #[test]
    fn energy_reduction_uses_highest_el_and_first_strict_minimum() {
        let mut scratch = CapeWorkspace {
            buoyancy: vec![-1.0, -1.0, 0.0, 2.0, 0.0, -2.0, 0.0, 1.0, 0.0, -1.0],
            relative_height: (0..10).map(|level| level as f64 * 100.0).collect(),
            ..Default::default()
        };
        let result = finish_buoyancy_profile(&mut scratch, 0);
        assert_close(result.cape, 100.0, 1.0e-12);
        assert_close(result.cin, 150.0, 1.0e-12);
        assert_eq!(result.lfc_index, 2);
    }

    #[test]
    fn no_equilibrium_level_is_missing_and_uses_top_as_lfc() {
        let mut scratch = CapeWorkspace {
            buoyancy: vec![-1.0, -0.5, -0.25],
            relative_height: vec![0.0, 500.0, 1000.0],
            ..Default::default()
        };
        let result = finish_buoyancy_profile(&mut scratch, 0);
        assert!(result.cape.is_nan());
        assert!(result.cin.is_nan());
        assert_eq!(result.lfc_zrel, 1000.0);
    }

    #[test]
    fn cin_is_missing_when_cape_is_below_one_hundred() {
        let mut scratch = CapeWorkspace {
            buoyancy: vec![-0.2, 0.0, 0.8, 0.0, -0.2],
            relative_height: vec![0.0, 100.0, 200.0, 300.0, 400.0],
            ..Default::default()
        };
        let result = finish_buoyancy_profile(&mut scratch, 0);
        assert!(result.cape < 100.0);
        assert!(result.cin.is_nan());
    }

    #[test]
    fn cape2d_returns_rip_ordered_height_components_for_a_valid_column() {
        let pressure = [1000.0, 900.0, 800.0, 700.0];
        let temperature = [300.0, 294.0, 287.0, 279.0];
        let mixing_ratio = [0.014, 0.010, 0.006, 0.003];
        let height = [100.0, 1000.0, 2000.0, 3100.0];
        let result = cape2d_column(
            &pressure,
            &temperature,
            &mixing_ratio,
            &height,
            50.0,
            1010.0,
        )
        .unwrap();
        assert!(result.lcl_agl.is_finite());
        assert!(result.lfc_agl.is_finite());
        assert!(result.lcl_agl >= 0.0);
        assert!(result.lfc_agl >= result.lcl_agl);
    }

    #[test]
    fn cape3d_preserves_rip_top_level_zero_convention() {
        let pressure = [1000.0, 900.0, 800.0, 700.0];
        let temperature = [300.0, 294.0, 287.0, 279.0];
        let mixing_ratio = [0.014, 0.010, 0.006, 0.003];
        let height = [100.0, 1000.0, 2000.0, 3100.0];
        let (cape, cin) = cape3d_column(&pressure, &temperature, &mixing_ratio, &height).unwrap();
        assert_eq!(cape.len(), pressure.len());
        assert_eq!(cin.len(), pressure.len());
        assert_eq!(cape[pressure.len() - 1], 0.0);
        assert_eq!(cin[pressure.len() - 1], 0.0);
    }
}
