//! Wind calculations for vertical profiles.
//!
//! Profile-based calculations (shear, helicity, storm motion)
//! that operate on sounding data rather than 2-D grids.
//!
//! Vendored from metrust crate for self-contained builds.

use std::f64::consts::PI;

// Helpers

/// Linearly interpolate a value at `target_h` from a height-sorted profile.
///
/// Returns `None` if `target_h` is outside the profile range.
fn interp_at_height(profile: &[f64], heights: &[f64], target_h: f64) -> Option<f64> {
    debug_assert_eq!(profile.len(), heights.len());
    let n = heights.len();
    if n == 0 {
        return None;
    }
    if target_h <= heights[0] {
        return Some(profile[0]);
    }
    if target_h >= heights[n - 1] {
        return Some(profile[n - 1]);
    }
    for i in 1..n {
        if heights[i] >= target_h {
            let frac = (target_h - heights[i - 1]) / (heights[i] - heights[i - 1]);
            return Some(profile[i - 1] + frac * (profile[i] - profile[i - 1]));
        }
    }
    None
}

// Profile-based functions

/// Bulk wind shear between two height levels.
///
/// Computes (delta-u, delta-v) = wind(top) - wind(bottom), interpolating to
/// exact height levels when they fall between profile points.
///
/// # Arguments
/// * `u_prof` -- u-component profile (m/s), ordered by ascending height
/// * `v_prof` -- v-component profile (m/s)
/// * `height_prof` -- height AGL profile (m), must be monotonically increasing
/// * `bottom_m` -- bottom of the shear layer (m AGL)
/// * `top_m` -- top of the shear layer (m AGL)
///
/// # Panics
/// Panics if slices have mismatched lengths or fewer than 2 levels.
pub fn bulk_shear(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    assert_eq!(u_prof.len(), v_prof.len());
    assert_eq!(u_prof.len(), height_prof.len());
    assert!(u_prof.len() >= 2, "need at least 2 levels");

    let u_bot = interp_at_height(u_prof, height_prof, bottom_m).unwrap();
    let v_bot = interp_at_height(v_prof, height_prof, bottom_m).unwrap();
    let u_top = interp_at_height(u_prof, height_prof, top_m).unwrap();
    let v_top = interp_at_height(v_prof, height_prof, top_m).unwrap();

    (u_top - u_bot, v_top - v_bot)
}

/// Storm-relative helicity integrated from the surface to `depth_m` AGL.
///
/// SRH = integral of (storm-relative wind cross vertical wind shear) dz.
/// Uses the trapezoidal approximation on the profile segments within the layer.
///
/// # Returns
/// `(positive_srh, negative_srh, total_srh)` where:
/// - `positive_srh` >= 0 -- sum of positive (cyclonic in NH) contributions
/// - `negative_srh` <= 0 -- sum of negative (anticyclonic) contributions
/// - `total_srh` = positive_srh + negative_srh
///
/// # Arguments
/// * `u_prof`, `v_prof` -- wind components (m/s), ascending height
/// * `height_prof` -- heights AGL (m)
/// * `depth_m` -- integration depth from surface (m), e.g. 1000.0 or 3000.0
/// * `storm_u`, `storm_v` -- storm motion components (m/s)
///
/// # Panics
/// Panics on mismatched lengths or fewer than 2 levels.
pub fn storm_relative_helicity(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    depth_m: f64,
    storm_u: f64,
    storm_v: f64,
) -> (f64, f64, f64) {
    let n = u_prof.len();
    assert_eq!(n, v_prof.len());
    assert_eq!(n, height_prof.len());
    assert!(n >= 2, "need at least 2 levels");

    // Build a sub-profile from surface to depth_m AGL, interpolating endpoints if needed.
    // depth_m is an absolute height AGL (e.g. 1000 = 1000m AGL), NOT a relative
    // depth from the first profile level. This avoids the 10m offset bug when
    // 10m winds are prepended at height=10.
    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();

    let h_start = height_prof[0];
    let h_end = depth_m;

    for i in 0..n {
        if height_prof[i] >= h_start && height_prof[i] <= h_end {
            heights.push(height_prof[i]);
            us.push(u_prof[i]);
            vs.push(v_prof[i]);
        }
    }

    // If the top of the layer doesn't exactly match a profile level, interpolate.
    if let Some(&last_h) = heights.last() {
        if last_h < h_end {
            if let (Some(u_top), Some(v_top)) = (
                interp_at_height(u_prof, height_prof, h_end),
                interp_at_height(v_prof, height_prof, h_end),
            ) {
                heights.push(h_end);
                us.push(u_top);
                vs.push(v_top);
            }
        }
    }

    let m = heights.len();
    if m < 2 {
        return (0.0, 0.0, 0.0);
    }

    let mut pos_srh = 0.0;
    let mut neg_srh = 0.0;

    // SRH via trapezoidal integration of the cross-product:
    // SRH = sum_i [ (sru[i+1]*srv[i]) - (sru[i]*srv[i+1]) ]
    for i in 0..(m - 1) {
        let sru_i = us[i] - storm_u;
        let srv_i = vs[i] - storm_v;
        let sru_ip1 = us[i + 1] - storm_u;
        let srv_ip1 = vs[i + 1] - storm_v;

        // Cross-product contribution (2-D "area" form)
        // Convention: positive for clockwise-turning (veering) hodographs in NH
        // Matches MetPy/SHARPpy sign convention
        let contrib = (sru_ip1 * srv_i) - (sru_i * srv_ip1);

        if contrib > 0.0 {
            pos_srh += contrib;
        } else {
            neg_srh += contrib;
        }
    }

    (pos_srh, neg_srh, pos_srh + neg_srh)
}

/// Mean wind over a height layer, computed as a height-weighted (trapezoidal)
/// average of the wind components.
///
/// # Arguments
/// * `u_prof`, `v_prof` -- wind components (m/s), ascending height
/// * `height_prof` -- heights AGL (m)
/// * `bottom_m` -- bottom of the layer (m AGL)
/// * `top_m` -- top of the layer (m AGL)
///
/// # Returns
/// `(mean_u, mean_v)` in m/s.
///
/// # Panics
/// Panics on mismatched lengths or fewer than 2 levels.
pub fn mean_wind(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    let n = u_prof.len();
    assert_eq!(n, v_prof.len());
    assert_eq!(n, height_prof.len());
    assert!(n >= 2, "need at least 2 levels");

    // Build sub-profile over the layer, interpolating endpoints.
    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();

    // Interpolate at bottom
    let u_bot = interp_at_height(u_prof, height_prof, bottom_m).unwrap();
    let v_bot = interp_at_height(v_prof, height_prof, bottom_m).unwrap();
    heights.push(bottom_m);
    us.push(u_bot);
    vs.push(v_bot);

    // Add interior points within the layer
    for i in 0..n {
        if height_prof[i] > bottom_m && height_prof[i] < top_m {
            heights.push(height_prof[i]);
            us.push(u_prof[i]);
            vs.push(v_prof[i]);
        }
    }

    // Interpolate at top
    let u_top = interp_at_height(u_prof, height_prof, top_m).unwrap();
    let v_top = interp_at_height(v_prof, height_prof, top_m).unwrap();
    heights.push(top_m);
    us.push(u_top);
    vs.push(v_top);

    // Trapezoidal integration
    let m = heights.len();
    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut total_dz = 0.0;

    for i in 0..(m - 1) {
        let dz = heights[i + 1] - heights[i];
        sum_u += 0.5 * (us[i] + us[i + 1]) * dz;
        sum_v += 0.5 * (vs[i] + vs[i + 1]) * dz;
        total_dz += dz;
    }

    if total_dz.abs() < 1e-10 {
        return (u_bot, v_bot);
    }

    (sum_u / total_dz, sum_v / total_dz)
}

/// Native-level arithmetic mean wind over a height layer.
///
/// This legacy, pressure-less helper averages the supplied native profile
/// levels plus interpolated height endpoints. It is retained for callers that
/// do not have pressure data, but it is **not** SHARPpy's `mean_wind_npw`:
/// SHARPpy first resamples the profile every 1 hPa. Use
/// [`mean_wind_npw_pressure_resampled`] when pressure is available.
///
/// # Arguments
/// * `u_prof`, `v_prof` -- wind components (m/s), ascending height
/// * `height_prof` -- heights AGL (m)
/// * `bottom_m` -- bottom of the layer (m AGL)
/// * `top_m` -- top of the layer (m AGL)
///
/// # Returns
/// `(mean_u, mean_v)` in m/s.
pub fn mean_wind_npw(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    let n = u_prof.len();
    assert_eq!(n, v_prof.len());
    assert_eq!(n, height_prof.len());
    if n == 0 {
        return (0.0, 0.0);
    }

    // Include interpolated layer endpoints plus any interior profile levels.
    // Matches Solarpower07 / SHARPpy convention.
    let u_bot = interp_at_height(u_prof, height_prof, bottom_m).unwrap_or(u_prof[0]);
    let v_bot = interp_at_height(v_prof, height_prof, bottom_m).unwrap_or(v_prof[0]);
    let u_top = interp_at_height(u_prof, height_prof, top_m).unwrap_or(u_prof[n - 1]);
    let v_top = interp_at_height(v_prof, height_prof, top_m).unwrap_or(v_prof[n - 1]);

    let mut sum_u = u_bot + u_top;
    let mut sum_v = v_bot + v_top;
    let mut count = 2usize;

    for i in 0..n {
        if height_prof[i] > bottom_m && height_prof[i] < top_m {
            sum_u += u_prof[i];
            sum_v += v_prof[i];
            count += 1;
        }
        if height_prof[i] >= top_m {
            break;
        }
    }

    if count == 0 {
        return mean_wind(u_prof, v_prof, height_prof, bottom_m, top_m);
    }

    (sum_u / count as f64, sum_v / count as f64)
}

/// SHARPpy 1.4 non-pressure-weighted mean wind over a height layer.
///
/// The layer endpoints are converted to pressure with log-pressure height
/// interpolation. Wind is then log-pressure interpolated at the exact
/// `np.arange(pbot, ptop - 1, -1)` sequence used by SHARPpy's
/// `winds.mean_wind_npw` default (`dp=-1 hPa`) and arithmetically averaged.
/// This makes the result independent of the WRF eta-level density.
///
/// Falls back to [`mean_wind_npw`] for a non-monotonic or otherwise unusable
/// pressure profile. The fallback is deliberately bounded so malformed input
/// cannot trigger an unreasonably long pressure-sampling loop.
pub fn mean_wind_npw_pressure_resampled(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    pressure_hpa: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    const MAX_PRESSURE_SAMPLES: usize = 2_500;

    let n = u_prof.len();
    if n < 2
        || v_prof.len() != n
        || height_prof.len() != n
        || pressure_hpa.len() != n
        || !bottom_m.is_finite()
        || !top_m.is_finite()
        || bottom_m > top_m
    {
        return (0.0, 0.0);
    }

    let usable = (0..n).all(|i| {
        u_prof[i].is_finite()
            && v_prof[i].is_finite()
            && height_prof[i].is_finite()
            && pressure_hpa[i].is_finite()
            && pressure_hpa[i] > 0.0
            && (i == 0
                || (height_prof[i] > height_prof[i - 1] && pressure_hpa[i] < pressure_hpa[i - 1]))
    });
    if !usable {
        return mean_wind_npw(u_prof, v_prof, height_prof, bottom_m, top_m);
    }

    let pressure_at_height = |target_h: f64| {
        if target_h <= height_prof[0] {
            return pressure_hpa[0];
        }
        if target_h >= height_prof[n - 1] {
            return pressure_hpa[n - 1];
        }
        for i in 1..n {
            if height_prof[i] >= target_h {
                let fraction =
                    (target_h - height_prof[i - 1]) / (height_prof[i] - height_prof[i - 1]);
                let log_p = pressure_hpa[i - 1].ln()
                    + fraction * (pressure_hpa[i].ln() - pressure_hpa[i - 1].ln());
                return log_p.exp();
            }
        }
        pressure_hpa[n - 1]
    };

    let p_bottom = pressure_at_height(bottom_m);
    let p_top = pressure_at_height(top_m);
    if !p_bottom.is_finite() || !p_top.is_finite() || p_bottom <= p_top {
        return mean_wind_npw(u_prof, v_prof, height_prof, bottom_m, top_m);
    }

    // Equivalent to len(np.arange(p_bottom, p_top - 1.0, -1.0)).
    let sample_count = (p_bottom - p_top + 1.0).ceil() as usize;
    let last_sample_pressure = p_bottom - sample_count.saturating_sub(1) as f64;
    if sample_count == 0 || sample_count > MAX_PRESSURE_SAMPLES || last_sample_pressure <= 0.0 {
        return mean_wind_npw(u_prof, v_prof, height_prof, bottom_m, top_m);
    }

    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut segment = 0usize;
    for sample in 0..sample_count {
        let p = p_bottom - sample as f64;
        while segment + 1 < n - 1 && p < pressure_hpa[segment + 1] {
            segment += 1;
        }

        let p0 = pressure_hpa[segment];
        let p1 = pressure_hpa[segment + 1];
        let fraction = (p.ln() - p0.ln()) / (p1.ln() - p0.ln());
        sum_u += u_prof[segment] + fraction * (u_prof[segment + 1] - u_prof[segment]);
        sum_v += v_prof[segment] + fraction * (v_prof[segment + 1] - v_prof[segment]);
    }

    (sum_u / sample_count as f64, sum_v / sample_count as f64)
}

/// Bunkers storm motion estimate using the internal dynamics (ID) method.
///
/// Computes right-moving, left-moving, and mean-wind vectors for a supercell
/// storm motion estimate. Uses the 0-6 km non-pressure-weighted mean wind and
/// the canonical low/high-layer shear vector between the 0-0.5 km and 5.5-6 km
/// mean winds.
///
/// # Arguments
/// * `u_prof`, `v_prof` -- wind components (m/s), ascending height
/// * `height_prof` -- heights AGL (m)
///
/// # Returns
/// `(right_mover, left_mover, mean_wind)` where each is `(u, v)` in m/s.
///
/// # References
/// Bunkers et al. (2000): Predicting Supercell Motion Using a New Hodograph
/// Technique. *Wea. Forecasting*, **15**, 61-79.
pub fn bunkers_storm_motion(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let deviation = 7.5; // m/s perpendicular offset

    let (mw_u, mw_v) = mean_wind_npw(u_prof, v_prof, height_prof, 0.0, 6000.0);
    let (low_u, low_v) = mean_wind_npw(u_prof, v_prof, height_prof, 0.0, 500.0);
    let (high_u, high_v) = mean_wind_npw(u_prof, v_prof, height_prof, 5500.0, 6000.0);

    let shr_u = high_u - low_u;
    let shr_v = high_v - low_v;

    // Normalize the shear vector, then get the perpendicular
    let shear_mag = (shr_u * shr_u + shr_v * shr_v).sqrt();

    if shear_mag < 1e-10 {
        // Degenerate case: no shear => storm motion is the mean wind
        return ((mw_u, mw_v), (mw_u, mw_v), (mw_u, mw_v));
    }

    // Unit shear vector
    let shr_u_hat = shr_u / shear_mag;
    let shr_v_hat = shr_v / shear_mag;

    // Perpendicular (90 deg clockwise rotation for right mover)
    let perp_u = shr_v_hat;
    let perp_v = -shr_u_hat;

    let right_u = mw_u + deviation * perp_u;
    let right_v = mw_v + deviation * perp_v;

    let left_u = mw_u - deviation * perp_u;
    let left_v = mw_v - deviation * perp_v;

    ((right_u, right_v), (left_u, left_v), (mw_u, mw_v))
}

/// Bunkers layer-mean method using SHARPpy 1.4's pressure-resampled NPW means.
///
/// This is the pressure-aware counterpart to [`bunkers_storm_motion`]. It
/// preserves the same 0--500 m and 5.5--6 km layer-mean shear construction,
/// while removing dependence on the placement of native model levels.
pub fn bunkers_storm_motion_npw_pressure_resampled(
    u_prof: &[f64],
    v_prof: &[f64],
    height_prof: &[f64],
    pressure_hpa: &[f64],
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let deviation = 7.5;

    let (mw_u, mw_v) =
        mean_wind_npw_pressure_resampled(u_prof, v_prof, height_prof, pressure_hpa, 0.0, 6000.0);
    let (low_u, low_v) =
        mean_wind_npw_pressure_resampled(u_prof, v_prof, height_prof, pressure_hpa, 0.0, 500.0);
    let (high_u, high_v) =
        mean_wind_npw_pressure_resampled(u_prof, v_prof, height_prof, pressure_hpa, 5500.0, 6000.0);

    let shear_u = high_u - low_u;
    let shear_v = high_v - low_v;
    let shear_mag = shear_u.hypot(shear_v);
    if shear_mag < 1.0e-10 {
        return ((mw_u, mw_v), (mw_u, mw_v), (mw_u, mw_v));
    }

    let dev_u = deviation * shear_v / shear_mag;
    let dev_v = -deviation * shear_u / shear_mag;
    (
        (mw_u + dev_u, mw_v + dev_v),
        (mw_u - dev_u, mw_v - dev_v),
        (mw_u, mw_v),
    )
}

/// Select the cyclonically favored Bunkers mover for a latitude.
///
/// Right movers are cyclonic in the Northern Hemisphere; the mirrored left
/// mover is cyclonic in the Southern Hemisphere. The returned motion does not
/// alter SRH's mathematical sign: cyclonic SRH remains negative in a mirrored
/// Southern Hemisphere profile, matching wrf-python's latitude-aware raw SRH.
pub fn cyclonic_bunkers_motion(
    latitude_deg: f64,
    right_mover: (f64, f64),
    left_mover: (f64, f64),
) -> (f64, f64) {
    if latitude_deg < 0.0 {
        left_mover
    } else {
        right_mover
    }
}

/// Critical angle between the storm-relative inflow vector and the 0-500 m
/// shear vector.
///
/// A critical angle near 90 degrees is most favorable for low-level
/// mesocyclone development because it means the storm-relative inflow is
/// perpendicular to the low-level shear, maximizing streamwise vorticity
/// tilting.
///
/// # Arguments
/// * `storm_u`, `storm_v` -- Storm motion components (m/s)
/// * `u_sfc`, `v_sfc` -- Surface wind components (m/s)
/// * `u_500m`, `v_500m` -- Wind at 500 m AGL (m/s)
///
/// # Returns
/// The angle in degrees [0, 180] between the two vectors. Returns 0.0 if
/// either vector has near-zero magnitude.
///
/// # References
/// Esterheld, J. M., and D. J. Giuliano, 2008: Discriminating between
/// tornadic and non-tornadic supercells: A new hodograph technique.
pub fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    // Operational SHARPpy/MetPy convention: vector from the surface wind
    // point on the hodograph toward the storm-motion point.
    let inflow_u = storm_u - u_sfc;
    let inflow_v = storm_v - v_sfc;

    // 0-500 m shear vector
    let shear_u = u_500m - u_sfc;
    let shear_v = v_500m - v_sfc;

    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();

    if mag_inflow < 1e-10 || mag_shear < 1e-10 {
        return 0.0;
    }

    let cos_angle = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear);
    // Clamp to [-1, 1] to avoid NaN from floating-point rounding
    cos_angle.clamp(-1.0, 1.0).acos() * (180.0 / PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn bunkers_matches_documented_layer_mean_sounding() {
        // Based on the classic MetPy/operational-style Bunkers example profile.
        let heights_m = [250.0, 700.0, 1500.0, 3100.0, 5720.0, 7120.0];
        let u_prof = [
            -0.6657395241936066,
            -9.450182969108416e-16,
            1.7866452622337277,
            7.716660000000002,
            16.5339114538791,
            29.005153836455154,
        ];
        let v_prof = [
            2.4845737288972667,
            7.71666,
            10.132568793812247,
            13.365647184734451,
            19.70434837479498,
            10.557012636781815,
        ];

        let ((rm_u, rm_v), (lm_u, lm_v), (mean_u, mean_v)) =
            bunkers_storm_motion(&u_prof, &v_prof, &heights_m);

        assert_close(mean_u, 6.247699656874275);
        assert_close(mean_v, 10.53760757690408);
        assert_close(rm_u, 11.1883411258556);
        assert_close(rm_v, 4.89490770228891);
        assert_close(lm_u, 1.3070581878929506);
        assert_close(lm_v, 16.180307451519248);
    }

    #[test]
    fn mean_wind_npw_includes_interpolated_top_endpoint() {
        let heights_m = [0.0, 3000.0, 9000.0];
        let u_prof = [0.0, 6.0, 18.0];
        let v_prof = [0.0, 0.0, 0.0];

        let (mean_u, mean_v) = mean_wind_npw(&u_prof, &v_prof, &heights_m, 0.0, 6000.0);

        // Surface + interior 3 km level + interpolated 6 km endpoint (12 m/s)
        assert_close(mean_u, 6.0);
        assert_close(mean_v, 0.0);
    }

    #[test]
    fn mean_wind_npw_uses_interpolated_bottom_for_elevated_layers() {
        let heights_m = [0.0, 1000.0, 2000.0, 5500.0, 6000.0, 7000.0];
        let u_prof = [0.0, 10.0, 20.0, 55.0, 60.0, 70.0];
        let v_prof = [0.0, 0.0, 0.0, 5.0, 10.0, 10.0];

        let (mean_u, mean_v) = mean_wind_npw(&u_prof, &v_prof, &heights_m, 5500.0, 6000.0);

        assert_close(mean_u, 57.5);
        assert_close(mean_v, 7.5);
    }

    #[test]
    fn sharppy_npw_resamples_uniformly_in_pressure_not_native_levels() {
        let heights_m = [0.0, 10.0, 6000.0];
        let pressure_hpa = [1000.0, 999.0, 900.0];
        let u_prof = [0.0, 100.0, 0.0];
        let v_prof = [0.0, 0.0, 0.0];

        let (resampled_u, resampled_v) = mean_wind_npw_pressure_resampled(
            &u_prof,
            &v_prof,
            &heights_m,
            &pressure_hpa,
            0.0,
            6000.0,
        );
        let (native_u, _) = mean_wind_npw(&u_prof, &v_prof, &heights_m, 0.0, 6000.0);

        // SHARPpy's dp=-1 sequence has 101 samples (1000 through 900 hPa).
        assert_close(resampled_u, 50.357_154_417_278_1);
        assert_close(resampled_v, 0.0);
        assert!((resampled_u - native_u).abs() > 10.0);
    }

    #[test]
    fn critical_angle_matches_sharppy_inflow_direction() {
        let angle = critical_angle(10.0, 0.0, 0.0, 0.0, 10.0, 10.0);

        assert_close(angle, 45.0);
    }

    #[test]
    fn cyclonic_bunkers_motion_switches_mover_by_hemisphere() {
        let right = (10.0, 2.0);
        let left = (4.0, 8.0);

        assert_eq!(cyclonic_bunkers_motion(35.0, right, left), right);
        assert_eq!(cyclonic_bunkers_motion(-35.0, right, left), left);
    }
}
