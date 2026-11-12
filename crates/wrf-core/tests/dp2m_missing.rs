//! The 2 m dewpoint keeps a missing input missing.  A converter from a
//! limited-area MPAS mesh writes NaN at every point off the mesh; up to
//! 0.2.39 the body floored Q2 with `q.max(1e-10)`, which turns NaN into
//! 1e-10, so the frame around the mesh was painted with the -40 F end of
//! the dewpoint scale.

mod common;

use wrf_core::diag::moisture::dewpoint_2m_c;
use wrf_core::{getvar, ComputeOpts, WrfFile};

#[test]
fn missing_q2_or_psfc_gives_missing_dewpoint() {
    assert!(dewpoint_2m_c(f64::NAN, 97_000.0).is_nan());
    assert!(dewpoint_2m_c(0.010, f64::NAN).is_nan());
    assert!(dewpoint_2m_c(f64::NAN, f64::NAN).is_nan());
    // f32 NaN widened by the reader is still NaN.
    assert!(dewpoint_2m_c(f32::NAN as f64, 97_000.0).is_nan());
}

#[test]
fn finite_inputs_keep_the_old_formula_bit_for_bit() {
    let old = |q: f64, p_pa: f64| {
        let q = q.max(1e-10);
        let p_hpa = p_pa / 100.0;
        let e = q * p_hpa / (0.622 + q);
        let ln_e = (e / 6.112).max(1e-10).ln();
        (243.5 * ln_e) / (17.67 - ln_e)
    };
    for &(q, p) in &[
        (0.010, 97_000.0),
        (0.0001, 85_000.0),
        (0.0, 101_325.0),
        (-1.0e-6, 99_000.0),
        (0.022, 100_500.0),
    ] {
        assert_eq!(dewpoint_2m_c(q, p).to_bits(), old(q, p).to_bits(), "q {q} p {p}");
    }
    // A moist plains surface: q 10 g/kg at 970 hPa is a Td near 13.5 C.
    let td = dewpoint_2m_c(0.010, 97_000.0);
    assert!((td - 13.5).abs() < 0.5, "{td}");
}

#[test]
fn dp2m_of_a_file_keeps_its_missing_points_missing() {
    // Off-mesh points (i == 0) carry NaN Q2 or NaN PSFC; the rest are finite.
    let (nx, ny, nz) = (4usize, 3usize, 2usize);
    let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
    file.mass3("T", |_, _, _| 0.0)
        .plane("Q2", |j, i| if i == 0 && j != 1 { f64::NAN } else { 0.010 })
        .plane("PSFC", |j, i| if i == 0 && j == 1 { f64::NAN } else { 97_000.0 });
    let path = file.write("dp2m-missing");
    let opened = WrfFile::open(&path).expect("open test file");
    std::fs::remove_file(&path).ok();
    let out = getvar(&opened, "dp2m", Some(0), &ComputeOpts::default()).unwrap();
    for j in 0..ny {
        for i in 0..nx {
            let value = out.data[j * nx + i];
            if i == 0 {
                assert!(value.is_nan(), "({j}, {i}): {value}");
            } else {
                assert_eq!(value.to_bits(), dewpoint_2m_c(0.010f32 as f64, 97_000.0).to_bits());
            }
        }
    }
}
