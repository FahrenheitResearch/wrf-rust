//! The destaggers and the element-wise full-field maps run on rayon; each
//! output element is still computed on its own, so the answer must be the
//! serial arithmetic bit for bit, on fields large enough to be split.

mod common;

use wrf_core::grid::{destagger_x, destagger_y, destagger_z};
use wrf_core::{getvar, ComputeOpts, WrfFile};

fn field(n: usize, seed: f64) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.618_033_988_75 + seed).sin() * 40.0 + 280.0).collect()
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn destaggers_match_the_serial_average() {
    let (nz, ny, nx) = (12usize, 37usize, 53usize);
    let u = field(nz * ny * (nx + 1), 0.1);
    let mut want = Vec::new();
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        let at = k * ny * (nx + 1) + j * (nx + 1) + i;
        want.push(0.5 * (u[at] + u[at + 1]));
    } } }
    assert!(same_bits(&destagger_x(&u, nz, ny, nx + 1), &want));

    let v = field(nz * (ny + 1) * nx, 0.2);
    let mut want = Vec::new();
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        let at = k * (ny + 1) * nx + j * nx + i;
        want.push(0.5 * (v[at] + v[at + nx]));
    } } }
    assert!(same_bits(&destagger_y(&v, nz, ny + 1, nx), &want));

    let w = field((nz + 1) * ny * nx, 0.3);
    let plane = ny * nx;
    let want: Vec<f64> = (0..nz * plane).map(|idx| 0.5 * (w[idx] + w[idx + plane])).collect();
    assert!(same_bits(&destagger_z(&w, nz + 1, ny, nx), &want));
}

#[test]
fn element_wise_fields_match_the_serial_arithmetic() {
    let (nx, ny, nz) = (40usize, 30usize, 8usize); // 9600 points a volume
    let p = field(nz * ny * nx, 0.4);
    let pb = field(nz * ny * nx, 0.5);
    let t = field(nz * ny * nx, 0.6);
    let ph = field((nz + 1) * ny * nx, 0.7);
    let phb = field((nz + 1) * ny * nx, 0.8);
    let at3 = |data: &[f64], k: usize, j: usize, i: usize| data[(k * ny + j) * nx + i];
    let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
    // The test file stores float32: the expected values start from the same rounding.
    let r = |x: f64| x as f32 as f64;
    file.mass3("P", |k, j, i| at3(&p, k, j, i) - 280.0)
        .mass3("PB", |k, j, i| 50_000.0 + 100.0 * at3(&pb, k, j, i))
        .mass3("T", |k, j, i| at3(&t, k, j, i) - 290.0)
        .zstag3("PH", |k, j, i| at3(&ph, k, j, i))
        .zstag3("PHB", |k, j, i| 10.0 * at3(&phb, k, j, i) * (k + 1) as f64)
        .plane("HGT", |j, i| (j + i) as f64);
    let path = file.write("parallel-maps");
    let opened = WrfFile::open(&path).expect("open test file");
    std::fs::remove_file(&path).ok();

    let n = nz * ny * nx;
    let pres: Vec<f64> = (0..n).map(|idx| r(p[idx] - 280.0) + r(50_000.0 + 100.0 * pb[idx])).collect();
    let theta: Vec<f64> = (0..n).map(|idx| r(t[idx] - 290.0) + 300.0).collect();
    let nzs = nz + 1;
    let phb_at = |idx: usize| r(10.0 * phb[idx] * (idx / (ny * nx) + 1) as f64);
    let geo_stag: Vec<f64> = (0..nzs * ny * nx).map(|idx| r(ph[idx]) + phb_at(idx)).collect();
    let geo = destagger_z(&geo_stag, nzs, ny, nx);

    let out = |name: &str| getvar(&opened, name, Some(0), &ComputeOpts::default()).unwrap().data;
    assert!(same_bits(&out("pressure"), &pres.iter().map(|v| v / 100.0).collect::<Vec<_>>()), "pressure");
    assert!(same_bits(&out("theta"), &theta), "theta");
    assert!(same_bits(&out("geopt"), &geo), "geopt");
    let height: Vec<f64> = geo.iter().map(|g| g / 9.80665).collect();
    assert!(same_bits(&out("height"), &height), "height");
}
