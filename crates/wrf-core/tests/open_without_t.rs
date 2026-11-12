//! A history file without the 3-D potential temperature T still opens.
//! Up to 0.2.39 the pure-Rust reader took the grid dimensions from T alone
//! and refused such a file ("variable 'T' not found"), so a surface-only
//! selection could not be read at all.

mod common;

use wrf_core::{getvar, ComputeOpts, WrfFile};

fn open(tag: &str, file: &common::WrfLike) -> WrfFile {
    let path = file.write(tag);
    let opened = WrfFile::open(&path);
    std::fs::remove_file(&path).ok();
    opened.expect("open test file")
}

#[test]
fn another_mass_volume_gives_the_dimensions() {
    let (nx, ny, nz) = (5usize, 4usize, 3usize);
    let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
    file.mass3("P", |_, _, _| 0.0).mass3("PB", |k, _, _| 90_000.0 - 1_000.0 * k as f64);
    let opened = open("no-t-volume", &file);
    assert_eq!((opened.nx, opened.ny, opened.nz, opened.nt), (nx, ny, nz, 1));
    let pressure = getvar(&opened, "pressure", Some(0), &ComputeOpts::default()).unwrap();
    assert_eq!(pressure.shape, vec![nz, ny, nx]);
}

#[test]
fn a_surface_only_file_takes_the_latitude_plane_and_the_level_count() {
    let (nx, ny, nz) = (5usize, 4usize, 3usize);
    let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
    file.plane("XLAT", |j, _| 40.0 + j as f64)
        .plane("XLONG", |_, i| -100.0 + i as f64)
        .plane("T2", |j, i| 280.0 + (j * nx + i) as f64);
    let opened = open("surface-only", &file);
    // BOTTOM-TOP_GRID_DIMENSION is nz + 1 (the staggered count).
    assert_eq!((opened.nx, opened.ny, opened.nz, opened.nt), (nx, ny, nz, 1));
    let t2 = getvar(&opened, "T2", Some(0), &ComputeOpts::default()).unwrap();
    assert_eq!(t2.shape, vec![ny, nx]);
    assert_eq!(t2.data[nx + 2], 280.0 + (nx + 2) as f64);
}

#[test]
fn a_file_with_t_keeps_the_dimensions_of_t() {
    let (nx, ny, nz) = (5usize, 4usize, 3usize);
    let mut file = common::WrfLike::new(nx, ny, nz, 3000.0, 3000.0);
    file.mass3("T", |_, _, _| 0.0).plane("XLAT", |j, _| 40.0 + j as f64);
    let opened = open("with-t", &file);
    assert_eq!((opened.nx, opened.ny, opened.nz, opened.nt), (nx, ny, nz, 1));
}

#[test]
fn a_file_without_any_grid_field_still_refuses_to_open() {
    let mut file = common::WrfLike::new(5, 4, 3, 3000.0, 3000.0);
    file.plane_u("MAPFAC_U", |_, _| 1.0);
    let path = file.write("no-grid");
    let opened = WrfFile::open(&path);
    std::fs::remove_file(&path).ok();
    assert!(opened.is_err());
}
