pub mod compute;
pub mod diag;
pub mod error;
pub mod extract;
pub mod file;
pub mod grid;
pub mod met;
pub mod multi;
pub mod projection;
pub mod units;
pub mod variables;

/// WRF/NCAR gravitational acceleration used by diagnostics derived from model fields.
///
/// WRF constructs geopotential with 9.81 m s^-2, and wrf-python exposes the same
/// value from `fortran/wrf_constants.f90`.
pub(crate) const WRF_GRAVITY_M_S2: f64 = 9.81;

#[cfg(feature = "pure-rust-reader")]
pub mod classic_netcdf_reader;
#[cfg(feature = "pure-rust-reader")]
pub mod hdf5_reader;
#[cfg(feature = "pure-rust-reader")]
pub mod pure_reader;

pub use compute::{
    getvar, getvar_all_times, ComputeOpts, StormMotion, StormMotionMethod, VarOutput,
};
pub use error::{WrfError, WrfResult};
pub use file::WrfFile;
pub use projection::WrfProjection;
pub use units::WrfUnits;
