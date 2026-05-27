/// Capability report for the active WRF file reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderCapabilities {
    pub backend: &'static str,
    pub file_format: &'static str,
    pub supports_time_slicing: bool,
    pub supports_classic_netcdf: bool,
    pub supports_netcdf4_hdf5: bool,
    pub supports_chunked: bool,
    pub supports_deflate: bool,
    pub supports_shuffle: bool,
    pub unsupported_features: &'static [&'static str],
}

impl ReaderCapabilities {
    pub fn netcdf_backend() -> Self {
        Self {
            backend: "netcdf-backend",
            file_format: "netcdf",
            supports_time_slicing: true,
            supports_classic_netcdf: true,
            supports_netcdf4_hdf5: true,
            supports_chunked: true,
            supports_deflate: true,
            supports_shuffle: true,
            unsupported_features: &[],
        }
    }

    pub fn pure_classic() -> Self {
        Self {
            backend: "pure-rust-reader",
            file_format: "classic_netcdf",
            supports_time_slicing: true,
            supports_classic_netcdf: true,
            supports_netcdf4_hdf5: false,
            supports_chunked: false,
            supports_deflate: false,
            supports_shuffle: false,
            unsupported_features: &["netcdf4_hdf5_features"],
        }
    }

    pub fn pure_hdf5() -> Self {
        Self {
            backend: "pure-rust-reader",
            file_format: "netcdf4_hdf5",
            supports_time_slicing: true,
            supports_classic_netcdf: false,
            supports_netcdf4_hdf5: true,
            supports_chunked: true,
            supports_deflate: true,
            supports_shuffle: true,
            unsupported_features: &[
                "hdf5_compound_types",
                "hdf5_variable_length_types",
                "hdf5_external_storage",
                "non_wrf_group_layouts",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_reader_capabilities_name_supported_formats() {
        let classic = ReaderCapabilities::pure_classic();
        assert_eq!(classic.backend, "pure-rust-reader");
        assert!(classic.supports_classic_netcdf);
        assert!(!classic.supports_netcdf4_hdf5);

        let hdf5 = ReaderCapabilities::pure_hdf5();
        assert!(hdf5.supports_netcdf4_hdf5);
        assert!(hdf5.supports_deflate);
        assert!(hdf5.unsupported_features.contains(&"hdf5_compound_types"));
    }
}
