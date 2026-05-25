//! Local WRF run and file discovery.
//!
//! This crate is intentionally boring: paths, catalogs, manifests, and retention
//! candidates only. No meteorological calculations and no rendering.

use std::fmt;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store root does not exist: {0}")]
    MissingRoot(PathBuf),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrfStore {
    root: PathBuf,
}

impl WrfStore {
    pub fn new<P: Into<PathBuf>>(root: P) -> Self {
        Self { root: root.into() }
    }

    pub fn open<P: Into<PathBuf>>(root: P) -> StoreResult<Self> {
        let store = Self::new(root);
        if !store.root.exists() {
            return Err(StoreError::MissingRoot(store.root));
        }
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn run_dir(&self, run: &WrfRunKey) -> PathBuf {
        self.root
            .join(sanitize_component(&run.model_id))
            .join(run.init_slug())
            .join(&run.domain)
    }

    pub fn file_path(&self, key: &WrfFileKey) -> PathBuf {
        self.run_dir(&key.run).join(key.filename())
    }

    pub fn manifest_path(&self, run: &WrfRunKey) -> PathBuf {
        self.run_dir(run).join("wrf-run-manifest.json")
    }

    pub fn rendered_product_path(
        &self,
        key: &WrfFileKey,
        product: &str,
        extension: &str,
    ) -> PathBuf {
        self.run_dir(&key.run)
            .join("products")
            .join(sanitize_component(product))
            .join(format!(
                "{}.{}",
                key.valid_time.format("%Y%m%d%H%M%S"),
                extension.trim_start_matches('.')
            ))
    }

    pub fn discover_files(&self, run: &WrfRunKey, kind: WrfFileKind) -> StoreResult<Vec<PathBuf>> {
        let dir = self.run_dir(run);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let prefix = format!("{}_{}_", kind.prefix(), run.domain);
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with(&prefix) {
                paths.push(entry.path());
            }
        }
        paths.sort();
        Ok(paths)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WrfRunKey {
    pub model_id: String,
    pub init_time: DateTime<Utc>,
    pub domain: String,
}

impl WrfRunKey {
    pub fn init_slug(&self) -> String {
        self.init_time.format("%Y%m%d%H").to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrfFileKind {
    Wrfout,
    Wrfinput,
    Wrfbdy,
    Wrfrst,
    Auxhist,
}

impl WrfFileKind {
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Wrfout => "wrfout",
            Self::Wrfinput => "wrfinput",
            Self::Wrfbdy => "wrfbdy",
            Self::Wrfrst => "wrfrst",
            Self::Auxhist => "auxhist",
        }
    }
}

impl fmt::Display for WrfFileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.prefix())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WrfFileKey {
    pub run: WrfRunKey,
    pub valid_time: DateTime<Utc>,
    pub kind: WrfFileKind,
}

impl WrfFileKey {
    pub fn filename(&self) -> String {
        match self.kind {
            WrfFileKind::Wrfinput | WrfFileKind::Wrfbdy => {
                format!("{}_{}", self.kind.prefix(), self.run.domain)
            }
            WrfFileKind::Wrfout | WrfFileKind::Wrfrst | WrfFileKind::Auxhist => format!(
                "{}_{}_{}",
                self.kind.prefix(),
                self.run.domain,
                self.valid_time.format("%Y-%m-%d_%H_%M_%S")
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedProductKey {
    pub input: WrfFileKey,
    pub product: String,
    pub extension: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionCandidate {
    pub path: PathBuf,
    pub reason: String,
}

fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_run() -> WrfRunKey {
        WrfRunKey {
            model_id: "my-wrf".to_string(),
            init_time: Utc.with_ymd_and_hms(2026, 5, 18, 18, 0, 0).unwrap(),
            domain: "d01".to_string(),
        }
    }

    #[test]
    fn builds_canonical_wrfout_filename() {
        let key = WrfFileKey {
            run: sample_run(),
            valid_time: Utc.with_ymd_and_hms(2026, 5, 18, 19, 6, 0).unwrap(),
            kind: WrfFileKind::Wrfout,
        };
        assert_eq!(key.filename(), "wrfout_d01_2026-05-18_19_06_00");
    }

    #[test]
    fn store_keeps_paths_under_run_dir() {
        let store = WrfStore::new("/tmp/wrf-store");
        let key = WrfFileKey {
            run: sample_run(),
            valid_time: Utc.with_ymd_and_hms(2026, 5, 18, 19, 0, 0).unwrap(),
            kind: WrfFileKind::Wrfout,
        };
        let path = store.file_path(&key);
        assert!(path.ends_with("my-wrf/2026051818/d01/wrfout_d01_2026-05-18_19_00_00"));
    }
}
