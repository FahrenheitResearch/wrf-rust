//! Tiny WRF-like files for integration tests: a minimal classic NetCDF
//! (CDF-1) writer, enough for `WrfFile::open` through the pure-Rust reader.
#![allow(dead_code)]

use std::path::PathBuf;

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;
const NC_CHAR: u32 = 2;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;

enum Att {
    Int(i32),
    Float(f32),
    Text(String),
}

enum Data {
    Float(Vec<f32>),
    Char(Vec<u8>),
}

struct Var {
    name: String,
    dims: Vec<usize>,
    data: Data,
}

/// A WRF-shaped file under construction: dimensions Time (1), bottom_top,
/// south_north, west_east and their staggered forms, plus DateStrLen.
pub struct WrfLike {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    dims: Vec<(String, usize)>,
    atts: Vec<(String, Att)>,
    vars: Vec<Var>,
}

impl WrfLike {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64) -> Self {
        let dims = vec![
            ("Time".to_string(), 1),
            ("DateStrLen".to_string(), 19),
            ("west_east".to_string(), nx),
            ("south_north".to_string(), ny),
            ("bottom_top".to_string(), nz),
            ("west_east_stag".to_string(), nx + 1),
            ("south_north_stag".to_string(), ny + 1),
            ("bottom_top_stag".to_string(), nz + 1),
        ];
        let mut file = Self { nx, ny, nz, dims, atts: Vec::new(), vars: Vec::new() };
        file.atts.push(("DX".to_string(), Att::Float(dx as f32)));
        file.atts.push(("DY".to_string(), Att::Float(dy as f32)));
        file.atts.push(("BOTTOM-TOP_GRID_DIMENSION".to_string(), Att::Int(nz as i32 + 1)));
        file.atts.push(("MAP_PROJ".to_string(), Att::Int(1)));
        file.atts.push(("TITLE".to_string(), Att::Text("wrf-core test file".to_string())));
        file.vars.push(Var {
            name: "Times".to_string(),
            dims: vec![0, 1],
            data: Data::Char(b"2026-01-01_00:00:00".to_vec()),
        });
        file
    }

    fn dim(&self, name: &str) -> usize {
        self.dims.iter().position(|(n, _)| n == name).expect("known dimension")
    }

    fn push(&mut self, name: &str, dims: &[&str], values: Vec<f32>) -> &mut Self {
        let ids: Vec<usize> = dims.iter().map(|d| self.dim(d)).collect();
        let expected: usize = ids.iter().map(|&id| self.dims[id].1).product();
        assert_eq!(values.len(), expected, "{name}: {} values for {dims:?}", values.len());
        self.vars.push(Var { name: name.to_string(), dims: ids, data: Data::Float(values) });
        self
    }

    /// A 3-D mass-point field `[Time, bottom_top, south_north, west_east]`.
    pub fn mass3(&mut self, name: &str, f: impl Fn(usize, usize, usize) -> f64) -> &mut Self {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let values = grid3(nz, ny, nx, f);
        self.push(name, &["Time", "bottom_top", "south_north", "west_east"], values)
    }

    /// U on the x-staggered faces `[Time, bottom_top, south_north, west_east_stag]`.
    pub fn u(&mut self, f: impl Fn(usize, usize, usize) -> f64) -> &mut Self {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let values = grid3(nz, ny, nx + 1, f);
        self.push("U", &["Time", "bottom_top", "south_north", "west_east_stag"], values)
    }

    /// V on the y-staggered faces `[Time, bottom_top, south_north_stag, west_east]`.
    pub fn v(&mut self, f: impl Fn(usize, usize, usize) -> f64) -> &mut Self {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let values = grid3(nz, ny + 1, nx, f);
        self.push("V", &["Time", "bottom_top", "south_north_stag", "west_east"], values)
    }

    /// PH or PHB on the z-staggered levels `[Time, bottom_top_stag, south_north, west_east]`.
    pub fn zstag3(&mut self, name: &str, f: impl Fn(usize, usize, usize) -> f64) -> &mut Self {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let values = grid3(nz + 1, ny, nx, f);
        self.push(name, &["Time", "bottom_top_stag", "south_north", "west_east"], values)
    }

    /// A mass-point plane `[Time, south_north, west_east]`.
    pub fn plane(&mut self, name: &str, f: impl Fn(usize, usize) -> f64) -> &mut Self {
        let (nx, ny) = (self.nx, self.ny);
        let values = grid3(1, ny, nx, |_, j, i| f(j, i));
        self.push(name, &["Time", "south_north", "west_east"], values)
    }

    /// A plane on the u faces `[Time, south_north, west_east_stag]`.
    pub fn plane_u(&mut self, name: &str, f: impl Fn(usize, usize) -> f64) -> &mut Self {
        let (nx, ny) = (self.nx, self.ny);
        let values = grid3(1, ny, nx + 1, |_, j, i| f(j, i));
        self.push(name, &["Time", "south_north", "west_east_stag"], values)
    }

    /// A plane on the v faces `[Time, south_north_stag, west_east]`.
    pub fn plane_v(&mut self, name: &str, f: impl Fn(usize, usize) -> f64) -> &mut Self {
        let (nx, ny) = (self.nx, self.ny);
        let values = grid3(1, ny + 1, nx, |_, j, i| f(j, i));
        self.push(name, &["Time", "south_north_stag", "west_east"], values)
    }

    /// Write the file into the system temporary folder under a name unique
    /// to this process and `tag`, and return its path.
    pub fn write(&self, tag: &str) -> PathBuf {
        let path = std::env::temp_dir()
            .join(format!("wrf-core-test-{}-{tag}.nc", std::process::id()));
        std::fs::write(&path, self.encode()).expect("write test file");
        path
    }

    fn encode(&self) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(b"CDF\x01");
        put_u32(&mut header, 0); // numrecs: no record variables
        put_u32(&mut header, NC_DIMENSION);
        put_u32(&mut header, self.dims.len() as u32);
        for (name, len) in &self.dims {
            put_name(&mut header, name);
            put_u32(&mut header, *len as u32);
        }
        put_u32(&mut header, NC_ATTRIBUTE);
        put_u32(&mut header, self.atts.len() as u32);
        for (name, att) in &self.atts {
            put_name(&mut header, name);
            match att {
                Att::Int(value) => {
                    put_u32(&mut header, NC_INT);
                    put_u32(&mut header, 1);
                    header.extend_from_slice(&value.to_be_bytes());
                }
                Att::Float(value) => {
                    put_u32(&mut header, NC_FLOAT);
                    put_u32(&mut header, 1);
                    header.extend_from_slice(&value.to_be_bytes());
                }
                Att::Text(text) => {
                    put_u32(&mut header, NC_CHAR);
                    put_u32(&mut header, text.len() as u32);
                    header.extend_from_slice(text.as_bytes());
                    pad4(&mut header);
                }
            }
        }
        // The variable list ends in each variable's data offset, which
        // depends on the header length: lay it out once to measure it.
        let sizes: Vec<usize> = self.vars.iter().map(|var| padded(var_bytes(var).len())).collect();
        let var_list = |offsets: &[usize]| {
            let mut out = Vec::new();
            put_u32(&mut out, NC_VARIABLE);
            put_u32(&mut out, self.vars.len() as u32);
            for (index, var) in self.vars.iter().enumerate() {
                put_name(&mut out, &var.name);
                put_u32(&mut out, var.dims.len() as u32);
                for &id in &var.dims {
                    put_u32(&mut out, id as u32);
                }
                put_u32(&mut out, 0); // no variable attributes
                put_u32(&mut out, 0);
                put_u32(&mut out, match var.data { Data::Float(_) => NC_FLOAT, Data::Char(_) => NC_CHAR });
                put_u32(&mut out, sizes[index] as u32);
                put_u32(&mut out, offsets[index] as u32);
            }
            out
        };
        let header_len = header.len() + var_list(&vec![0; self.vars.len()]).len();
        let mut offsets = Vec::with_capacity(self.vars.len());
        let mut at = header_len;
        for size in &sizes {
            offsets.push(at);
            at += size;
        }
        let mut out = header;
        out.extend(var_list(&offsets));
        for var in &self.vars {
            out.extend(var_bytes(var));
            pad4(&mut out);
        }
        out
    }
}

fn grid3(nz: usize, ny: usize, nx: usize, f: impl Fn(usize, usize, usize) -> f64) -> Vec<f32> {
    let mut values = Vec::with_capacity(nz * ny * nx);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                values.push(f(k, j, i) as f32);
            }
        }
    }
    values
}

fn var_bytes(var: &Var) -> Vec<u8> {
    match &var.data {
        Data::Float(values) => values.iter().flat_map(|v| v.to_be_bytes()).collect(),
        Data::Char(bytes) => bytes.clone(),
    }
}

fn padded(len: usize) -> usize {
    len.div_ceil(4) * 4
}

fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_name(out: &mut Vec<u8>, name: &str) {
    put_u32(out, name.len() as u32);
    out.extend_from_slice(name.as_bytes());
    pad4(out);
}
