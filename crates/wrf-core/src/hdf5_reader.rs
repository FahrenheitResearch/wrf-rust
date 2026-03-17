//! Pure-Rust HDF5/netCDF4 reader for WRF output files.
//!
//! Handles HDF5 superblock v2, object header v2, B-tree v2 (for dense link/attribute storage),
//! fractal heap, chunked + deflate + shuffle compressed datasets, and global attributes.
//! Designed specifically for the subset of HDF5 used by netCDF4/WRF output.
//!
//! This module is gated behind the `pure-rust-reader` feature flag.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, BufReader};
use std::path::Path;

use flate2::read::ZlibDecoder;

use crate::error::{WrfError, WrfResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];
const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";
const BTHD_SIGNATURE: [u8; 4] = *b"BTHD";
const BTLF_SIGNATURE: [u8; 4] = *b"BTLF";
const FRHP_SIGNATURE: [u8; 4] = *b"FRHP";
const FHDB_SIGNATURE: [u8; 4] = *b"FHDB";
const FHIB_SIGNATURE: [u8; 4] = *b"FHIB";
const TREE_SIGNATURE: [u8; 4] = *b"TREE";
const UNDEF_ADDR: u64 = 0xFFFF_FFFF_FFFF_FFFF;

// HDF5 message types
const MSG_DATASPACE: u8 = 0x01;
const MSG_LINK_INFO: u8 = 0x02;
const MSG_DATATYPE: u8 = 0x03;
const MSG_DATA_LAYOUT: u8 = 0x08;
const MSG_FILTER_PIPELINE: u8 = 0x0B;
const MSG_ATTRIBUTE: u8 = 0x0C;
const MSG_CONTINUATION: u8 = 0x10;
const MSG_ATTR_INFO: u8 = 0x15;
const MSG_LINK: u8 = 0x06;

// ---------------------------------------------------------------------------
// Error helper -- convert any string-like error into WrfError::NetCdf
// (we reuse the NetCdf variant to mean "file I/O error" generically)
// ---------------------------------------------------------------------------

fn hdf5_err(msg: impl Into<String>) -> WrfError {
    WrfError::NetCdf(msg.into())
}

fn io_err(e: std::io::Error) -> WrfError {
    WrfError::NetCdf(e.to_string())
}

// ---------------------------------------------------------------------------
// Low-level read helpers
// ---------------------------------------------------------------------------

fn read_u8_at(r: &RefCell<BufReader<std::fs::File>>, off: u64) -> WrfResult<u8> {
    let mut f = r.borrow_mut();
    f.seek(SeekFrom::Start(off)).map_err(io_err)?;
    let mut b = [0u8; 1];
    f.read_exact(&mut b).map_err(io_err)?;
    Ok(b[0])
}

fn read_bytes(r: &RefCell<BufReader<std::fs::File>>, off: u64, n: usize) -> WrfResult<Vec<u8>> {
    let mut f = r.borrow_mut();
    f.seek(SeekFrom::Start(off)).map_err(io_err)?;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(io_err)?;
    Ok(buf)
}

fn read_u16(r: &RefCell<BufReader<std::fs::File>>, off: u64) -> WrfResult<u16> {
    let b = read_bytes(r, off, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(r: &RefCell<BufReader<std::fs::File>>, off: u64) -> WrfResult<u32> {
    let b = read_bytes(r, off, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(r: &RefCell<BufReader<std::fs::File>>, off: u64) -> WrfResult<u64> {
    let b = read_bytes(r, off, 8)?;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}

fn le_u16(b: &[u8]) -> u16 { u16::from_le_bytes([b[0], b[1]]) }
fn le_u32(b: &[u8]) -> u32 { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
fn le_u64(b: &[u8]) -> u64 { u64::from_le_bytes(b[0..8].try_into().unwrap()) }

// ---------------------------------------------------------------------------
// Parsed structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DatasetInfo {
    shape: Vec<usize>,
    dtype: DType,
    layout: Layout,
    filters: Vec<Filter>,
}

#[derive(Debug, Clone)]
enum DType {
    F32,
    F64,
    I32,
    U8,
}

impl DType {
    fn size(&self) -> usize {
        match self { DType::F32 => 4, DType::F64 => 8, DType::I32 => 4, DType::U8 => 1 }
    }
}

#[derive(Debug, Clone)]
enum Layout {
    Contiguous { addr: u64, size: u64 },
    Chunked { addr: u64, chunk_dims: Vec<u32>, ndims: u8 },
}

#[derive(Debug, Clone)]
enum Filter {
    Deflate { level: u32 },
    Shuffle { element_size: u32 },
}

#[derive(Debug, Clone)]
pub(crate) struct HdfAttributeValue {
    f32_val: Option<f32>,
    i32_val: Option<i32>,
    string_val: Option<String>,
    f64_val: Option<f64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pure-Rust HDF5 file reader.  Reads netCDF4/HDF5 files produced by WRF
/// without requiring any C libraries.
pub struct PureRustFile {
    reader: RefCell<BufReader<std::fs::File>>,
    #[allow(dead_code)]
    root_ohdr_addr: u64,
    /// dataset name -> object header address
    datasets: HashMap<String, u64>,
    /// cached dataset info
    ds_cache: RefCell<HashMap<String, DatasetInfo>>,
    /// global attributes (read once)
    global_attrs: HashMap<String, HdfAttributeValue>,
}

impl PureRustFile {
    /// Open an HDF5/netCDF4 file and parse the root group.
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let file = std::fs::File::open(path.as_ref()).map_err(io_err)?;
        let reader = RefCell::new(BufReader::new(file));

        // --- Superblock v2 ---
        let sig = read_bytes(&reader, 0, 8)?;
        if sig != HDF5_SIGNATURE {
            return Err(hdf5_err("Not an HDF5 file"));
        }
        let version = read_u8_at(&reader, 8)?;
        if version < 2 {
            return Err(hdf5_err(format!("Unsupported superblock version {version}, need v2+")));
        }
        let offset_size = read_u8_at(&reader, 9)?;
        let length_size = read_u8_at(&reader, 10)?;
        if offset_size != 8 || length_size != 8 {
            return Err(hdf5_err(format!(
                "Unsupported offset_size={offset_size} length_size={length_size}"
            )));
        }
        // file_consistency_flags at 11
        let _base_addr = read_u64(&reader, 12)?;
        let _sb_ext_addr = read_u64(&reader, 20)?;
        let _eof_addr = read_u64(&reader, 28)?;
        let root_ohdr_addr = read_u64(&reader, 36)?;

        // Read root group links (datasets)
        let datasets = Self::read_group_links_static(&reader, root_ohdr_addr)?;

        // Read global attributes from root group
        let global_attrs = Self::read_attributes_static(&reader, root_ohdr_addr)?;

        Ok(PureRustFile {
            reader,
            root_ohdr_addr,
            datasets,
            ds_cache: RefCell::new(HashMap::new()),
            global_attrs,
        })
    }

    // --- Public attribute methods ---

    /// Read a global attribute as f32.
    pub fn global_attr_f32(&self, name: &str) -> WrfResult<f32> {
        let attr = self.global_attrs.get(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        if let Some(v) = attr.f32_val { return Ok(v); }
        if let Some(v) = attr.f64_val { return Ok(v as f32); }
        if let Some(v) = attr.i32_val { return Ok(v as f32); }
        Err(WrfError::AttrType(format!("'{name}' is not numeric")))
    }

    /// Read a global attribute as f64.
    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        let attr = self.global_attrs.get(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        if let Some(v) = attr.f64_val { return Ok(v); }
        if let Some(v) = attr.f32_val { return Ok(v as f64); }
        if let Some(v) = attr.i32_val { return Ok(v as f64); }
        Err(WrfError::AttrType(format!("'{name}' is not numeric")))
    }

    /// Read a global attribute as i32.
    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        let attr = self.global_attrs.get(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        if let Some(v) = attr.i32_val { return Ok(v); }
        if let Some(v) = attr.f32_val { return Ok(v as i32); }
        if let Some(v) = attr.f64_val { return Ok(v as i32); }
        Err(WrfError::AttrType(format!("'{name}' is not numeric")))
    }

    /// Read a global attribute as String.
    pub fn global_attr_string(&self, name: &str) -> WrfResult<String> {
        let attr = self.global_attrs.get(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        if let Some(ref s) = attr.string_val { return Ok(s.clone()); }
        if let Some(v) = attr.i32_val { return Ok(v.to_string()); }
        if let Some(v) = attr.f32_val { return Ok(v.to_string()); }
        Err(WrfError::AttrType(format!("'{name}' cannot be read as string")))
    }

    // --- Public dataset methods ---

    /// Check whether a variable/dataset exists.
    pub fn has_dataset(&self, name: &str) -> bool {
        self.datasets.contains_key(name)
    }

    /// Return the shape (dimensions) of a dataset.
    pub fn dataset_shape(&self, name: &str) -> WrfResult<Vec<usize>> {
        let info = self.get_dataset_info(name)?;
        Ok(info.shape.clone())
    }

    /// Read a dataset and return its values as `Vec<f64>`.
    /// Works for F32, F64, I32, and U8 source types.
    pub fn read_f64(&self, name: &str) -> WrfResult<Vec<f64>> {
        let info = self.get_dataset_info(name)?;
        let raw = self.read_raw_data(name, &info)?;
        let out = match info.dtype {
            DType::F32 => {
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect()
            }
            DType::F64 => {
                raw.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            DType::I32 => {
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect()
            }
            DType::U8 => {
                raw.iter().map(|&b| b as f64).collect()
            }
        };
        Ok(out)
    }

    /// Read a dataset as raw bytes (useful for U8/char data like Times).
    pub fn read_u8(&self, name: &str) -> WrfResult<Vec<u8>> {
        let info = self.get_dataset_info(name)?;
        self.read_raw_data(name, &info)
    }

    // -----------------------------------------------------------------------
    // Internal: dataset info resolution
    // -----------------------------------------------------------------------

    fn get_dataset_info(&self, name: &str) -> WrfResult<DatasetInfo> {
        // Check cache
        if let Some(info) = self.ds_cache.borrow().get(name) {
            return Ok(info.clone());
        }
        let ohdr_addr = *self.datasets.get(name)
            .ok_or_else(|| WrfError::VarNotFound(name.to_string()))?;
        let info = self.parse_dataset_ohdr(ohdr_addr)?;
        self.ds_cache.borrow_mut().insert(name.to_string(), info.clone());
        Ok(info)
    }

    fn parse_dataset_ohdr(&self, addr: u64) -> WrfResult<DatasetInfo> {
        let mut shape = Vec::new();
        let mut dtype = None;
        let mut layout = None;
        let mut filters = Vec::new();

        self.walk_ohdr_messages(addr, &mut |msg_type, data| {
            match msg_type {
                MSG_DATASPACE => {
                    shape = parse_dataspace(data)?;
                }
                MSG_DATATYPE => {
                    dtype = Some(parse_datatype(data)?);
                }
                MSG_DATA_LAYOUT => {
                    layout = Some(parse_layout(data)?);
                }
                MSG_FILTER_PIPELINE => {
                    filters = parse_filters(data)?;
                }
                _ => {}
            }
            Ok(())
        })?;

        Ok(DatasetInfo {
            shape,
            dtype: dtype.ok_or_else(|| hdf5_err("No datatype message in dataset object header"))?,
            layout: layout.ok_or_else(|| hdf5_err("No layout message in dataset object header"))?,
            filters,
        })
    }

    // -----------------------------------------------------------------------
    // Internal: walk object header v2 messages
    // -----------------------------------------------------------------------

    fn walk_ohdr_messages(
        &self,
        addr: u64,
        cb: &mut dyn FnMut(u8, &[u8]) -> WrfResult<()>,
    ) -> WrfResult<()> {
        let sig = read_bytes(&self.reader, addr, 4)?;
        if sig != OHDR_SIGNATURE {
            return Err(hdf5_err(format!("Expected OHDR at 0x{addr:x}, got {sig:?}")));
        }
        let version = read_u8_at(&self.reader, addr + 4)?;
        if version != 2 {
            return Err(hdf5_err(format!("Unsupported OHDR version {version}")));
        }
        let flags = read_u8_at(&self.reader, addr + 5)?;

        let mut pos = addr + 6;

        // bit 5 (0x20): times stored (4*4=16 bytes)
        if flags & 0x20 != 0 {
            pos += 16;
        }
        // bit 4 (0x10): non-default attr phase change values (2+2=4 bytes)
        if flags & 0x10 != 0 {
            pos += 4;
        }

        // Chunk#0 size (variable length based on flags bits 0-1)
        let size_of_chunk_size = 1 << (flags & 0x03);
        let chunk_size = match size_of_chunk_size {
            1 => read_u8_at(&self.reader, pos)? as u64,
            2 => read_u16(&self.reader, pos)? as u64,
            4 => read_u32(&self.reader, pos)? as u64,
            8 => read_u64(&self.reader, pos)?,
            _ => return Err(hdf5_err("Invalid chunk size length")),
        };
        pos += size_of_chunk_size as u64;

        // Now parse messages in this chunk
        let chunk_end = pos + chunk_size;
        // bit 2: attribute creation order tracked => messages have creation order field
        let msg_has_co = (flags & 0x04) != 0;
        self.parse_ohdr_messages(pos, chunk_end, msg_has_co, cb)?;

        Ok(())
    }

    fn parse_ohdr_messages(
        &self,
        start: u64,
        end: u64,
        msg_has_creation_order: bool,
        cb: &mut dyn FnMut(u8, &[u8]) -> WrfResult<()>,
    ) -> WrfResult<()> {
        let msg_hdr_size: u64 = if msg_has_creation_order { 6 } else { 4 };
        let mut pos = start;
        while pos + msg_hdr_size <= end {
            let msg_type = read_u8_at(&self.reader, pos)?;
            let msg_size = read_u16(&self.reader, pos + 1)? as u64;
            let _msg_flags = read_u8_at(&self.reader, pos + 3)?;
            pos += msg_hdr_size;

            if msg_type == 0 && msg_size == 0 {
                // Padding / end of messages
                break;
            }

            if pos + msg_size > end {
                break;
            }

            if msg_type == MSG_CONTINUATION {
                // Continuation: addr(8) + length(8)
                let data = read_bytes(&self.reader, pos, msg_size as usize)?;
                if data.len() >= 16 {
                    let cont_addr = le_u64(&data[0..8]);
                    let cont_len = le_u64(&data[8..16]);
                    if cont_addr != UNDEF_ADDR && cont_len > 0 {
                        // Continuation block starts with OCHK signature (4 bytes)
                        let cont_sig = read_bytes(&self.reader, cont_addr, 4)?;
                        if &cont_sig == b"OCHK" {
                            self.parse_ohdr_messages(
                                cont_addr + 4,
                                cont_addr + cont_len - 4,
                                msg_has_creation_order,
                                cb,
                            )?;
                        }
                    }
                }
            } else {
                let data = read_bytes(&self.reader, pos, msg_size as usize)?;
                cb(msg_type, &data)?;
            }

            pos += msg_size;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal: read group links (from root group OHDR)
    // -----------------------------------------------------------------------

    fn read_group_links_static(
        reader: &RefCell<BufReader<std::fs::File>>,
        ohdr_addr: u64,
    ) -> WrfResult<HashMap<String, u64>> {
        let mut links: HashMap<String, u64> = HashMap::new();
        let mut link_info_heap: Option<u64> = None;
        let mut link_info_bt2: Option<u64> = None;
        let mut link_info_bt2_co: Option<u64> = None;

        // Parse OHDR manually
        let sig = read_bytes(reader, ohdr_addr, 4)?;
        if sig != OHDR_SIGNATURE {
            return Err(hdf5_err(format!("Expected OHDR at 0x{ohdr_addr:x}")));
        }
        let flags = read_u8_at(reader, ohdr_addr + 5)?;
        let mut pos = ohdr_addr + 6;
        if flags & 0x20 != 0 { pos += 16; } // bit 5: times
        if flags & 0x10 != 0 { pos += 4; }  // bit 4: phase change
        let size_of_chunk_size = 1u64 << (flags & 0x03);
        let chunk_size = match size_of_chunk_size {
            1 => read_u8_at(reader, pos)? as u64,
            2 => read_u16(reader, pos)? as u64,
            4 => read_u32(reader, pos)? as u64,
            8 => read_u64(reader, pos)?,
            _ => return Err(hdf5_err("bad chunk size")),
        };
        pos += size_of_chunk_size;
        let chunk_end = pos + chunk_size;
        let msg_has_co = (flags & 0x04) != 0;
        let msg_hdr = if msg_has_co { 6u64 } else { 4u64 };

        // Recursive message parsing with continuation support
        fn parse_msgs(
            reader: &RefCell<BufReader<std::fs::File>>,
            start: u64,
            end: u64,
            msg_hdr: u64,
            links: &mut HashMap<String, u64>,
            link_info_heap: &mut Option<u64>,
            link_info_bt2: &mut Option<u64>,
            link_info_bt2_co: &mut Option<u64>,
        ) -> WrfResult<()> {
            let mut p = start;
            while p + msg_hdr <= end {
                let mt = read_u8_at(reader, p)?;
                let ms = read_u16(reader, p + 1)? as u64;
                let _mf = read_u8_at(reader, p + 3)?;
                p += msg_hdr;
                if mt == 0 && ms == 0 { break; }
                if p + ms > end { break; }

                match mt {
                    MSG_LINK => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        if let Ok((name, addr)) = parse_link_message(&data) {
                            links.insert(name, addr);
                        }
                    }
                    MSG_LINK_INFO => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        parse_link_info_message(
                            &data, link_info_heap, link_info_bt2, link_info_bt2_co,
                        );
                    }
                    MSG_CONTINUATION => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        if data.len() >= 16 {
                            let ca = le_u64(&data[0..8]);
                            let cl = le_u64(&data[8..16]);
                            if ca != UNDEF_ADDR && cl > 0 {
                                let cs = read_bytes(reader, ca, 4)?;
                                if &cs == b"OCHK" {
                                    parse_msgs(
                                        reader, ca + 4, ca + cl - 4, msg_hdr,
                                        links, link_info_heap, link_info_bt2, link_info_bt2_co,
                                    )?;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                p += ms;
            }
            Ok(())
        }

        parse_msgs(
            reader, pos, chunk_end, msg_hdr,
            &mut links, &mut link_info_heap, &mut link_info_bt2, &mut link_info_bt2_co,
        )?;

        // If we have dense link storage, resolve via fractal heap + btree v2
        if let (Some(heap_addr), Some(bt2_addr)) = (link_info_heap, link_info_bt2) {
            let dense_links = read_dense_links(reader, heap_addr, bt2_addr)?;
            links.extend(dense_links);
        }

        Ok(links)
    }

    // -----------------------------------------------------------------------
    // Internal: read global attributes from root OHDR
    // -----------------------------------------------------------------------

    fn read_attributes_static(
        reader: &RefCell<BufReader<std::fs::File>>,
        ohdr_addr: u64,
    ) -> WrfResult<HashMap<String, HdfAttributeValue>> {
        let mut attrs: HashMap<String, HdfAttributeValue> = HashMap::new();
        let mut attr_info_heap: Option<u64> = None;
        let mut attr_info_bt2: Option<u64> = None;

        let sig = read_bytes(reader, ohdr_addr, 4)?;
        if sig != OHDR_SIGNATURE {
            return Err(hdf5_err(format!("Expected OHDR at 0x{ohdr_addr:x}")));
        }
        let flags = read_u8_at(reader, ohdr_addr + 5)?;
        let mut pos = ohdr_addr + 6;
        if flags & 0x20 != 0 { pos += 16; } // bit 5: times
        if flags & 0x10 != 0 { pos += 4; }  // bit 4: phase change
        let scs = 1u64 << (flags & 0x03);
        let chunk_size = match scs {
            1 => read_u8_at(reader, pos)? as u64,
            2 => read_u16(reader, pos)? as u64,
            4 => read_u32(reader, pos)? as u64,
            8 => read_u64(reader, pos)?,
            _ => return Err(hdf5_err("bad chunk size")),
        };
        pos += scs;
        let chunk_end = pos + chunk_size;
        let msg_has_co = (flags & 0x04) != 0;
        let mh = if msg_has_co { 6u64 } else { 4u64 };

        fn parse_attr_msgs(
            reader: &RefCell<BufReader<std::fs::File>>,
            start: u64, end: u64, mh: u64,
            attrs: &mut HashMap<String, HdfAttributeValue>,
            attr_info_heap: &mut Option<u64>,
            attr_info_bt2: &mut Option<u64>,
        ) -> WrfResult<()> {
            let mut p = start;
            while p + mh <= end {
                let mt = read_u8_at(reader, p)?;
                let ms = read_u16(reader, p + 1)? as u64;
                let _mf = read_u8_at(reader, p + 3)?;
                p += mh;
                if mt == 0 && ms == 0 { break; }
                if p + ms > end { break; }

                match mt {
                    MSG_ATTRIBUTE => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        if let Ok((name, val)) = parse_attribute_message(&data) {
                            attrs.insert(name, val);
                        }
                    }
                    MSG_ATTR_INFO => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        parse_attr_info_message(&data, attr_info_heap, attr_info_bt2);
                    }
                    MSG_CONTINUATION => {
                        let data = read_bytes(reader, p, ms as usize)?;
                        if data.len() >= 16 {
                            let ca = le_u64(&data[0..8]);
                            let cl = le_u64(&data[8..16]);
                            if ca != UNDEF_ADDR && cl > 0 {
                                let cs = read_bytes(reader, ca, 4)?;
                                if &cs == b"OCHK" {
                                    parse_attr_msgs(
                                        reader, ca + 4, ca + cl - 4, mh,
                                        attrs, attr_info_heap, attr_info_bt2,
                                    )?;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                p += ms;
            }
            Ok(())
        }

        parse_attr_msgs(
            reader, pos, chunk_end, mh,
            &mut attrs, &mut attr_info_heap, &mut attr_info_bt2,
        )?;

        // Dense attribute storage
        if let (Some(heap_addr), Some(bt2_addr)) = (attr_info_heap, attr_info_bt2) {
            let dense_attrs = read_dense_attributes(reader, heap_addr, bt2_addr)?;
            attrs.extend(dense_attrs);
        }

        Ok(attrs)
    }

    // -----------------------------------------------------------------------
    // Internal: read raw dataset data (contiguous or chunked)
    // -----------------------------------------------------------------------

    fn read_raw_data(&self, _name: &str, info: &DatasetInfo) -> WrfResult<Vec<u8>> {
        match &info.layout {
            Layout::Contiguous { addr, size } => {
                if *addr == UNDEF_ADDR || *size == 0 {
                    return Ok(Vec::new());
                }
                read_bytes(&self.reader, *addr, *size as usize)
            }
            Layout::Chunked { addr, chunk_dims, ndims } => {
                if *addr == UNDEF_ADDR {
                    return Ok(Vec::new());
                }
                self.read_chunked_data(
                    *addr, &info.shape, chunk_dims, *ndims, &info.dtype, &info.filters,
                )
            }
        }
    }

    fn read_chunked_data(
        &self,
        btree_addr: u64,
        shape: &[usize],
        chunk_dims: &[u32],
        ndims: u8,
        dtype: &DType,
        filters: &[Filter],
    ) -> WrfResult<Vec<u8>> {
        let elem_size = dtype.size();
        let total_elems: usize = shape.iter().product();
        let total_bytes = total_elems * elem_size;
        let mut output = vec![0u8; total_bytes];

        // Chunk dimensions (skip the last one which is element size in the btree key)
        let chunk_shape: Vec<usize> = chunk_dims.iter().map(|&d| d as usize).collect();
        let chunk_elems: usize = chunk_shape.iter().product();
        let chunk_bytes = chunk_elems * elem_size;

        // Collect all chunks from B-tree v1
        let mut chunks: Vec<(Vec<u64>, u64, u32, u32)> = Vec::new();
        self.collect_btree_v1_chunks(btree_addr, ndims, &mut chunks)?;

        for (offsets, chunk_addr, compressed_size, filter_mask) in &chunks {
            if *chunk_addr == UNDEF_ADDR {
                continue;
            }

            let compressed = read_bytes(&self.reader, *chunk_addr, *compressed_size as usize)?;

            // Decompress
            let decompressed = if *filter_mask == 0 && !filters.is_empty() {
                decompress_chunk(&compressed, filters, chunk_bytes)?
            } else {
                // No filters or filter_mask says skip all
                compressed
            };

            // Copy to output at correct position
            let ndim = shape.len();
            let chunk_offsets: Vec<usize> =
                offsets.iter().take(ndim).map(|&o| o as usize).collect();

            // Copy element by element along the chunk, handling edge chunks
            copy_chunk_to_output(
                &decompressed, &mut output, shape, &chunk_shape, &chunk_offsets, elem_size,
            );
        }

        Ok(output)
    }

    fn collect_btree_v1_chunks(
        &self,
        addr: u64,
        ndims: u8,
        chunks: &mut Vec<(Vec<u64>, u64, u32, u32)>,
    ) -> WrfResult<()> {
        let sig = read_bytes(&self.reader, addr, 4)?;
        if sig != TREE_SIGNATURE {
            return Err(hdf5_err(format!("Expected TREE at 0x{addr:x}, got {sig:?}")));
        }

        let node_type = read_u8_at(&self.reader, addr + 4)?;
        if node_type != 1 {
            return Err(hdf5_err(format!(
                "Expected B-tree v1 node_type=1 (raw data chunks), got {node_type}"
            )));
        }
        let _node_level = read_u8_at(&self.reader, addr + 5)?;
        let entries_used = read_u16(&self.reader, addr + 6)? as usize;
        let _left_sibling = read_u64(&self.reader, addr + 8)?;
        let _right_sibling = read_u64(&self.reader, addr + 16)?;

        // Key size: 4 (chunk_size) + 4 (filter_mask) + ndims*8 (offsets)
        let key_size = 4 + 4 + (ndims as usize) * 8;
        let data_start = addr + 24; // after header

        let node_level = _node_level;
        if node_level == 0 {
            // Leaf node
            for i in 0..entries_used {
                let key_off = data_start + (i as u64) * (key_size as u64 + 8);
                let key_data = read_bytes(&self.reader, key_off, key_size)?;
                let chunk_size = le_u32(&key_data[0..4]);
                let filter_mask = le_u32(&key_data[4..8]);
                let mut offsets = Vec::new();
                for d in 0..(ndims as usize) {
                    offsets.push(le_u64(&key_data[8 + d * 8..]));
                }
                let child_addr = read_u64(&self.reader, key_off + key_size as u64)?;
                chunks.push((offsets, child_addr, chunk_size, filter_mask));
            }
        } else {
            // Internal node: recurse into children
            for i in 0..entries_used {
                let key_off = data_start + (i as u64) * (key_size as u64 + 8);
                let child_addr = read_u64(&self.reader, key_off + key_size as u64)?;
                self.collect_btree_v1_chunks(child_addr, ndims, chunks)?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Message parsers
// ---------------------------------------------------------------------------

fn parse_dataspace(data: &[u8]) -> WrfResult<Vec<usize>> {
    if data.is_empty() { return Ok(Vec::new()); }
    let version = data[0];
    let ndims = data[1] as usize;
    let _flags = data[2];
    let dim_start = if version == 1 { 8 } else { 4 };
    let mut dims = Vec::with_capacity(ndims);
    for i in 0..ndims {
        let off = dim_start + i * 8;
        if off + 8 > data.len() { break; }
        dims.push(le_u64(&data[off..]) as usize);
    }
    Ok(dims)
}

fn parse_datatype(data: &[u8]) -> WrfResult<DType> {
    if data.len() < 8 {
        return Err(hdf5_err("Datatype message too short"));
    }
    let class_and_version = data[0];
    let class = class_and_version & 0x0F;
    let size = le_u32(&data[4..8]);

    match class {
        0 => {
            // Fixed-point (integer)
            match size {
                4 => Ok(DType::I32),
                1 => Ok(DType::U8),
                _ => Err(hdf5_err(format!("Unsupported integer size {size}"))),
            }
        }
        1 => {
            // Floating-point
            match size {
                4 => Ok(DType::F32),
                8 => Ok(DType::F64),
                _ => Err(hdf5_err(format!("Unsupported float size {size}"))),
            }
        }
        3 => {
            // String type - treat as U8
            Ok(DType::U8)
        }
        _ => Err(hdf5_err(format!("Unsupported datatype class {class}"))),
    }
}

fn parse_layout(data: &[u8]) -> WrfResult<Layout> {
    if data.len() < 2 {
        return Err(hdf5_err("Layout message too short"));
    }
    let version = data[0];
    let layout_class = data[1];

    match (version, layout_class) {
        (3, 1) => {
            // Contiguous, version 3: addr(8) + size(8)
            if data.len() < 18 {
                return Err(hdf5_err("Contiguous layout too short"));
            }
            let addr = le_u64(&data[2..10]);
            let size = le_u64(&data[10..18]);
            Ok(Layout::Contiguous { addr, size })
        }
        (3, 2) => {
            // Chunked, version 3: dimensionality(1), addr(8), chunk_dims(ndims * 4)
            let ndims = data[2]; // includes +1 for element size
            let addr = le_u64(&data[3..11]);
            let num_chunk_dims = (ndims - 1) as usize; // exclude element size pseudo-dim
            let mut chunk_dims = Vec::with_capacity(num_chunk_dims);
            for i in 0..num_chunk_dims {
                let off = 11 + i * 4;
                if off + 4 > data.len() { break; }
                chunk_dims.push(le_u32(&data[off..]));
            }
            Ok(Layout::Chunked { addr, chunk_dims, ndims })
        }
        (4, 1) => {
            // Contiguous, version 4
            if data.len() < 18 {
                return Err(hdf5_err("Contiguous layout v4 too short"));
            }
            let addr = le_u64(&data[2..10]);
            let size = le_u64(&data[10..18]);
            Ok(Layout::Contiguous { addr, size })
        }
        (4, 2) => {
            // Chunked, version 4
            let _flags = data[2];
            let ndims = data[3];
            let _dim_size_encoded = data[4];
            let num_chunk_dims = ndims as usize;
            let mut chunk_dims = Vec::with_capacity(num_chunk_dims);
            let mut off = 5;
            for _ in 0..num_chunk_dims {
                if off + 4 > data.len() { break; }
                chunk_dims.push(le_u32(&data[off..]));
                off += 4;
            }
            let _idx_type = if off < data.len() { data[off] } else { 0 };
            off += 1;
            let addr = if off + 8 <= data.len() {
                le_u64(&data[off..])
            } else {
                UNDEF_ADDR
            };
            // Layout v4 chunked uses ndims differently (no +1)
            // Add 1 to ndims for btree key compatibility
            Ok(Layout::Chunked { addr, chunk_dims, ndims: ndims + 1 })
        }
        _ => {
            // Compact (class 0) or unknown
            Ok(Layout::Contiguous { addr: UNDEF_ADDR, size: 0 })
        }
    }
}

fn parse_filters(data: &[u8]) -> WrfResult<Vec<Filter>> {
    if data.len() < 2 { return Ok(Vec::new()); }
    let version = data[0];
    let nfilters = data[1] as usize;
    let mut filters = Vec::new();
    let mut pos = if version == 1 { 8 } else { 2 }; // v1 has 6 reserved bytes

    for _ in 0..nfilters {
        if pos + 2 > data.len() { break; }
        let filter_id = le_u16(&data[pos..]);
        pos += 2;

        if version == 1 || (version == 2 && filter_id >= 256) {
            // name_length present
            let name_len = le_u16(&data[pos..]) as usize;
            pos += 2;
            let _flags = le_u16(&data[pos..]);
            pos += 2;
            let nparams = le_u16(&data[pos..]) as usize;
            pos += 2;
            // name (null-terminated, padded to 8 bytes)
            if name_len > 0 {
                let padded = (name_len + 7) & !7;
                pos += padded;
            }
            // parameters
            let mut params = Vec::new();
            for _ in 0..nparams {
                if pos + 4 > data.len() { break; }
                params.push(le_u32(&data[pos..]));
                pos += 4;
            }
            // v1: pad to even number of params
            if version == 1 && nparams % 2 != 0 {
                pos += 4;
            }

            match filter_id {
                1 => filters.push(Filter::Deflate {
                    level: params.first().copied().unwrap_or(6),
                }),
                2 => filters.push(Filter::Shuffle {
                    element_size: params.first().copied().unwrap_or(4),
                }),
                _ => {} // skip unknown filters
            }
        } else {
            // version 2 with id < 256: no name
            let _flags = le_u16(&data[pos..]);
            pos += 2;
            let nparams = le_u16(&data[pos..]) as usize;
            pos += 2;
            let mut params = Vec::new();
            for _ in 0..nparams {
                if pos + 4 > data.len() { break; }
                params.push(le_u32(&data[pos..]));
                pos += 4;
            }
            match filter_id {
                1 => filters.push(Filter::Deflate {
                    level: params.first().copied().unwrap_or(6),
                }),
                2 => filters.push(Filter::Shuffle {
                    element_size: params.first().copied().unwrap_or(4),
                }),
                _ => {}
            }
        }
    }
    Ok(filters)
}

fn parse_link_message(data: &[u8]) -> WrfResult<(String, u64)> {
    if data.len() < 3 {
        return Err(hdf5_err("Link message too short"));
    }
    let _version = data[0];
    let flags = data[1];
    let mut pos = 2;

    // Optional link type (if bit 3 of flags set)
    let link_type = if flags & 0x08 != 0 {
        let lt = data[pos];
        pos += 1;
        lt
    } else {
        0 // hard link
    };

    // Optional creation order (if bit 2 of flags set)
    if flags & 0x04 != 0 {
        pos += 8;
    }

    // Optional link name charset (if bit 4 of flags set)
    if flags & 0x10 != 0 {
        pos += 1;
    }

    // Name length: depends on bits 0-1
    let name_len_size = 1 << (flags & 0x03);
    let name_len = match name_len_size {
        1 => { let v = data[pos] as usize; pos += 1; v }
        2 => { let v = le_u16(&data[pos..]) as usize; pos += 2; v }
        4 => { let v = le_u32(&data[pos..]) as usize; pos += 4; v }
        8 => { let v = le_u64(&data[pos..]) as usize; pos += 8; v }
        _ => return Err(hdf5_err("bad name length size")),
    };

    if pos + name_len > data.len() {
        return Err(hdf5_err("Link name extends past data"));
    }
    let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
    pos += name_len;

    // Link value: for hard links, it's an 8-byte address
    if link_type == 0 {
        if pos + 8 > data.len() {
            return Err(hdf5_err("Hard link address missing"));
        }
        let addr = le_u64(&data[pos..]);
        Ok((name, addr))
    } else {
        Err(hdf5_err("Only hard links supported"))
    }
}

fn parse_link_info_message(
    data: &[u8],
    heap: &mut Option<u64>,
    bt2_name: &mut Option<u64>,
    bt2_co: &mut Option<u64>,
) {
    if data.len() < 2 { return; }
    let _version = data[0];
    let flags = data[1];
    let mut pos = 2;

    // If bit 0 set, creation order is tracked: max_creation_index (8 bytes)
    if flags & 0x01 != 0 {
        pos += 8;
    }

    if pos + 8 <= data.len() {
        let fh_addr = le_u64(&data[pos..]);
        if fh_addr != UNDEF_ADDR { *heap = Some(fh_addr); }
        pos += 8;
    }
    if pos + 8 <= data.len() {
        let bt2_addr = le_u64(&data[pos..]);
        if bt2_addr != UNDEF_ADDR { *bt2_name = Some(bt2_addr); }
        pos += 8;
    }
    if flags & 0x01 != 0 && pos + 8 <= data.len() {
        let bt2_co_addr = le_u64(&data[pos..]);
        if bt2_co_addr != UNDEF_ADDR { *bt2_co = Some(bt2_co_addr); }
    }
}

fn parse_attr_info_message(
    data: &[u8],
    heap: &mut Option<u64>,
    bt2: &mut Option<u64>,
) {
    if data.len() < 2 { return; }
    let _version = data[0];
    let flags = data[1];
    let mut pos = 2;

    // If bit 0: Maximum Creation Index present (2 bytes)
    if flags & 0x01 != 0 {
        pos += 2; // max_creation_index (u16)
    }

    if pos + 8 <= data.len() {
        let fh_addr = le_u64(&data[pos..]);
        if fh_addr != UNDEF_ADDR { *heap = Some(fh_addr); }
        pos += 8;
    }
    if pos + 8 <= data.len() {
        let bt2_addr = le_u64(&data[pos..]);
        if bt2_addr != UNDEF_ADDR { *bt2 = Some(bt2_addr); }
    }
}

fn parse_attribute_message(data: &[u8]) -> WrfResult<(String, HdfAttributeValue)> {
    if data.len() < 6 {
        return Err(hdf5_err("Attribute message too short"));
    }
    let version = data[0];
    if version < 1 || version > 3 {
        return Err(hdf5_err(format!("Unsupported attribute version {version}")));
    }

    let _flags = if version >= 2 { data[1] } else { 0 };
    let name_size = le_u16(&data[2..4]) as usize;
    let datatype_size = le_u16(&data[4..6]) as usize;
    let dataspace_size = le_u16(&data[6..8]) as usize;
    let _encoding = if version >= 3 { data[8] } else { 0 };

    let header_size = if version >= 3 { 9 } else { 8 };
    let mut pos = header_size;

    // Name (null-terminated)
    if pos + name_size > data.len() {
        return Err(hdf5_err("Attribute name overflow"));
    }
    let name_bytes = &data[pos..pos + name_size];
    let name = String::from_utf8_lossy(name_bytes).trim_end_matches('\0').to_string();
    pos += name_size;

    // v1 pads to 8-byte boundary
    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Datatype
    if pos + datatype_size > data.len() {
        return Err(hdf5_err("Attribute datatype overflow"));
    }
    let dt_data = &data[pos..pos + datatype_size];
    let dtype = parse_datatype(dt_data).unwrap_or(DType::U8);
    pos += datatype_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Dataspace
    if pos + dataspace_size > data.len() {
        return Err(hdf5_err("Attribute dataspace overflow"));
    }
    let ds_data = &data[pos..pos + dataspace_size];
    let dims = parse_dataspace(ds_data).unwrap_or_default();
    pos += dataspace_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Data
    let total_elems: usize = if dims.is_empty() { 1 } else { dims.iter().product() };
    let data_bytes = total_elems * dtype.size();
    let remaining = &data[pos..];

    let val = match dtype {
        DType::F32 => {
            if remaining.len() >= 4 {
                HdfAttributeValue {
                    f32_val: Some(f32::from_le_bytes(remaining[0..4].try_into().unwrap())),
                    i32_val: None, string_val: None, f64_val: None,
                }
            } else {
                return Err(hdf5_err("F32 attr data too short"));
            }
        }
        DType::F64 => {
            if remaining.len() >= 8 {
                HdfAttributeValue {
                    f64_val: Some(f64::from_le_bytes(remaining[0..8].try_into().unwrap())),
                    f32_val: None, i32_val: None, string_val: None,
                }
            } else {
                return Err(hdf5_err("F64 attr data too short"));
            }
        }
        DType::I32 => {
            if remaining.len() >= 4 {
                HdfAttributeValue {
                    i32_val: Some(i32::from_le_bytes(remaining[0..4].try_into().unwrap())),
                    f32_val: None, string_val: None, f64_val: None,
                }
            } else {
                return Err(hdf5_err("I32 attr data too short"));
            }
        }
        DType::U8 => {
            let end = data_bytes.min(remaining.len());
            let s = String::from_utf8_lossy(&remaining[..end])
                .trim_end_matches('\0')
                .to_string();
            HdfAttributeValue {
                string_val: Some(s), f32_val: None, i32_val: None, f64_val: None,
            }
        }
    };

    Ok((name, val))
}

// ---------------------------------------------------------------------------
// Dense link storage: Fractal Heap + B-tree v2
// ---------------------------------------------------------------------------

fn read_dense_links(
    reader: &RefCell<BufReader<std::fs::File>>,
    heap_addr: u64,
    bt2_addr: u64,
) -> WrfResult<HashMap<String, u64>> {
    let mut links = HashMap::new();

    let heap_info = parse_fractal_heap_header(reader, heap_addr)?;
    let heap_ids = enumerate_btree_v2(reader, bt2_addr)?;

    for heap_id in &heap_ids {
        if let Ok(link_data) = resolve_heap_id(reader, &heap_info, heap_id) {
            if let Ok((name, addr)) = parse_link_message(&link_data) {
                links.insert(name, addr);
            }
        }
    }

    Ok(links)
}

fn read_dense_attributes(
    reader: &RefCell<BufReader<std::fs::File>>,
    heap_addr: u64,
    bt2_addr: u64,
) -> WrfResult<HashMap<String, HdfAttributeValue>> {
    let mut attrs = HashMap::new();

    let heap_info = parse_fractal_heap_header(reader, heap_addr)?;
    let heap_ids = enumerate_btree_v2(reader, bt2_addr)?;

    for heap_id in &heap_ids {
        if let Ok(attr_data) = resolve_heap_id(reader, &heap_info, heap_id) {
            if let Ok((name, val)) = parse_attribute_message(&attr_data) {
                attrs.insert(name, val);
            }
        }
    }

    Ok(attrs)
}

// ---------------------------------------------------------------------------
// Fractal Heap
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct FractalHeapInfo {
    root_block_addr: u64,
    starting_block_size: u64,
    max_direct_block_size: u64,
    table_width: u16,
    #[allow(dead_code)]
    root_rows: u16,
    nrows: u16,
    has_io_filters: bool,
    #[allow(dead_code)]
    io_filter_size: u64,
    max_heap_size: u16,
    #[allow(dead_code)]
    id_len: u16,
    starting_row_indirect: u32,
    row_block_sizes: Vec<u64>,
    #[allow(dead_code)]
    managed_obj_count: u64,
    #[allow(dead_code)]
    managed_space: u64,
    heap_off_bits: usize,
    heap_len_bits: usize,
}

fn parse_fractal_heap_header(
    reader: &RefCell<BufReader<std::fs::File>>,
    addr: u64,
) -> WrfResult<FractalHeapInfo> {
    let sig = read_bytes(reader, addr, 4)?;
    if sig != FRHP_SIGNATURE {
        return Err(hdf5_err(format!("Expected FRHP at 0x{addr:x}")));
    }
    let _version = read_u8_at(reader, addr + 4)?;
    let heap_id_len = read_u16(reader, addr + 5)?;
    let io_filter_len = read_u16(reader, addr + 7)?;
    let _flags = read_u8_at(reader, addr + 9)?;

    let _max_managed_obj_size = read_u32(reader, addr + 10)?;
    let _next_huge_id = read_u64(reader, addr + 14)?;
    let _bt2_huge_addr = read_u64(reader, addr + 22)?;
    let _managed_free_space = read_u64(reader, addr + 30)?;
    let _free_space_manager_addr = read_u64(reader, addr + 38)?;
    let managed_space = read_u64(reader, addr + 46)?;
    let _alloc_managed_space = read_u64(reader, addr + 54)?;
    let _iter_offset = read_u64(reader, addr + 62)?;
    let managed_obj_count = read_u64(reader, addr + 70)?;
    let _huge_obj_size = read_u64(reader, addr + 78)?;
    let _huge_obj_count = read_u64(reader, addr + 86)?;
    let _tiny_obj_size = read_u64(reader, addr + 94)?;
    let _tiny_obj_count = read_u64(reader, addr + 102)?;

    let table_width = read_u16(reader, addr + 110)?;
    let starting_block_size = read_u64(reader, addr + 112)?;
    let max_direct_block_size = read_u64(reader, addr + 120)?;
    let max_heap_size = read_u16(reader, addr + 128)?;
    let start_nrows = read_u16(reader, addr + 130)?;

    let root_block_addr = read_u64(reader, addr + 132)?;
    let cur_nrows = read_u16(reader, addr + 140)?;

    let has_io_filters = io_filter_len > 0;
    let io_filter_size = if has_io_filters {
        read_u64(reader, addr + 142)?
    } else {
        0
    };

    // Compute block sizes per row
    let max_rows = max_heap_size as usize;
    let mut row_block_sizes = Vec::with_capacity(max_rows);
    let mut bs = starting_block_size;
    for r in 0..max_rows {
        row_block_sizes.push(bs);
        if r > 0 {
            bs = bs.saturating_mul(2);
        }
    }

    // Heap ID bit layout
    let total_id_bits = (heap_id_len as usize - 1) * 8;
    let heap_off_bits = max_heap_size as usize;
    let heap_len_bits = total_id_bits - heap_off_bits;

    // Starting row for indirect blocks
    let mut starting_row_indirect = 0u32;
    {
        let mut check_bs = starting_block_size;
        for r in 0..max_rows {
            if check_bs > max_direct_block_size {
                starting_row_indirect = r as u32;
                break;
            }
            if r > 0 { check_bs *= 2; }
        }
        if starting_row_indirect == 0 && starting_block_size <= max_direct_block_size {
            let mut bs2 = starting_block_size;
            for r in 0..max_rows {
                if bs2 > max_direct_block_size {
                    starting_row_indirect = r as u32;
                    break;
                }
                if r > 0 { bs2 *= 2; }
            }
            if starting_row_indirect == 0 {
                starting_row_indirect = max_rows as u32;
            }
        }
    }

    Ok(FractalHeapInfo {
        root_block_addr,
        starting_block_size,
        max_direct_block_size,
        table_width,
        root_rows: start_nrows,
        nrows: cur_nrows,
        has_io_filters,
        io_filter_size,
        max_heap_size,
        id_len: heap_id_len,
        starting_row_indirect,
        row_block_sizes,
        managed_obj_count,
        managed_space,
        heap_off_bits,
        heap_len_bits,
    })
}

fn resolve_heap_id(
    reader: &RefCell<BufReader<std::fs::File>>,
    heap: &FractalHeapInfo,
    id: &[u8],
) -> WrfResult<Vec<u8>> {
    if id.is_empty() {
        return Err(hdf5_err("Empty heap ID"));
    }

    let managed_type = (id[0] >> 6) & 0x03;
    if managed_type != 0 {
        return Err(hdf5_err(format!("Non-managed heap ID type {managed_type}")));
    }

    let remaining = &id[1..];
    let (offset, length) =
        extract_heap_id_offset_length(remaining, heap.heap_off_bits, heap.heap_len_bits);

    if length == 0 {
        return Err(hdf5_err("Zero-length heap ID"));
    }

    resolve_managed_object(reader, heap, offset, length as usize)
}

fn extract_heap_id_offset_length(data: &[u8], off_bits: usize, len_bits: usize) -> (u64, u64) {
    let mut val: u128 = 0;
    for (i, &b) in data.iter().enumerate() {
        val |= (b as u128) << (i * 8);
    }

    let offset = val & ((1u128 << off_bits) - 1);
    let length = (val >> off_bits) & ((1u128 << len_bits) - 1);

    (offset as u64, length as u64)
}

fn resolve_managed_object(
    reader: &RefCell<BufReader<std::fs::File>>,
    heap: &FractalHeapInfo,
    offset: u64,
    length: usize,
) -> WrfResult<Vec<u8>> {
    if heap.nrows == 0 {
        read_from_direct_block(
            reader, heap.root_block_addr, heap, offset, length, heap.starting_block_size,
        )
    } else {
        read_from_indirect_block(
            reader, heap.root_block_addr, heap, offset, length, heap.nrows as usize,
        )
    }
}

fn read_from_direct_block(
    reader: &RefCell<BufReader<std::fs::File>>,
    block_addr: u64,
    _heap: &FractalHeapInfo,
    local_offset: u64,
    length: usize,
    _block_size: u64,
) -> WrfResult<Vec<u8>> {
    if block_addr == UNDEF_ADDR {
        return Err(hdf5_err("Direct block at undefined address"));
    }

    let sig = read_bytes(reader, block_addr, 4)?;
    if sig != FHDB_SIGNATURE {
        return Err(hdf5_err(format!("Expected FHDB at 0x{block_addr:x}, got {sig:?}")));
    }

    read_bytes(reader, block_addr + local_offset, length)
}

fn read_from_indirect_block(
    reader: &RefCell<BufReader<std::fs::File>>,
    block_addr: u64,
    heap: &FractalHeapInfo,
    offset: u64,
    length: usize,
    nrows: usize,
) -> WrfResult<Vec<u8>> {
    if block_addr == UNDEF_ADDR {
        return Err(hdf5_err("Indirect block at undefined address"));
    }

    let sig = read_bytes(reader, block_addr, 4)?;
    if sig != FHIB_SIGNATURE {
        // Could be a direct block if nrows is small
        if sig == FHDB_SIGNATURE {
            return read_from_direct_block(
                reader, block_addr, heap, offset, length, heap.starting_block_size,
            );
        }
        return Err(hdf5_err(format!("Expected FHIB at 0x{block_addr:x}, got {sig:?}")));
    }

    // FHIB header: sig(4) + version(1) + heap_header_addr(8) + block_offset(variable)
    let block_offset_size = ((heap.max_heap_size as usize) + 7) / 8;
    let header_size = 4 + 1 + 8 + block_offset_size;

    // Read the block_offset of this indirect block
    let iblock_offset_pos = block_addr + 5 + 8;
    let iblock_bo_bytes = read_bytes(reader, iblock_offset_pos, block_offset_size)?;
    let mut iblock_base_offset: u64 = 0;
    for i in 0..block_offset_size {
        iblock_base_offset |= (iblock_bo_bytes[i] as u64) << (i * 8);
    }

    // Count direct block entries and indirect block entries
    let ndirect_rows = nrows.min(heap.starting_row_indirect as usize);
    let nindirect_rows = if nrows > heap.starting_row_indirect as usize {
        nrows - heap.starting_row_indirect as usize
    } else {
        0
    };

    let entry_size = if heap.has_io_filters { 8 + 8 + 4 } else { 8 };
    let mut pos = block_addr + header_size as u64;

    let mut cum_offset: u64 = iblock_base_offset;

    // Check direct blocks
    for row in 0..ndirect_rows {
        let block_size = heap.row_block_sizes.get(row).copied()
            .unwrap_or(heap.starting_block_size);

        for _col in 0..heap.table_width {
            let child_addr = read_u64(reader, pos)?;
            pos += entry_size as u64;

            if child_addr != UNDEF_ADDR
                && offset >= cum_offset
                && offset < cum_offset + block_size
            {
                let local_offset = offset - cum_offset;
                return read_from_direct_block(
                    reader, child_addr, heap, local_offset, length, block_size,
                );
            }
            cum_offset += block_size;
        }
    }

    // Check indirect blocks
    for row in 0..nindirect_rows {
        let actual_row = heap.starting_row_indirect as usize + row;
        let child_nrows = actual_row;
        let child_capacity = indirect_block_capacity(heap, child_nrows);

        for _col in 0..heap.table_width {
            let child_addr = read_u64(reader, pos)?;
            pos += 8;

            if child_addr != UNDEF_ADDR
                && offset >= cum_offset
                && offset < cum_offset + child_capacity
            {
                return read_from_indirect_block(
                    reader, child_addr, heap, offset, length, child_nrows,
                );
            }
            cum_offset += child_capacity;
        }
    }

    Err(hdf5_err(format!(
        "Heap offset {offset} not found in indirect block at 0x{block_addr:x}"
    )))
}

fn indirect_block_capacity(heap: &FractalHeapInfo, nrows: usize) -> u64 {
    let ndirect = nrows.min(heap.starting_row_indirect as usize);
    let mut total: u64 = 0;
    for row in 0..ndirect {
        let bs = heap.row_block_sizes.get(row).copied()
            .unwrap_or(heap.starting_block_size);
        total += bs * heap.table_width as u64;
    }
    total
}

// ---------------------------------------------------------------------------
// B-tree v2 enumeration
// ---------------------------------------------------------------------------

fn enumerate_btree_v2(
    reader: &RefCell<BufReader<std::fs::File>>,
    addr: u64,
) -> WrfResult<Vec<Vec<u8>>> {
    let sig = read_bytes(reader, addr, 4)?;
    if sig != BTHD_SIGNATURE {
        return Err(hdf5_err(format!("Expected BTHD at 0x{addr:x}")));
    }

    let _version = read_u8_at(reader, addr + 4)?;
    let record_type = read_u8_at(reader, addr + 5)?;
    let node_size = read_u32(reader, addr + 6)?;
    let record_size = read_u16(reader, addr + 10)?;
    let depth = read_u16(reader, addr + 12)?;
    let root_node_addr = read_u64(reader, addr + 16)?;
    let num_records_root = read_u16(reader, addr + 24)?;
    let total_records = read_u64(reader, addr + 26)?;

    if root_node_addr == UNDEF_ADDR || total_records == 0 {
        return Ok(Vec::new());
    }

    let mut heap_ids = Vec::new();
    enumerate_btree_v2_node(
        reader, root_node_addr, depth, record_type, record_size,
        node_size, num_records_root as u32, &mut heap_ids,
    )?;

    Ok(heap_ids)
}

fn enumerate_btree_v2_node(
    reader: &RefCell<BufReader<std::fs::File>>,
    addr: u64,
    depth: u16,
    record_type: u8,
    record_size: u16,
    node_size: u32,
    num_records: u32,
    heap_ids: &mut Vec<Vec<u8>>,
) -> WrfResult<()> {
    if depth == 0 {
        // Leaf node (BTLF)
        let sig = read_bytes(reader, addr, 4)?;
        if sig != BTLF_SIGNATURE {
            return Err(hdf5_err(format!("Expected BTLF at 0x{addr:x}")));
        }
        let mut pos = addr + 6;

        for _ in 0..num_records {
            let rec = read_bytes(reader, pos, record_size as usize)?;
            let heap_id = match record_type {
                5 => rec[4..].to_vec(),
                6 => rec[8..].to_vec(),
                8 => {
                    let id_len = record_size as usize - 9;
                    rec[..id_len].to_vec()
                }
                9 => {
                    let id_len = record_size as usize - 5;
                    rec[..id_len].to_vec()
                }
                _ => rec.clone(),
            };

            if heap_id.iter().any(|&b| b != 0) {
                heap_ids.push(heap_id);
            }
            pos += record_size as u64;
        }
    } else {
        // Internal node (BTIN)
        let sig = read_bytes(reader, addr, 4)?;
        if &sig != b"BTIN" {
            return Err(hdf5_err(format!("Expected BTIN at 0x{addr:x}, got {sig:?}")));
        }

        let max_leaf_records = (node_size as usize - 10) / record_size as usize;
        let nrec_size: u64 = if max_leaf_records <= 255 { 1 } else { 2 };
        let total_nrec_size: u64 = if depth > 1 { 2 } else { 0 };
        let child_ptr_size = 8 + nrec_size + total_nrec_size;

        let records_start = addr + 6;
        let children_start = records_start + (num_records as u64) * (record_size as u64);

        let num_children = num_records + 1;
        let mut children: Vec<(u64, u32)> = Vec::new();
        for i in 0..num_children {
            let cp_off = children_start + (i as u64) * child_ptr_size;
            let child_addr = read_u64(reader, cp_off)?;
            let child_nrec = if nrec_size == 1 {
                read_u8_at(reader, cp_off + 8)? as u32
            } else {
                read_u16(reader, cp_off + 8)? as u32
            };
            children.push((child_addr, child_nrec));
        }

        // Process records stored in this internal node
        for i in 0..num_records {
            let rec_off = records_start + (i as u64) * (record_size as u64);
            let rec = read_bytes(reader, rec_off, record_size as usize)?;
            let heap_id = match record_type {
                5 => rec[4..].to_vec(),
                6 => rec[8..].to_vec(),
                8 => {
                    let id_len = record_size as usize - 9;
                    rec[..id_len].to_vec()
                }
                9 => {
                    let id_len = record_size as usize - 5;
                    rec[..id_len].to_vec()
                }
                _ => rec.clone(),
            };
            if heap_id.iter().any(|&b| b != 0) {
                heap_ids.push(heap_id);
            }
        }

        for (child_addr, child_nrec) in children {
            if child_addr != UNDEF_ADDR {
                enumerate_btree_v2_node(
                    reader, child_addr, depth - 1, record_type,
                    record_size, node_size, child_nrec, heap_ids,
                )?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Decompression
// ---------------------------------------------------------------------------

fn decompress_chunk(data: &[u8], filters: &[Filter], expected_size: usize) -> WrfResult<Vec<u8>> {
    let mut buf = data.to_vec();

    // Apply filters in reverse pipeline order
    for filter in filters.iter().rev() {
        match filter {
            Filter::Deflate { .. } => {
                let mut decoder = ZlibDecoder::new(&buf[..]);
                let mut decompressed = Vec::with_capacity(expected_size);
                decoder.read_to_end(&mut decompressed).map_err(io_err)?;
                buf = decompressed;
            }
            Filter::Shuffle { element_size } => {
                buf = unshuffle(&buf, *element_size as usize);
            }
        }
    }

    Ok(buf)
}

fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let n = data.len();
    let num_elements = n / element_size;
    let mut output = vec![0u8; n];

    for i in 0..num_elements {
        for b in 0..element_size {
            output[i * element_size + b] = data[b * num_elements + i];
        }
    }

    // Copy any trailing bytes that don't fill a complete element
    let complete = num_elements * element_size;
    if complete < n {
        output[complete..].copy_from_slice(&data[complete..]);
    }

    output
}

// ---------------------------------------------------------------------------
// Chunk assembly
// ---------------------------------------------------------------------------

fn copy_chunk_to_output(
    chunk_data: &[u8],
    output: &mut [u8],
    shape: &[usize],
    chunk_shape: &[usize],
    chunk_offsets: &[usize],
    elem_size: usize,
) {
    let ndim = shape.len();
    if ndim == 0 { return; }

    // Compute strides for the output array (row-major / C order)
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * shape[d + 1];
    }

    // Compute strides for the chunk
    let mut chunk_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        chunk_strides[d] = chunk_strides[d + 1] * chunk_shape[d + 1];
    }

    // Iterate over all elements in the chunk
    let chunk_total: usize = chunk_shape.iter().product();
    let mut indices = vec![0usize; ndim];

    for i in 0..chunk_total {
        // Compute multi-dimensional index within chunk
        let mut rem = i;
        for d in 0..ndim {
            indices[d] = rem / chunk_strides[d];
            rem %= chunk_strides[d];
        }

        // Check bounds: global index must be within shape
        let mut in_bounds = true;
        let mut out_linear = 0usize;
        for d in 0..ndim {
            let global = chunk_offsets[d] + indices[d];
            if global >= shape[d] {
                in_bounds = false;
                break;
            }
            out_linear += global * out_strides[d];
        }

        if in_bounds {
            let src_start = i * elem_size;
            let dst_start = out_linear * elem_size;
            if src_start + elem_size <= chunk_data.len()
                && dst_start + elem_size <= output.len()
            {
                output[dst_start..dst_start + elem_size]
                    .copy_from_slice(&chunk_data[src_start..src_start + elem_size]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unshuffle_roundtrip() {
        let original = vec![
            0x01, 0x02, 0x03, 0x04,
            0x11, 0x12, 0x13, 0x14,
            0x21, 0x22, 0x23, 0x24,
            0x31, 0x32, 0x33, 0x34,
        ];
        let shuffled = vec![
            0x01, 0x11, 0x21, 0x31,
            0x02, 0x12, 0x22, 0x32,
            0x03, 0x13, 0x23, 0x33,
            0x04, 0x14, 0x24, 0x34,
        ];
        assert_eq!(unshuffle(&shuffled, 4), original);
    }

    #[test]
    fn test_le_helpers() {
        assert_eq!(le_u16(&[0x01, 0x02]), 0x0201);
        assert_eq!(le_u32(&[0x01, 0x02, 0x03, 0x04]), 0x04030201);
    }
}
