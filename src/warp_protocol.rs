//! W.A.R.P. Network Protocol Definition
//!
//! Binary wire format for multi-node P2P KV block transfers.
//!
//! # Frame Layout
//! [Magic:4][Version:2][Features:8][BodyLen:4][Body...]
//!
//! # Handshake (Frame 0)
//! [GGUF_Hash:32][Model_Name_Len:2][Model_Name...]

use std::io::{self, Read, Write};

pub const WARP_MAGIC: [u8; 4] = *b"WARP";
pub const WARP_VERSION: u16 = 0x0101; // v1.1.0

/// Feature bitmask for capability negotiation (Decision Q22).
pub enum WarpFeature {
    Lz4Compression   = 1 << 0,
    Int8Quantization = 1 << 1,
    GhostPruning     = 1 << 2,
    AsyncPipelining  = 1 << 3,
}

#[derive(Debug, Clone)]
pub struct WarpHeader {
    pub features: u64,
    pub body_len: u32,
}

impl WarpHeader {
    /// Serialise to Little-Endian bytes (Decision Q18).
    pub fn serialise(&self) -> [u8; 18] {
        let mut buf = [0u8; 18];
        buf[0..4].copy_from_slice(&WARP_MAGIC);
        buf[4..6].copy_from_slice(&WARP_VERSION.to_le_bytes());
        buf[6..14].copy_from_slice(&self.features.to_le_bytes());
        buf[14..18].copy_from_slice(&self.body_len.to_le_bytes());
        buf
    }

    pub fn deserialise<R: Read>(mut r: R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != WARP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WARP magic"));
        }

        let mut version_buf = [0u8; 2];
        r.read_exact(&mut version_buf)?;
        let version = u16::from_le_bytes(version_buf);
        if version >> 8 != WARP_VERSION >> 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Incompatible WARP major version"));
        }

        let mut features_buf = [0u8; 8];
        r.read_exact(&mut features_buf)?;
        let features = u64::from_le_bytes(features_buf);

        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let body_len = u32::from_le_bytes(len_buf);

        Ok(Self { features, body_len })
    }
}
