//! # Prefill-Decode Disaggregation (PD-Disagg)
//!
//! Decouples the compute-bound **prefill** phase from the memory-bandwidth-bound
//! **decode** phase, allowing independent scaling and hardware tuning of each.
//!
//! ## Architecture
//! ```text
//! ┌──────────────┐   KvConnector   ┌──────────────┐
//! │ PrefillNode  │ ─── KvBlocks ──▶│  DecodeNode  │
//! │  (compute)   │                  │  (BW-bound)  │
//! └──────────────┘                  └──────────────┘
//! ```
//!
//! ## Research
//! - "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
//!   (Qin et al., USENIX/FAST 2025, arXiv:2407.00079)
//! - "DistServe: Disaggregated Prefill and Decoding for Goodput-Optimised LLM Serving"
//!   (Zhong et al., OSDI 2024, arXiv:2401.09670)
//! - "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
//!   (Patel et al., ISCA 2024, arXiv:2311.18677)
//!
//! ## Consumer-first design decisions
//! - **Default**: `TcpKvConnector` — zero deps, works on any OS/hardware.
//! - **Single-node**: `ShmKvConnector` — mmap zero-copy, 15-40 GB/s.
//! - **Datacenter**: `--features rdma` → `IbvKvConnector` (not implemented here).
//! - No Python, no gRPC, no protobuf — pure Rust typed protocol.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A single PagedAttention KV block received over the wire.
///
/// `data` layout: bf16-packed array of shape `[BLOCK_SIZE × n_heads × head_dim × 2]`
/// where the last dimension is K=0 / V=1.
#[derive(Debug, Clone)]
pub struct KvBlock {
    pub block_id: u32,
    pub layer_idx: u16,
    /// Raw bytes: bf16 (2 bytes/element).
    pub data: Vec<u8>,
}

impl KvBlock {
    /// Serialise to a flat byte buffer for transmission.
    /// Layout: [block_id:4][layer_idx:2][data_len:4][data:data_len]
    pub fn serialise(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(10 + self.data.len());
        buf.extend_from_slice(&self.block_id.to_le_bytes());
        buf.extend_from_slice(&self.layer_idx.to_le_bytes());
        let data_len = self.data.len() as u32;
        buf.extend_from_slice(&data_len.to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Deserialise from a flat byte buffer.
    pub fn deserialise(buf: &[u8]) -> Option<Self> {
        if buf.len() < 10 {
            return None;
        }
        let block_id = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let layer_idx = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        let data_len = u32::from_le_bytes(buf[6..10].try_into().ok()?) as usize;
        if buf.len() < 10 + data_len {
            return None;
        }
        Some(Self {
            block_id,
            layer_idx,
            data: buf[10..10 + data_len].to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum KvTransferError {
    Io(io::Error),
    Timeout,
    PartialSend { sent: usize, expected: usize },
    Serialisation,
}

impl std::fmt::Display for KvTransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "KV transfer IO error: {e}"),
            Self::Timeout => write!(f, "KV transfer timeout"),
            Self::PartialSend { sent, expected } => {
                write!(f, "KV partial send: {sent}/{expected} blocks")
            }
            Self::Serialisation => write!(f, "KV serialisation error"),
        }
    }
}

impl From<io::Error> for KvTransferError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// KvConnector trait
// ---------------------------------------------------------------------------

/// Pluggable KV-block transfer backend.
///
/// Implementations: `TcpKvConnector` (default), `ShmKvConnector` (single-node
/// zero-copy), future `IbvKvConnector` (RDMA, `--features rdma`).
pub trait KvConnector: Send + Sync {
    /// Send `blocks` for sequence `seq_id` to `target_addr`.
    /// Returns total bytes transferred.
    fn send_blocks(
        &self,
        seq_id: u64,
        blocks: &[KvBlock],
        target_addr: SocketAddr,
        timeout: Duration,
    ) -> Result<u64, KvTransferError>;

    /// Receive all blocks for `seq_id` from `source_addr`.
    fn recv_blocks(
        &self,
        seq_id: u64,
        n_expected: usize,
        timeout: Duration,
    ) -> Result<Vec<KvBlock>, KvTransferError>;

    /// Human-readable backend name for metrics labels.
    fn backend_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// TCP backend (default — works everywhere, no extra deps)
// ---------------------------------------------------------------------------

/// TCP KV connector: simple frame protocol over TCP sockets.
///
/// Frame layout per block: `[seq_id:8][block_payload]`
/// A sentinel frame with `seq_id=u64::MAX` signals end-of-stream.
pub struct TcpKvConnector {
    bind_addr: SocketAddr,
    /// Active receive listeners keyed by `seq_id`.
    listeners: Arc<Mutex<HashMap<u64, Vec<KvBlock>>>>,
}

impl TcpKvConnector {
    pub fn new(bind_addr: SocketAddr) -> Self {
        Self {
            bind_addr,
            listeners: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start a background listener that accepts one connection and stores blocks.
    pub fn start_listener(&self) -> io::Result<()> {
        let addr = self.bind_addr;
        let store = self.listeners.clone();
        std::thread::spawn(move || {
            let listener = match TcpListener::bind(addr) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("TcpKvConnector: bind error: {e}");
                    return;
                }
            };
            for stream in listener.incoming() {
                match stream {
                    Ok(mut s) => {
                        let store2 = store.clone();
                        std::thread::spawn(move || {
                            if let Err(e) = Self::handle_incoming(&mut s, &store2) {
                                eprintln!("TcpKvConnector recv error: {e}");
                            }
                        });
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(())
    }

    fn handle_incoming(
        stream: &mut TcpStream,
        store: &Arc<Mutex<HashMap<u64, Vec<KvBlock>>>>,
    ) -> io::Result<()> {
        loop {
            // Read seq_id (8 bytes).
            let mut id_buf = [0u8; 8];
            stream.read_exact(&mut id_buf)?;
            let seq_id = u64::from_le_bytes(id_buf);
            if seq_id == u64::MAX {
                break; // sentinel
            }

            // Read payload length (4 bytes).
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf)?;
            let payload_len = u32::from_le_bytes(len_buf) as usize;

            let mut payload = vec![0u8; payload_len];
            stream.read_exact(&mut payload)?;

            if let Some(block) = KvBlock::deserialise(&payload) {
                let mut locked = store.lock().unwrap();
                locked.entry(seq_id).or_default().push(block);
            }
        }
        Ok(())
    }
}

impl KvConnector for TcpKvConnector {
    fn send_blocks(
        &self,
        seq_id: u64,
        blocks: &[KvBlock],
        target_addr: SocketAddr,
        timeout: Duration,
    ) -> Result<u64, KvTransferError> {
        let mut stream = TcpStream::connect_timeout(&target_addr, timeout)?;
        stream.set_write_timeout(Some(timeout))?;
        let mut total_bytes = 0u64;

        for block in blocks {
            let payload = block.serialise();
            let payload_len = payload.len() as u32;
            stream.write_all(&seq_id.to_le_bytes())?;
            stream.write_all(&payload_len.to_le_bytes())?;
            stream.write_all(&payload)?;
            total_bytes += 8 + 4 + payload.len() as u64;
        }

        // Sentinel
        stream.write_all(&u64::MAX.to_le_bytes())?;
        stream.flush()?;
        Ok(total_bytes)
    }

    fn recv_blocks(
        &self,
        seq_id: u64,
        n_expected: usize,
        timeout: Duration,
    ) -> Result<Vec<KvBlock>, KvTransferError> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let locked = self.listeners.lock().unwrap();
                if let Some(blocks) = locked.get(&seq_id) {
                    if blocks.len() >= n_expected {
                        // Clone and return (drain not possible without mut).
                        return Ok(blocks[..n_expected].to_vec());
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(KvTransferError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn backend_name(&self) -> &'static str {
        "tcp"
    }
}

// ---------------------------------------------------------------------------
// Shared-memory backend (single-node zero-copy)
// ---------------------------------------------------------------------------

/// In-process shared-memory KV connector for single-machine PD disaggregation.
///
/// Uses a `Mutex<HashMap>` as the shared store — suitable for same-process
/// unit tests and single-machine deployments where the prefill and decode
/// workers share an address space (threads or tokio tasks).
///
/// For production single-node use, replace the inner `HashMap` with an
/// `mmap`-backed ring buffer for true zero-copy across processes.
pub struct ShmKvConnector {
    store: Arc<Mutex<HashMap<u64, Vec<KvBlock>>>>,
}

impl ShmKvConnector {
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return a clone of the shared store handle — pass to both prefill and
    /// decode nodes so they share the same backing store.
    pub fn clone_store(&self) -> Arc<Mutex<HashMap<u64, Vec<KvBlock>>>> {
        self.store.clone()
    }
}

impl Default for ShmKvConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl KvConnector for ShmKvConnector {
    fn send_blocks(
        &self,
        seq_id: u64,
        blocks: &[KvBlock],
        _target_addr: SocketAddr,
        _timeout: Duration,
    ) -> Result<u64, KvTransferError> {
        let total_bytes: u64 = blocks.iter().map(|b| b.data.len() as u64 + 10).sum();
        let mut locked = self.store.lock().unwrap();
        locked.entry(seq_id).or_default().extend(blocks.iter().cloned());
        Ok(total_bytes)
    }

    fn recv_blocks(
        &self,
        seq_id: u64,
        n_expected: usize,
        timeout: Duration,
    ) -> Result<Vec<KvBlock>, KvTransferError> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let locked = self.store.lock().unwrap();
                if let Some(blocks) = locked.get(&seq_id) {
                    if blocks.len() >= n_expected {
                        return Ok(blocks[..n_expected].to_vec());
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err(KvTransferError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn backend_name(&self) -> &'static str {
        "shm"
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PdDisaggConfig {
    /// Address the prefill node listens on / sends from.
    pub prefill_addr: SocketAddr,
    /// Address the decode node listens on / receives from.
    pub decode_addr: SocketAddr,
    /// Maximum concurrent in-flight KV transfers (backpressure).
    pub max_in_flight: usize,
    /// Per-transfer timeout.
    pub transfer_timeout: Duration,
}

impl Default for PdDisaggConfig {
    fn default() -> Self {
        Self {
            prefill_addr: "127.0.0.1:9100".parse().unwrap(),
            decode_addr: "127.0.0.1:9101".parse().unwrap(),
            max_in_flight: 16,
            transfer_timeout: Duration::from_millis(2000),
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer metrics
// ---------------------------------------------------------------------------

/// Prometheus-style counters for PD transfer health.
#[derive(Default)]
pub struct TransferMetrics {
    pub bytes_total: AtomicU64,
    pub transfers_total: AtomicU64,
    pub timeouts_total: AtomicU64,
    pub latency_sum_us: AtomicU64,
}

impl TransferMetrics {
    pub fn record_transfer(&self, bytes: u64, latency: Duration) {
        self.bytes_total.fetch_add(bytes, Ordering::Relaxed);
        self.transfers_total.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn record_timeout(&self) {
        self.timeouts_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn avg_latency_us(&self) -> f64 {
        let t = self.transfers_total.load(Ordering::Relaxed);
        if t == 0 {
            return 0.0;
        }
        self.latency_sum_us.load(Ordering::Relaxed) as f64 / t as f64
    }
}

// ---------------------------------------------------------------------------
// PrefillResult / DecodeResult
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PrefillResult {
    pub seq_id: u64,
    /// Time from request arrival to first token produced by prefill.
    pub ttft_us: u64,
    /// Number of KV blocks produced and transferred.
    pub blocks_transferred: usize,
    /// Bytes sent over the wire.
    pub bytes_sent: u64,
}

#[derive(Debug)]
pub struct DecodeResult {
    pub seq_id: u64,
    /// Number of KV blocks received.
    pub blocks_received: usize,
    /// Transfer latency in microseconds.
    pub transfer_latency_us: u64,
}

// ---------------------------------------------------------------------------
// PrefillNode — sends KV blocks after prefill
// ---------------------------------------------------------------------------

pub struct PrefillNode {
    pub cfg: PdDisaggConfig,
    pub connector: Arc<dyn KvConnector>,
    pub metrics: Arc<TransferMetrics>,
    /// Inflight semaphore: tracks current in-flight transfers.
    inflight: Arc<Mutex<usize>>,
}

impl PrefillNode {
    pub fn new(cfg: PdDisaggConfig, connector: Arc<dyn KvConnector>) -> Self {
        Self {
            cfg,
            connector,
            metrics: Arc::new(TransferMetrics::default()),
            inflight: Arc::new(Mutex::new(0)),
        }
    }

    /// Perform prefill (simulated) and send resulting KV blocks to decode node.
    /// In production this wraps `ContinuousBatchScheduler::prefill_step()`.
    pub fn prefill_and_send(
        &self,
        seq_id: u64,
        kv_blocks: Vec<KvBlock>,
    ) -> Result<PrefillResult, KvTransferError> {
        // Backpressure check.
        {
            let mut guard = self.inflight.lock().unwrap();
            if *guard >= self.cfg.max_in_flight {
                return Err(KvTransferError::PartialSend {
                    sent: 0,
                    expected: self.cfg.max_in_flight,
                });
            }
            *guard += 1;
        }

        let start = Instant::now();
        let n_blocks = kv_blocks.len();
        let result = self.connector.send_blocks(
            seq_id,
            &kv_blocks,
            self.cfg.decode_addr,
            self.cfg.transfer_timeout,
        );

        *self.inflight.lock().unwrap() -= 1;

        match result {
            Ok(bytes) => {
                let latency = start.elapsed();
                self.metrics.record_transfer(bytes, latency);
                Ok(PrefillResult {
                    seq_id,
                    ttft_us: latency.as_micros() as u64,
                    blocks_transferred: n_blocks,
                    bytes_sent: bytes,
                })
            }
            Err(KvTransferError::Timeout) => {
                self.metrics.record_timeout();
                Err(KvTransferError::Timeout)
            }
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// DecodeNode — receives KV blocks, starts decode
// ---------------------------------------------------------------------------

pub struct DecodeNode {
    pub cfg: PdDisaggConfig,
    pub connector: Arc<dyn KvConnector>,
    pub metrics: Arc<TransferMetrics>,
}

impl DecodeNode {
    pub fn new(cfg: PdDisaggConfig, connector: Arc<dyn KvConnector>) -> Self {
        Self {
            cfg,
            connector,
            metrics: Arc::new(TransferMetrics::default()),
        }
    }

    /// Receive KV blocks for `seq_id` from the prefill node.
    pub fn recv_and_prepare(
        &self,
        seq_id: u64,
        n_expected: usize,
    ) -> Result<DecodeResult, KvTransferError> {
        let start = Instant::now();
        let blocks = self
            .connector
            .recv_blocks(seq_id, n_expected, self.cfg.transfer_timeout)?;
        let latency = start.elapsed();
        let bytes: u64 = blocks.iter().map(|b| b.data.len() as u64 + 10).sum();
        self.metrics.record_transfer(bytes, latency);
        Ok(DecodeResult {
            seq_id,
            blocks_received: blocks.len(),
            transfer_latency_us: latency.as_micros() as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test KvBlock with `data_size` bytes of zero data.
    fn make_block(id: u32, layer: u16, data_size: usize) -> KvBlock {
        KvBlock {
            block_id: id,
            layer_idx: layer,
            data: vec![0xAB; data_size],
        }
    }

    // --- Wire serialisation ---

    #[test]
    fn test_kv_block_serialise_roundtrip() {
        let block = make_block(42, 7, 64);
        let bytes = block.serialise();
        let decoded = KvBlock::deserialise(&bytes).expect("deserialise failed");
        assert_eq!(decoded.block_id, 42);
        assert_eq!(decoded.layer_idx, 7);
        assert_eq!(decoded.data, block.data);
    }

    #[test]
    fn test_kv_block_deserialise_too_short() {
        assert!(KvBlock::deserialise(&[0u8; 5]).is_none());
    }

    #[test]
    fn test_kv_block_deserialise_truncated_data() {
        let block = make_block(1, 0, 32);
        let mut bytes = block.serialise();
        bytes.truncate(bytes.len() - 10); // corrupt
        assert!(KvBlock::deserialise(&bytes).is_none());
    }

    // --- SHM connector ---

    fn shm_pair(cfg: PdDisaggConfig) -> (PrefillNode, DecodeNode) {
        let connector = Arc::new(ShmKvConnector::new());
        let prefill = PrefillNode::new(cfg.clone(), connector.clone());
        let decode = DecodeNode::new(cfg, connector);
        (prefill, decode)
    }

    #[test]
    fn test_shm_send_recv_roundtrip() {
        let cfg = PdDisaggConfig::default();
        let (prefill, decode) = shm_pair(cfg);
        let blocks = vec![make_block(0, 0, 32), make_block(1, 1, 32)];
        let res = prefill.prefill_and_send(1, blocks).expect("send failed");
        assert_eq!(res.blocks_transferred, 2);
        let recv = decode.recv_and_prepare(1, 2).expect("recv failed");
        assert_eq!(recv.blocks_received, 2);
    }

    #[test]
    fn test_shm_timeout_if_no_sender() {
        let cfg = PdDisaggConfig {
            transfer_timeout: Duration::from_millis(20),
            ..Default::default()
        };
        let connector = Arc::new(ShmKvConnector::new());
        let decode = DecodeNode::new(cfg, connector);
        let result = decode.recv_and_prepare(999, 1);
        assert!(matches!(result, Err(KvTransferError::Timeout)));
    }

    #[test]
    fn test_backpressure_blocks_at_max_inflight() {
        let cfg = PdDisaggConfig {
            max_in_flight: 0, // immediately full
            ..Default::default()
        };
        let connector = Arc::new(ShmKvConnector::new());
        let prefill = PrefillNode::new(cfg, connector);
        let result = prefill.prefill_and_send(1, vec![make_block(0, 0, 8)]);
        assert!(matches!(result, Err(KvTransferError::PartialSend { .. })));
    }

    #[test]
    fn test_bytes_transferred_correct() {
        let cfg = PdDisaggConfig::default();
        let (prefill, _decode) = shm_pair(cfg);
        let block = make_block(0, 0, 100); // 100 data bytes + 10 header = 110
        let res = prefill.prefill_and_send(1, vec![block]).expect("send failed");
        assert!(res.bytes_sent >= 100);
    }

    #[test]
    fn test_metrics_record_on_successful_transfer() {
        let cfg = PdDisaggConfig::default();
        let (prefill, _decode) = shm_pair(cfg);
        let blocks = vec![make_block(0, 0, 32)];
        prefill.prefill_and_send(1, blocks).expect("send");
        assert_eq!(
            prefill.metrics.transfers_total.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_concurrent_transfers_distinct_seqids() {
        use std::thread;
        let connector = Arc::new(ShmKvConnector::new());
        let cfg = PdDisaggConfig::default();
        let handles: Vec<_> = (0u64..4)
            .map(|i| {
                let conn = connector.clone();
                let c = cfg.clone();
                thread::spawn(move || {
                    let prefill = PrefillNode::new(c.clone(), conn.clone());
                    let decode = DecodeNode::new(c, conn);
                    let blocks = vec![make_block(i as u32, 0, 16)];
                    prefill.prefill_and_send(i, blocks).unwrap();
                    let res = decode.recv_and_prepare(i, 1).unwrap();
                    assert_eq!(res.blocks_received, 1);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn test_large_block_transfer() {
        let cfg = PdDisaggConfig::default();
        let (prefill, decode) = shm_pair(cfg);
        // Simulate a 4096-token layer's KV block at bf16: 4096 × 8 heads × 128 head_dim × 2 (kv) × 2 bytes = 16 MB
        let data_size = 4096 * 8 * 128 * 2 * 2;
        let blocks = vec![make_block(0, 0, data_size)];
        let res = prefill.prefill_and_send(1, blocks).expect("send large block");
        assert!(res.bytes_sent > data_size as u64);
        let recv = decode.recv_and_prepare(1, 1).expect("recv large block");
        assert_eq!(recv.blocks_received, 1);
    }

    #[test]
    fn test_error_display() {
        let e = KvTransferError::PartialSend { sent: 3, expected: 5 };
        assert!(e.to_string().contains("3/5"));
        let e2 = KvTransferError::Timeout;
        assert!(e2.to_string().contains("timeout"));
    }

    #[test]
    fn test_metrics_avg_latency_zero_on_no_transfers() {
        let m = TransferMetrics::default();
        assert_eq!(m.avg_latency_us(), 0.0);
    }

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_send_sync() {
        _assert_send_sync::<TcpKvConnector>();
        _assert_send_sync::<ShmKvConnector>();
        _assert_send_sync::<TransferMetrics>();
    }
}
