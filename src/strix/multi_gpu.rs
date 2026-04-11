//! Multi-GPU management and NVLink peer-to-peer transfers.
//!
//! From STRIX Protocol §12 (HAL) and hardware_implementation_guide.md Task 3.
//!
//! Provides:
//! - `GpuTopology` — runtime multi-GPU discovery + NVLink detection
//! - `ShardStrategy` — tensor distribution across GPUs
//! - `PeerTransfer` — cross-GPU tensor copy descriptor
//!
//! ## Architecture
//!
//! ```text
//! Single GPU:   Storage → RAM → GPU0 VRAM
//! Multi-GPU:    Storage → RAM → GPU0 VRAM ←─ NVLink ─→ GPU1 VRAM
//!                                                        ↕ NVLink
//!                                                      GPU2 VRAM
//! ```
//!
//! CUDA peer access FFI is feature-gated behind `#[cfg(feature = "cuda")]`.
//! Without the feature, topology defaults to a single synthetic device.

use super::hal::{GpuInfo, HalError};
use super::types::GpuPtr;
use std::fmt;

// ── CUDA Peer Access FFI ─────────────────────────────────────────────────

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceCanAccessPeer(canAccess: *mut i32, device: i32, peer: i32) -> i32;
    fn cudaDeviceEnablePeerAccess(peer: i32, flags: u32) -> i32;
    fn cudaDeviceDisablePeerAccess(peer: i32) -> i32;
    fn cudaMemcpyPeerAsync(
        dst: *mut u8,
        dst_device: i32,
        src: *const u8,
        src_device: i32,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

// ── GPU Topology ─────────────────────────────────────────────────────────

/// Interconnect type between two GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interconnect {
    /// No direct access — must go through host RAM.
    HostBridge,
    /// PCIe peer-to-peer (slow — ~32 GB/s).
    PciePeer,
    /// NVLink v3 (~600 GB/s bidirectional).
    NvLinkV3,
    /// NVLink v4 (~900 GB/s bidirectional).
    NvLinkV4,
}

impl Interconnect {
    /// Estimated unidirectional bandwidth in GB/s.
    pub fn bandwidth_gbps(&self) -> f64 {
        match self {
            Self::HostBridge => 12.0,   // PCIe 3.0 x16 via host
            Self::PciePeer   => 32.0,   // PCIe 4.0 x16 direct
            Self::NvLinkV3   => 300.0,  // half of 600 bidirectional
            Self::NvLinkV4   => 450.0,  // half of 900 bidirectional
        }
    }
}

impl fmt::Display for Interconnect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostBridge => write!(f, "Host Bridge (PCIe)"),
            Self::PciePeer   => write!(f, "PCIe Peer-to-Peer"),
            Self::NvLinkV3   => write!(f, "NVLink v3"),
            Self::NvLinkV4   => write!(f, "NVLink v4"),
        }
    }
}

/// Multi-GPU topology information.
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Number of GPUs detected.
    pub gpu_count: usize,
    /// Per-GPU information.
    pub gpus: Vec<GpuInfo>,
    /// Peer access matrix: `peer_access[i][j]` = true if GPU i can access GPU j.
    pub peer_access: Vec<Vec<bool>>,
    /// Interconnect type between each pair.
    pub interconnect: Vec<Vec<Interconnect>>,
    /// NVLink bandwidth between peers in GB/s (0 if no NVLink).
    pub nvlink_bandwidth: Vec<Vec<f64>>,
}

impl GpuTopology {
    /// Discover the GPU topology using CUDA APIs.
    ///
    /// Returns a topology with peer access enabled for all connected pairs.
    /// Falls back to single-GPU topology if CUDA is unavailable.
    #[cfg(feature = "cuda")]
    pub fn discover() -> Result<Self, HalError> {
        let mut count: i32 = 0;
        let code = unsafe { cudaGetDeviceCount(&mut count) };
        if code != 0 || count <= 0 {
            return Err(HalError::Unsupported("No CUDA devices found".into()));
        }

        let gpu_count = count as usize;
        let mut peer_access = vec![vec![false; gpu_count]; gpu_count];
        let mut nvlink_bandwidth = vec![vec![0.0; gpu_count]; gpu_count];
        let mut interconnect = vec![vec![Interconnect::HostBridge; gpu_count]; gpu_count];

        // Probe peer access for all pairs.
        for i in 0..gpu_count {
            for j in 0..gpu_count {
                if i == j { continue; }
                let mut can_access: i32 = 0;
                unsafe {
                    cudaSetDevice(i as i32);
                    cudaDeviceCanAccessPeer(&mut can_access, i as i32, j as i32);
                }
                peer_access[i][j] = can_access != 0;
                if can_access != 0 {
                    // Conservative default: NVLink v3 (~600 GB/s bidirectional).
                    // Real version detection would query NVLink topology via NVML.
                    nvlink_bandwidth[i][j] = 600.0;
                    interconnect[i][j] = Interconnect::NvLinkV3;
                }
            }
        }

        // Enable peer access for all connected pairs.
        for i in 0..gpu_count {
            for j in 0..gpu_count {
                if peer_access[i][j] {
                    unsafe {
                        cudaSetDevice(i as i32);
                        // Ignore error — may already be enabled.
                        cudaDeviceEnablePeerAccess(j as i32, 0);
                    }
                }
            }
        }

        Ok(Self {
            gpu_count,
            gpus: Vec::new(), // Populated by per-GPU info queries
            peer_access,
            interconnect,
            nvlink_bandwidth,
        })
    }

    /// Fallback topology when CUDA is not available.
    #[cfg(not(feature = "cuda"))]
    pub fn discover() -> Result<Self, HalError> {
        Ok(Self::single_gpu())
    }

    /// Create a single-GPU topology (no peer access, no NVLink).
    pub fn single_gpu() -> Self {
        Self {
            gpu_count: 1,
            gpus: Vec::new(),
            peer_access: vec![vec![false]],
            interconnect: vec![vec![Interconnect::HostBridge]],
            nvlink_bandwidth: vec![vec![0.0]],
        }
    }

    /// Create a topology from known parameters (for testing).
    pub fn from_params(
        gpu_count: usize,
        peer_access: Vec<Vec<bool>>,
        nvlink_bandwidth: Vec<Vec<f64>>,
    ) -> Self {
        let interconnect = peer_access.iter().enumerate().map(|(i, row)| {
            row.iter().enumerate().map(|(j, &has_peer)| {
                if i == j { Interconnect::HostBridge }
                else if has_peer && nvlink_bandwidth[i][j] > 500.0 { Interconnect::NvLinkV3 }
                else if has_peer { Interconnect::PciePeer }
                else { Interconnect::HostBridge }
            }).collect()
        }).collect();

        Self {
            gpu_count,
            gpus: Vec::new(),
            peer_access,
            interconnect,
            nvlink_bandwidth,
        }
    }

    /// Whether any NVLink connections exist.
    pub fn has_nvlink(&self) -> bool {
        self.interconnect.iter().any(|row| {
            row.iter().any(|ic| matches!(ic, Interconnect::NvLinkV3 | Interconnect::NvLinkV4))
        })
    }

    /// Whether any peer access exists (NVLink or PCIe).
    pub fn has_peer_access(&self) -> bool {
        self.peer_access.iter().any(|row| row.iter().any(|&v| v))
    }

    /// Total VRAM across all GPUs.
    pub fn total_vram(&self) -> usize {
        self.gpus.iter().map(|g| g.vram_total).sum()
    }

    /// Best interconnect between two GPUs.
    pub fn link_between(&self, gpu_a: usize, gpu_b: usize) -> Interconnect {
        if gpu_a >= self.gpu_count || gpu_b >= self.gpu_count {
            return Interconnect::HostBridge;
        }
        self.interconnect[gpu_a][gpu_b]
    }
}

impl fmt::Display for GpuTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Topology: {} devices", self.gpu_count)?;
        for i in 0..self.gpu_count {
            for j in 0..self.gpu_count {
                if i == j { continue; }
                if self.peer_access[i][j] {
                    writeln!(f, "  GPU{} ←→ GPU{}: {} ({:.0} GB/s)",
                        i, j, self.interconnect[i][j], self.nvlink_bandwidth[i][j])?;
                }
            }
        }
        if !self.has_peer_access() {
            writeln!(f, "  No peer access (single-GPU or PCIe via host)")?;
        }
        Ok(())
    }
}

// ── Shard Strategy ───────────────────────────────────────────────────────

/// Strategy for distributing model tensors across GPUs.
#[derive(Debug, Clone)]
pub enum ShardStrategy {
    /// All tensors on one GPU (single-GPU mode).
    SingleGpu(usize),
    /// Layer-parallel: each GPU holds a contiguous block of layers.
    ///
    /// `assignments[gpu_idx] = (layer_start, layer_end_exclusive)`
    LayerParallel {
        /// Per-GPU layer range.
        assignments: Vec<(usize, usize)>,
    },
    /// Tensor-parallel: each tensor is split across GPUs along a dimension.
    TensorParallel {
        /// Number of GPUs to split across.
        world_size: usize,
    },
}

impl ShardStrategy {
    /// Compute layer-parallel assignments for a model.
    ///
    /// Distributes `total_layers` evenly across GPUs. Extra layers are
    /// distributed to the first GPUs (round-robin remainder).
    pub fn layer_parallel(total_layers: usize, topology: &GpuTopology) -> Self {
        let n = topology.gpu_count;
        if n <= 1 {
            return Self::SingleGpu(0);
        }

        let layers_per_gpu = total_layers / n;
        let remainder = total_layers % n;

        let mut assignments = Vec::with_capacity(n);
        let mut start = 0;
        for i in 0..n {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + layers_per_gpu + extra;
            assignments.push((start, end));
            start = end;
        }

        Self::LayerParallel { assignments }
    }

    /// Which GPU holds a given layer.
    pub fn gpu_for_layer(&self, layer: usize) -> usize {
        match self {
            Self::SingleGpu(gpu) => *gpu,
            Self::LayerParallel { assignments } => {
                assignments.iter()
                    .position(|&(start, end)| layer >= start && layer < end)
                    .unwrap_or(0)
            }
            Self::TensorParallel { .. } => {
                // All GPUs hold all layers (split by dim) — use GPU 0 for scheduling.
                0
            }
        }
    }

    /// Whether this is single-GPU mode.
    pub fn is_single_gpu(&self) -> bool {
        matches!(self, Self::SingleGpu(_))
    }

    /// Number of GPUs used by this strategy.
    pub fn gpu_count(&self) -> usize {
        match self {
            Self::SingleGpu(_) => 1,
            Self::LayerParallel { assignments } => assignments.len(),
            Self::TensorParallel { world_size } => *world_size,
        }
    }
}

impl fmt::Display for ShardStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingleGpu(gpu) => write!(f, "Single GPU (device {})", gpu),
            Self::LayerParallel { assignments } => {
                write!(f, "Layer-Parallel ({} GPUs): ", assignments.len())?;
                for (i, &(start, end)) in assignments.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "GPU{}=[{}-{})", i, start, end)?;
                }
                Ok(())
            }
            Self::TensorParallel { world_size } => {
                write!(f, "Tensor-Parallel ({} GPUs)", world_size)
            }
        }
    }
}

// ── Peer Transfer ────────────────────────────────────────────────────────

/// A pending cross-GPU tensor transfer.
#[derive(Debug, Clone)]
pub struct PeerTransfer {
    /// Source GPU index.
    pub src_gpu: usize,
    /// Source VRAM pointer.
    pub src_ptr: GpuPtr,
    /// Destination GPU index.
    pub dst_gpu: usize,
    /// Destination VRAM pointer.
    pub dst_ptr: GpuPtr,
    /// Transfer size in bytes.
    pub size: usize,
    /// Transfer status.
    pub status: PeerTransferStatus,
    /// Interconnect used.
    pub interconnect: Interconnect,
}

/// Status of a peer transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerTransferStatus {
    /// Not yet submitted.
    Pending,
    /// In flight.
    InFlight,
    /// Completed successfully.
    Completed,
    /// Failed — fall back to host staging.
    Failed,
}

impl PeerTransfer {
    /// Create a new peer transfer descriptor.
    pub fn new(
        src_gpu: usize,
        src_ptr: GpuPtr,
        dst_gpu: usize,
        dst_ptr: GpuPtr,
        size: usize,
        topology: &GpuTopology,
    ) -> Self {
        let interconnect = topology.link_between(src_gpu, dst_gpu);
        Self {
            src_gpu, src_ptr, dst_gpu, dst_ptr, size,
            status: PeerTransferStatus::Pending,
            interconnect,
        }
    }

    /// Execute the peer copy via CUDA `cudaMemcpyPeerAsync`.
    #[cfg(feature = "cuda")]
    pub fn execute(&mut self) -> Result<(), HalError> {
        self.status = PeerTransferStatus::InFlight;
        let result = unsafe {
            cudaMemcpyPeerAsync(
                self.dst_ptr.0 as *mut u8,
                self.dst_gpu as i32,
                self.src_ptr.0 as *const u8,
                self.src_gpu as i32,
                self.size,
                std::ptr::null_mut(), // default stream
            )
        };
        if result != 0 {
            self.status = PeerTransferStatus::Failed;
            return Err(HalError::DriverError {
                code: result,
                message: format!("cudaMemcpyPeerAsync failed: GPU{}→GPU{}", self.src_gpu, self.dst_gpu),
            });
        }
        self.status = PeerTransferStatus::Completed;
        Ok(())
    }

    /// Fallback execute — always fails (no CUDA).
    #[cfg(not(feature = "cuda"))]
    pub fn execute(&mut self) -> Result<(), HalError> {
        self.status = PeerTransferStatus::Failed;
        Err(HalError::Unsupported("Peer copy requires CUDA feature".into()))
    }

    /// Estimated transfer time in ms.
    pub fn estimated_time_ms(&self) -> f64 {
        let size_gb = self.size as f64 / (1024.0 * 1024.0 * 1024.0);
        let bw = self.interconnect.bandwidth_gbps();
        if bw > 0.0 { (size_gb / bw) * 1000.0 } else { f64::INFINITY }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Topology ──────────────────────────────────────────────────

    #[test]
    fn single_gpu_topology() {
        let topo = GpuTopology::single_gpu();
        assert_eq!(topo.gpu_count, 1);
        assert!(!topo.has_nvlink());
        assert!(!topo.has_peer_access());
    }

    #[test]
    fn dual_gpu_nvlink_topology() {
        let topo = GpuTopology::from_params(
            2,
            vec![vec![false, true], vec![true, false]],
            vec![vec![0.0, 600.0], vec![600.0, 0.0]],
        );
        assert_eq!(topo.gpu_count, 2);
        assert!(topo.has_nvlink());
        assert!(topo.has_peer_access());
        assert!(topo.peer_access[0][1]);
        assert!(topo.peer_access[1][0]);
        assert_eq!(topo.link_between(0, 1), Interconnect::NvLinkV3);
    }

    #[test]
    fn four_gpu_topology() {
        let n = 4;
        let peer = vec![
            vec![false, true, true, true],
            vec![true, false, true, true],
            vec![true, true, false, true],
            vec![true, true, true, false],
        ];
        let bw = vec![
            vec![0.0, 600.0, 600.0, 600.0],
            vec![600.0, 0.0, 600.0, 600.0],
            vec![600.0, 600.0, 0.0, 600.0],
            vec![600.0, 600.0, 600.0, 0.0],
        ];
        let topo = GpuTopology::from_params(n, peer, bw);
        assert_eq!(topo.gpu_count, 4);
        assert!(topo.has_nvlink());
    }

    #[test]
    fn topology_display() {
        let topo = GpuTopology::from_params(
            2,
            vec![vec![false, true], vec![true, false]],
            vec![vec![0.0, 600.0], vec![600.0, 0.0]],
        );
        let s = format!("{}", topo);
        assert!(s.contains("2 devices"));
        assert!(s.contains("NVLink"));
    }

    #[test]
    fn link_between_bounds() {
        let topo = GpuTopology::single_gpu();
        assert_eq!(topo.link_between(0, 5), Interconnect::HostBridge);
        assert_eq!(topo.link_between(99, 0), Interconnect::HostBridge);
    }

    // ── Shard Strategy ───────────────────────────────────────────

    #[test]
    fn layer_parallel_even_split() {
        let topo = GpuTopology::from_params(
            2,
            vec![vec![false, true], vec![true, false]],
            vec![vec![0.0, 600.0], vec![600.0, 0.0]],
        );

        let strategy = ShardStrategy::layer_parallel(80, &topo);
        if let ShardStrategy::LayerParallel { assignments } = &strategy {
            assert_eq!(assignments[0], (0, 40));
            assert_eq!(assignments[1], (40, 80));
        } else {
            panic!("expected LayerParallel");
        }

        assert_eq!(strategy.gpu_for_layer(0), 0);
        assert_eq!(strategy.gpu_for_layer(39), 0);
        assert_eq!(strategy.gpu_for_layer(40), 1);
        assert_eq!(strategy.gpu_for_layer(79), 1);
    }

    #[test]
    fn layer_parallel_uneven_split() {
        let topo = GpuTopology::from_params(
            3,
            vec![vec![false; 3]; 3],
            vec![vec![0.0; 3]; 3],
        );
        let strategy = ShardStrategy::layer_parallel(80, &topo);
        if let ShardStrategy::LayerParallel { assignments } = &strategy {
            // 80 / 3 = 26 remainder 2 → first 2 GPUs get 27, last gets 26
            assert_eq!(assignments[0], (0, 27));
            assert_eq!(assignments[1], (27, 54));
            assert_eq!(assignments[2], (54, 80));
        } else {
            panic!("expected LayerParallel");
        }
        assert_eq!(strategy.gpu_for_layer(26), 0);
        assert_eq!(strategy.gpu_for_layer(27), 1);
        assert_eq!(strategy.gpu_for_layer(54), 2);
    }

    #[test]
    fn single_gpu_strategy() {
        let topo = GpuTopology::single_gpu();
        let strategy = ShardStrategy::layer_parallel(80, &topo);
        assert!(strategy.is_single_gpu());
        assert_eq!(strategy.gpu_count(), 1);
        assert_eq!(strategy.gpu_for_layer(0), 0);
        assert_eq!(strategy.gpu_for_layer(79), 0);
    }

    #[test]
    fn tensor_parallel_strategy() {
        let strategy = ShardStrategy::TensorParallel { world_size: 4 };
        assert_eq!(strategy.gpu_count(), 4);
        assert_eq!(strategy.gpu_for_layer(0), 0); // scheduling GPU
        assert!(!strategy.is_single_gpu());
    }

    #[test]
    fn strategy_display() {
        let strategy = ShardStrategy::SingleGpu(0);
        assert!(format!("{}", strategy).contains("Single GPU"));

        let strategy = ShardStrategy::TensorParallel { world_size: 4 };
        assert!(format!("{}", strategy).contains("Tensor-Parallel"));
    }

    // ── Peer Transfer ────────────────────────────────────────────

    #[test]
    fn peer_transfer_creation() {
        let topo = GpuTopology::from_params(
            2,
            vec![vec![false, true], vec![true, false]],
            vec![vec![0.0, 600.0], vec![600.0, 0.0]],
        );
        let transfer = PeerTransfer::new(0, GpuPtr(0x1000), 1, GpuPtr(0x2000), 4096, &topo);
        assert_eq!(transfer.status, PeerTransferStatus::Pending);
        assert_eq!(transfer.interconnect, Interconnect::NvLinkV3);
        assert_eq!(transfer.size, 4096);
    }

    #[test]
    fn peer_transfer_estimated_time() {
        let topo = GpuTopology::from_params(
            2,
            vec![vec![false, true], vec![true, false]],
            vec![vec![0.0, 600.0], vec![600.0, 0.0]],
        );
        // 1 GB over NVLinkV3 (300 GB/s) → ~3.3ms
        let transfer = PeerTransfer::new(
            0, GpuPtr(0x1000), 1, GpuPtr(0x2000),
            1024 * 1024 * 1024, &topo,
        );
        let time = transfer.estimated_time_ms();
        assert!(time > 2.0 && time < 5.0, "Expected ~3.3ms, got {:.1}ms", time);
    }

    #[test]
    fn peer_transfer_no_cuda() {
        let topo = GpuTopology::single_gpu();
        let mut transfer = PeerTransfer::new(0, GpuPtr(0x1000), 0, GpuPtr(0x2000), 4096, &topo);
        // Without CUDA feature, execute should fail gracefully
        if !cfg!(feature = "cuda") {
            let result = transfer.execute();
            assert!(result.is_err());
            assert_eq!(transfer.status, PeerTransferStatus::Failed);
        }
    }

    // ── Interconnect ─────────────────────────────────────────────

    #[test]
    fn interconnect_bandwidth() {
        assert!(Interconnect::NvLinkV4.bandwidth_gbps() > Interconnect::NvLinkV3.bandwidth_gbps());
        assert!(Interconnect::NvLinkV3.bandwidth_gbps() > Interconnect::PciePeer.bandwidth_gbps());
        assert!(Interconnect::PciePeer.bandwidth_gbps() > Interconnect::HostBridge.bandwidth_gbps());
    }

    #[test]
    fn interconnect_display() {
        let s = format!("{}", Interconnect::NvLinkV3);
        assert!(s.contains("NVLink"));
    }
}
