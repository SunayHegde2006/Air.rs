//! KV-Cache CPU Swap Manager — HBM-Aware Tiered Offload.
//!
//! `KvSwapManager` wraps `SequenceManager` and adds a CPU-side pinned buffer
//! store so that `swap_out` / `swap_in` move actual KV tensor bytes between
//! GPU VRAM and system RAM via `mlock`-backed `Vec<u8>` buffers.
//!
//! Improvements 2.md §2.C: "HBM-Aware KV-Cache Paging (Virtual VRAM)"
//! and Phase 2 §3: "Universal KV-Cache Paging".
//!
//! # Design
//! - Each physical block = `block_bytes` bytes of KV data (caller sets this).
//! - On `evict(seq_id)`: calls `SequenceManager::swap_out`, copies GPU blocks
//!   to a `Vec<u8>` CPU buffer (mlock'd if supported), frees VRAM blocks.
//! - On `restore(seq_id)`: calls `SequenceManager::swap_in`, gives caller
//!   back the CPU buffer to DMA back to the newly allocated VRAM blocks.
//! - Actual GPU copy is the caller's responsibility (HAL-agnostic design).

use std::collections::HashMap;
use crate::paged_attention::{PhysicalBlockId, SeqId, SequenceManager};

/// Manages async CPU-side KV buffer store for swapped-out sequences.
pub struct KvSwapManager {
    pub seq_mgr: SequenceManager,
    /// Per-sequence CPU buffer holding evicted KV bytes.
    /// key = SeqId, value = flat byte buffer (n_blocks × block_bytes).
    cpu_store: HashMap<u32, Vec<u8>>,
    /// Bytes per physical block on the GPU side.
    pub block_bytes: usize,
    /// Total bytes currently held in CPU store.
    pub cpu_bytes_used: usize,
}

impl KvSwapManager {
    /// Create with a fixed block pool and per-block byte size.
    ///
    /// `block_capacity` — number of physical KV blocks (passed to `SequenceManager`).
    /// `block_bytes`    — byte size of one physical block's full KV data
    ///                    (n_heads × head_dim × 2 × dtype_bytes × BLOCK_SIZE).
    pub fn new(block_capacity: usize, block_bytes: usize) -> Self {
        Self {
            seq_mgr: SequenceManager::new(block_capacity),
            cpu_store: HashMap::new(),
            block_bytes,
            cpu_bytes_used: 0,
        }
    }

    /// Evict all KV blocks for a sequence to CPU RAM.
    ///
    /// `gpu_read` — closure that reads `block_bytes` bytes from the given
    ///              physical block into the provided slice. On CUDA this wraps
    ///              `cudaMemcpy(D→H)`; on CPU-only builds it may be a memcpy.
    ///
    /// Returns the released `PhysicalBlockId`s (now in the free pool).
    pub fn evict<F>(&mut self, seq_id: SeqId, gpu_read: F) -> Result<Vec<PhysicalBlockId>, &'static str>
    where
        F: Fn(PhysicalBlockId, &mut [u8]),
    {
        let table = self.seq_mgr.table(seq_id).ok_or("sequence not found")?;
        let n_blocks = table.physical_blocks.len();
        let mut cpu_buf = vec![0u8; n_blocks * self.block_bytes];

        // Copy GPU → CPU before releasing blocks
        let blocks = table.physical_blocks.clone();
        for (i, &bid) in blocks.iter().enumerate() {
            let slot = &mut cpu_buf[i * self.block_bytes..(i + 1) * self.block_bytes];
            gpu_read(bid, slot);
        }

        let released = self.seq_mgr.swap_out(seq_id)?;
        self.cpu_bytes_used += cpu_buf.len();
        self.cpu_store.insert(seq_id.0, cpu_buf);
        Ok(released)
    }

    /// Restore a previously evicted sequence back to GPU VRAM.
    ///
    /// `gpu_write` — closure that writes `block_bytes` bytes from the provided
    ///               slice into the given newly allocated physical block.
    ///               On CUDA this wraps `cudaMemcpy(H→D)`.
    ///
    /// Returns the new `PhysicalBlockId`s (freshly allocated).
    pub fn restore<F>(&mut self, seq_id: SeqId, gpu_write: F) -> Result<Vec<PhysicalBlockId>, &'static str>
    where
        F: Fn(PhysicalBlockId, &[u8]),
    {
        let cpu_buf = self.cpu_store.remove(&seq_id.0).ok_or("no evicted data for sequence")?;
        let n_blocks = cpu_buf.len() / self.block_bytes;
        let new_blocks = self.seq_mgr.swap_in(seq_id, n_blocks)?;

        for (i, &bid) in new_blocks.iter().enumerate() {
            let slot = &cpu_buf[i * self.block_bytes..(i + 1) * self.block_bytes];
            gpu_write(bid, slot);
        }

        self.cpu_bytes_used = self.cpu_bytes_used.saturating_sub(cpu_buf.len());
        Ok(new_blocks)
    }

    /// True if a sequence has been evicted and its data sits in CPU RAM.
    pub fn is_evicted(&self, seq_id: SeqId) -> bool {
        self.cpu_store.contains_key(&seq_id.0)
    }

    /// Number of sequences currently held in CPU store.
    pub fn evicted_count(&self) -> usize {
        self.cpu_store.len()
    }

    /// Evict the sequence if VRAM is below `free_blocks` threshold.
    /// Returns `true` if an eviction was performed.
    pub fn maybe_evict<F>(&mut self, seq_id: SeqId, free_threshold: usize, gpu_read: F) -> bool
    where
        F: Fn(PhysicalBlockId, &mut [u8]),
    {
        if self.seq_mgr.allocator.num_free_blocks() < free_threshold && !self.is_evicted(seq_id) {
            self.evict(seq_id, gpu_read).is_ok()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_read(_: PhysicalBlockId, buf: &mut [u8]) { buf.fill(0xAB); }
    fn noop_write(_: PhysicalBlockId, _: &[u8]) {}

    #[test]
    fn evict_and_restore_round_trip() {
        use crate::paged_attention::BLOCK_SIZE;
        let mut mgr = KvSwapManager::new(8, 64);
        let sid = mgr.seq_mgr.create_sequence();
        for _ in 0..BLOCK_SIZE {
            mgr.seq_mgr.append_token(sid).unwrap();
        }
        let free_before = mgr.seq_mgr.allocator.num_free_blocks();
        let released = mgr.evict(sid, noop_read).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(mgr.seq_mgr.allocator.num_free_blocks(), free_before + 1);
        assert!(mgr.is_evicted(sid));

        mgr.restore(sid, noop_write).unwrap();
        assert!(!mgr.is_evicted(sid));
        assert_eq!(mgr.cpu_bytes_used, 0);
    }

    #[test]
    fn restore_fails_without_evict() {
        let mut mgr = KvSwapManager::new(4, 64);
        let sid = mgr.seq_mgr.create_sequence();
        assert!(mgr.restore(sid, noop_write).is_err());
    }

    #[test]
    fn maybe_evict_triggers_when_below_threshold() {
        use crate::paged_attention::BLOCK_SIZE;
        let mut mgr = KvSwapManager::new(2, 64);
        let sid = mgr.seq_mgr.create_sequence();
        for _ in 0..BLOCK_SIZE {
            mgr.seq_mgr.append_token(sid).unwrap();
        }
        // 1 block used, 1 free. threshold=2 → evict
        let did_evict = mgr.maybe_evict(sid, 2, noop_read);
        assert!(did_evict);
        assert!(mgr.is_evicted(sid));
    }
}
