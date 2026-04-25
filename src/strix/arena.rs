//! VRAM Arena — free-list allocator for GPU memory budget tracking.
//!
//! `VramArena` manages a virtual VRAM budget using a sorted free-list
//! strategy. It does **not** make real GPU allocation calls — those go
//! through `dyn GpuHal`. Instead, it tracks which regions of the budget
//! are in use and which are free, so the scheduler can make informed
//! eviction decisions.
//!
//! Key properties:
//! - Alignment-aware allocation (respects GPU page boundaries)
//! - Free-region coalescing on deallocation (prevents fragmentation)
//! - O(n) first-fit search (n = number of free regions, typically small)

use super::types::GpuPtr;

// ── Allocation ───────────────────────────────────────────────────────────

/// A successful allocation from the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    /// Opaque GPU pointer (offset within the arena).
    pub ptr: GpuPtr,
    /// Allocated size in bytes (may be larger than requested due to alignment).
    pub size: usize,
    /// Alignment that was applied.
    pub alignment: usize,
}

// ── FreeRegion ───────────────────────────────────────────────────────────

/// A contiguous free region within the arena.
#[derive(Debug, Clone, Copy)]
struct FreeRegion {
    offset: u64,
    size: usize,
}

// ── VramArena ────────────────────────────────────────────────────────────

/// Virtual VRAM budget allocator with free-list tracking.
///
/// Manages allocations against a fixed total budget, keeping a
/// safety margin reserved for framework overhead (CUDA contexts,
/// cuBLAS workspaces, etc.).
pub struct VramArena {
    /// Total VRAM available (bytes).
    total: usize,
    /// Reserved safety margin (bytes) — never allocated.
    safety_margin: usize,
    /// Sorted list of free regions (by offset).
    free_list: Vec<FreeRegion>,
    /// Total bytes currently allocated.
    used: usize,
}

impl VramArena {
    /// Create a new arena with the given total VRAM and safety margin.
    ///
    /// The usable budget is `total - safety_margin`. The entire usable
    /// region starts as a single free block.
    pub fn new(total_vram: usize, safety_margin: usize) -> Self {
        let usable = total_vram.saturating_sub(safety_margin);
        Self {
            total: total_vram,
            safety_margin,
            free_list: vec![FreeRegion {
                offset: 0,
                size: usable,
            }],
            used: 0,
        }
    }

    /// Total VRAM budget (including safety margin).
    pub fn total(&self) -> usize {
        self.total
    }

    /// Usable budget (total minus safety margin).
    pub fn usable(&self) -> usize {
        self.total.saturating_sub(self.safety_margin)
    }

    /// Bytes currently allocated.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Bytes currently available for allocation.
    pub fn available(&self) -> usize {
        self.usable().saturating_sub(self.used)
    }

    /// Utilization ratio (0.0 = empty, 1.0 = full).
    pub fn utilization(&self) -> f64 {
        let usable = self.usable();
        if usable == 0 {
            return 0.0;
        }
        self.used as f64 / usable as f64
    }

    /// Number of free regions (fragmentation indicator).
    pub fn fragment_count(&self) -> usize {
        self.free_list.len()
    }

    /// Allocate `size` bytes with the given alignment.
    ///
    /// Uses first-fit strategy on the sorted free list.
    /// Returns `None` if no suitable region is available.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<Allocation> {
        if size == 0 {
            return None;
        }
        let alignment = alignment.max(1);

        // First-fit search
        for i in 0..self.free_list.len() {
            let region = self.free_list[i];

            // Align the offset upwards
            let aligned_offset = align_up(region.offset, alignment as u64);
            let padding = (aligned_offset - region.offset) as usize;
            let total_needed = padding + size;

            if region.size >= total_needed {
                // This region fits. Split it.
                let alloc = Allocation {
                    ptr: GpuPtr(aligned_offset),
                    size,
                    alignment,
                };

                if region.size == total_needed {
                    // Exact fit — remove the region
                    self.free_list.remove(i);
                } else {
                    // Shrink the region (take from the front)
                    self.free_list[i] = FreeRegion {
                        offset: region.offset + total_needed as u64,
                        size: region.size - total_needed,
                    };
                }

                // If there was alignment padding, add the padding gap back as a free region
                if padding > 0 {
                    let gap = FreeRegion {
                        offset: region.offset,
                        size: padding,
                    };
                    self.insert_free_sorted(gap);
                }

                self.used += size;
                return Some(alloc);
            }
        }

        None // OOM
    }

    /// Free a previously allocated region.
    ///
    /// Adds the region back to the free list and coalesces with adjacent
    /// free regions to prevent fragmentation.
    pub fn free(&mut self, alloc: Allocation) {
        self.used = self.used.saturating_sub(alloc.size);

        let freed = FreeRegion {
            offset: alloc.ptr.0,
            size: alloc.size,
        };
        self.insert_free_sorted(freed);
        self.coalesce();
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Insert a free region into the sorted list (by offset).
    fn insert_free_sorted(&mut self, region: FreeRegion) {
        let pos = self
            .free_list
            .binary_search_by_key(&region.offset, |r| r.offset)
            .unwrap_or_else(|e| e);
        self.free_list.insert(pos, region);
    }

    /// Merge adjacent free regions into larger contiguous blocks.
    fn coalesce(&mut self) {
        if self.free_list.len() < 2 {
            return;
        }

        let mut merged = Vec::with_capacity(self.free_list.len());
        let mut current = self.free_list[0];

        for i in 1..self.free_list.len() {
            let next = self.free_list[i];
            let current_end = current.offset + current.size as u64;

            if current_end >= next.offset {
                // Merge: extend current to cover next
                let new_end = (next.offset + next.size as u64).max(current_end);
                current.size = (new_end - current.offset) as usize;
            } else {
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);

        self.free_list = merged;
    }
}

/// Align `value` upwards to the next multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) / alignment * alignment
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_allocate_and_free() {
        let mut arena = VramArena::new(1024, 0);
        assert_eq!(arena.available(), 1024);

        let alloc = arena.allocate(256, 1).unwrap();
        assert_eq!(alloc.size, 256);
        assert_eq!(arena.used(), 256);
        assert_eq!(arena.available(), 768);

        arena.free(alloc);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.available(), 1024);
    }

    #[test]
    fn alignment_enforcement() {
        let mut arena = VramArena::new(4096, 0);

        // First allocation at offset 0 (already aligned)
        let a1 = arena.allocate(100, 256).unwrap();
        assert_eq!(a1.ptr.0 % 256, 0);

        // Second allocation should also be aligned
        let a2 = arena.allocate(100, 256).unwrap();
        assert_eq!(a2.ptr.0 % 256, 0);
    }

    #[test]
    fn oom_on_exhaustion() {
        let mut arena = VramArena::new(256, 0);
        let _a1 = arena.allocate(256, 1).unwrap();
        // Arena is full — next allocation should fail
        assert!(arena.allocate(1, 1).is_none());
    }

    #[test]
    fn safety_margin_reserved() {
        let arena = VramArena::new(1024, 200);
        assert_eq!(arena.total(), 1024);
        assert_eq!(arena.usable(), 824);
        assert_eq!(arena.available(), 824);
    }

    #[test]
    fn coalescing_reduces_fragments() {
        let mut arena = VramArena::new(1024, 0);

        // Allocate 4 × 256-byte blocks
        let a0 = arena.allocate(256, 1).unwrap();
        let a1 = arena.allocate(256, 1).unwrap();
        let a2 = arena.allocate(256, 1).unwrap();
        let a3 = arena.allocate(256, 1).unwrap();
        assert_eq!(arena.used(), 1024);

        // Free middle two — creates 2 free regions initially
        arena.free(a1);
        arena.free(a2);
        // After coalescing, should be 1 free region (contiguous 512 bytes)
        assert_eq!(arena.fragment_count(), 1);
        assert_eq!(arena.available(), 512);

        // Free first — coalesces with the middle free block
        arena.free(a0);
        assert_eq!(arena.fragment_count(), 1);
        assert_eq!(arena.available(), 768);

        // Free last — everything coalesces into one big region
        arena.free(a3);
        assert_eq!(arena.fragment_count(), 1);
        assert_eq!(arena.available(), 1024);
    }

    #[test]
    fn utilization_math() {
        let mut arena = VramArena::new(1000, 0);
        assert!((arena.utilization() - 0.0).abs() < 1e-6);

        arena.allocate(500, 1).unwrap();
        assert!((arena.utilization() - 0.5).abs() < 1e-6);

        arena.allocate(500, 1).unwrap();
        assert!((arena.utilization() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_size_allocation_returns_none() {
        let mut arena = VramArena::new(1024, 0);
        assert!(arena.allocate(0, 1).is_none());
    }
}
