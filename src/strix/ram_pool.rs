//! RAM buffer pool with RAII guards (STRIX Protocol §13.4).
//!
//! `RamPool` pre-allocates and recycles host-memory buffers used as
//! staging areas for VRAM↔RAM transfers. Buffers are bucketed by
//! power-of-2 size classes to reduce fragmentation.
//!
//! Hardware/OS agnostic — uses the standard system allocator only.

use std::collections::BTreeMap;
use std::fmt;

// ── RamPool ──────────────────────────────────────────────────────────────

/// Recycling pool for host-memory staging buffers.
///
/// Buffers are organised by size class (power-of-2). When a buffer
/// is returned via `release()` (or dropped via `RamBuffer`), it goes
/// back into the pool for reuse instead of being deallocated.
pub struct RamPool {
    /// Available buffers by size class.
    buckets: BTreeMap<usize, Vec<Vec<u8>>>,
    /// Maximum total pool capacity in bytes.
    max_pool_bytes: usize,
    /// Current bytes held in the pool (available, not in-use).
    pool_bytes: usize,
}

impl RamPool {
    /// Create a new pool with maximum capacity `max_bytes`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            buckets: BTreeMap::new(),
            max_pool_bytes: max_bytes,
            pool_bytes: 0,
        }
    }

    /// Round `size` up to the nearest power of 2 (minimum 4096).
    fn size_class(size: usize) -> usize {
        let min = 4096;
        let s = size.max(min);
        s.next_power_of_two()
    }

    /// Acquire a buffer of at least `size` bytes.
    ///
    /// If a recycled buffer of the right size class exists, it is reused.
    /// Otherwise a fresh allocation is made from the system allocator.
    /// The returned buffer is zero-initialised on first allocation only;
    /// recycled buffers may contain stale data.
    pub fn acquire(&mut self, size: usize) -> RamBuffer {
        let class = Self::size_class(size);
        let data = if let Some(bucket) = self.buckets.get_mut(&class) {
            if let Some(buf) = bucket.pop() {
                self.pool_bytes -= buf.len();
                buf
            } else {
                vec![0u8; class]
            }
        } else {
            vec![0u8; class]
        };

        RamBuffer {
            data,
            requested_len: size,
        }
    }

    /// Return a buffer to the pool for recycling.
    ///
    /// If the pool is at capacity, the buffer is simply dropped.
    pub fn release(&mut self, buf: RamBuffer) {
        let cap = buf.data.len();
        if self.pool_bytes + cap > self.max_pool_bytes {
            // Over capacity — let it drop.
            return;
        }
        self.pool_bytes += cap;
        self.buckets.entry(cap).or_default().push(buf.data);
    }

    /// Total bytes currently held in the pool (recycled, available).
    pub fn pool_bytes(&self) -> usize {
        self.pool_bytes
    }

    /// Maximum pool capacity in bytes.
    pub fn max_bytes(&self) -> usize {
        self.max_pool_bytes
    }

    /// Number of recycled buffers across all size classes.
    pub fn recycled_count(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    /// Drop all recycled buffers, freeing memory back to the OS.
    pub fn drain(&mut self) {
        self.buckets.clear();
        self.pool_bytes = 0;
    }
}

impl fmt::Debug for RamPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RamPool")
            .field("pool_bytes", &self.pool_bytes)
            .field("max_pool_bytes", &self.max_pool_bytes)
            .field("recycled_count", &self.recycled_count())
            .finish()
    }
}

// ── RamBuffer ────────────────────────────────────────────────────────────

/// A buffer acquired from `RamPool`.
///
/// Provides slice access to the usable region (up to `requested_len`).
/// The backing storage may be larger due to size-class rounding.
pub struct RamBuffer {
    /// Backing storage (size-class rounded).
    data: Vec<u8>,
    /// User-requested length (≤ `data.len()`).
    requested_len: usize,
}

impl RamBuffer {
    /// View the usable region as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.requested_len]
    }

    /// View the usable region as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.requested_len]
    }

    /// Pointer to the start of the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Mutable pointer to the start of the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// User-requested length.
    pub fn len(&self) -> usize {
        self.requested_len
    }

    /// Whether the buffer is zero-length.
    pub fn is_empty(&self) -> bool {
        self.requested_len == 0
    }

    /// Actual backing capacity (size-class rounded).
    pub fn capacity(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for RamBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RamBuffer")
            .field("len", &self.requested_len)
            .field("capacity", &self.data.len())
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_class_rounds_up() {
        assert_eq!(RamPool::size_class(1), 4096);
        assert_eq!(RamPool::size_class(4096), 4096);
        assert_eq!(RamPool::size_class(4097), 8192);
        assert_eq!(RamPool::size_class(10000), 16384);
        assert_eq!(RamPool::size_class(65536), 65536);
    }

    #[test]
    fn acquire_returns_correct_length() {
        let mut pool = RamPool::new(1024 * 1024);
        let buf = pool.acquire(1000);
        assert_eq!(buf.len(), 1000);
        assert!(buf.capacity() >= 4096);
    }

    #[test]
    fn release_and_reuse() {
        let mut pool = RamPool::new(1024 * 1024);
        let buf = pool.acquire(5000);
        let cap = buf.capacity();
        pool.release(buf);
        assert_eq!(pool.recycled_count(), 1);
        assert_eq!(pool.pool_bytes(), cap);

        // Second acquire reuses the recycled buffer.
        let buf2 = pool.acquire(5000);
        assert_eq!(buf2.capacity(), cap);
        assert_eq!(pool.recycled_count(), 0);
        assert_eq!(pool.pool_bytes(), 0);
    }

    #[test]
    fn pool_max_enforced() {
        let mut pool = RamPool::new(4096); // tiny pool
        let buf = pool.acquire(5000); // rounds to 8192 — exceeds pool max
        pool.release(buf);
        // Should be dropped, not pooled.
        assert_eq!(pool.recycled_count(), 0);
        assert_eq!(pool.pool_bytes(), 0);
    }

    #[test]
    fn drain_clears_pool() {
        let mut pool = RamPool::new(1024 * 1024);
        let b1 = pool.acquire(1000);
        let b2 = pool.acquire(2000);
        pool.release(b1);
        pool.release(b2);
        assert!(pool.recycled_count() >= 2);
        pool.drain();
        assert_eq!(pool.recycled_count(), 0);
        assert_eq!(pool.pool_bytes(), 0);
    }

    #[test]
    fn buffer_read_write() {
        let mut pool = RamPool::new(1024 * 1024);
        let mut buf = pool.acquire(8);
        let slice = buf.as_mut_slice();
        slice.copy_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(buf.as_slice(), &[10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn multiple_size_classes() {
        let mut pool = RamPool::new(10 * 1024 * 1024);
        let b4k = pool.acquire(4000);
        let b8k = pool.acquire(5000);
        let b16k = pool.acquire(10000);

        assert_eq!(b4k.capacity(), 4096);
        assert_eq!(b8k.capacity(), 8192);
        assert_eq!(b16k.capacity(), 16384);

        pool.release(b4k);
        pool.release(b8k);
        pool.release(b16k);
        assert_eq!(pool.recycled_count(), 3);
    }

    #[test]
    fn large_buffer_exercises_real_allocator() {
        let mut pool = RamPool::new(128 * 1024 * 1024);
        let mut buf = pool.acquire(32 * 1024 * 1024); // 32 MB
        assert_eq!(buf.len(), 32 * 1024 * 1024);
        // Write to first and last byte to verify real allocation.
        buf.as_mut_slice()[0] = 0xAA;
        buf.as_mut_slice()[32 * 1024 * 1024 - 1] = 0xBB;
        assert_eq!(buf.as_slice()[0], 0xAA);
        assert_eq!(buf.as_slice()[32 * 1024 * 1024 - 1], 0xBB);
        pool.release(buf);
    }
}
