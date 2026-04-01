//! Security model for STRIX (Protocol §14).
//!
//! Provides defence-in-depth for VRAM and host-memory operations:
//!
//! - **`SecureAllocator`** — wraps any `GpuHal` to zero VRAM on free
//! - **`ShardedRwLock<T>`** — N-shard reader-writer lock for registry contention
//! - **`BoundsCheckedPtr`** — runtime-validated VRAM pointer + size pair
//! - **`OwnerToken`** — allocation tagging to prevent cross-session access
//! - **`SecurityAuditLog`** — ring buffer of `SecurityEvent`s

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::sync::RwLock;

// ── Owner Token ──────────────────────────────────────────────────────────

/// Opaque owner identity for allocation tagging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwnerToken(pub u64);

impl OwnerToken {
    /// Generate a new unique token from the given session id.
    pub fn new(session_id: u64) -> Self {
        Self(session_id)
    }
}

// ── Bounds-Checked Pointer ───────────────────────────────────────────────

/// A VRAM pointer paired with its allocation size for runtime bounds checking.
///
/// Every access through this type is validated against the known size,
/// preventing buffer over-reads and over-writes (Protocol §14.3).
#[derive(Debug, Clone, Copy)]
pub struct BoundsCheckedPtr {
    ptr: GpuPtr,
    size: usize,
    owner: OwnerToken,
}

/// Error from bounds-checked operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundsError {
    /// Access would extend beyond the allocation boundary.
    OutOfBounds {
        offset: usize,
        len: usize,
        size: usize,
    },
    /// Caller does not own this allocation.
    OwnerMismatch {
        expected: OwnerToken,
        actual: OwnerToken,
    },
}

impl std::fmt::Display for BoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds { offset, len, size } => {
                write!(f, "OOB: offset={offset} len={len} exceeds size={size}")
            }
            Self::OwnerMismatch { expected, actual } => {
                write!(f, "owner mismatch: expected {:?}, got {:?}", expected, actual)
            }
        }
    }
}

impl std::error::Error for BoundsError {}

impl BoundsCheckedPtr {
    /// Create a new bounds-checked pointer.
    pub fn new(ptr: GpuPtr, size: usize, owner: OwnerToken) -> Self {
        Self { ptr, size, owner }
    }

    /// Validate an access at `offset` of `len` bytes.
    ///
    /// Returns the absolute `GpuPtr` to the accessed region, or an error
    /// if the access would exceed the allocation boundary.
    pub fn validate_access(
        &self,
        offset: usize,
        len: usize,
        caller: OwnerToken,
    ) -> Result<GpuPtr, BoundsError> {
        if caller != self.owner {
            return Err(BoundsError::OwnerMismatch {
                expected: self.owner,
                actual: caller,
            });
        }
        if offset.saturating_add(len) > self.size {
            return Err(BoundsError::OutOfBounds {
                offset,
                len,
                size: self.size,
            });
        }
        Ok(GpuPtr(self.ptr.0 + offset as u64))
    }

    /// The base GPU pointer.
    pub fn ptr(&self) -> GpuPtr {
        self.ptr
    }

    /// Total size of the allocation.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The owner of this allocation.
    pub fn owner(&self) -> OwnerToken {
        self.owner
    }
}

// ── Sharded RwLock ───────────────────────────────────────────────────────

/// A reader-writer lock sharded across `N` independent locks.
///
/// Reduces contention when many threads read concurrently: each reader
/// hashes to one shard and only contends with writers on that shard.
/// Writers must acquire all shards (heavy, but rare in STRIX — eviction
/// is the only write path on the registry).
///
/// Protocol §14.2 — ShardedRwLock for tensor registry.
pub struct ShardedRwLock<T> {
    shards: Vec<RwLock<()>>,
    data: RwLock<T>,
    shard_count: usize,
}

impl<T> ShardedRwLock<T> {
    /// Create a new sharded lock with `n` shards.
    ///
    /// `n` is clamped to `[1, 64]`.
    pub fn new(data: T, n: usize) -> Self {
        let n = n.clamp(1, 64);
        let shards = (0..n).map(|_| RwLock::new(())).collect();
        Self {
            shards,
            data: RwLock::new(data),
            shard_count: n,
        }
    }

    /// Acquire a read lock on one shard (determined by `key`).
    ///
    /// Multiple readers with different keys run in parallel.
    pub fn read<F, R>(&self, key: u64, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let shard = (key as usize) % self.shard_count;
        let _shard_guard = self.shards[shard].read().unwrap();
        let data_guard = self.data.read().unwrap();
        f(&data_guard)
    }

    /// Acquire an exclusive write lock (locks ALL shards, then data).
    ///
    /// This serialises with all readers and other writers.
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        // Lock all shards to exclude all readers.
        let _guards: Vec<_> = self
            .shards
            .iter()
            .map(|s| s.write().unwrap())
            .collect();
        let mut data_guard = self.data.write().unwrap();
        f(&mut data_guard)
    }

    /// Number of shards.
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }
}

// ── Security Events ──────────────────────────────────────────────────────

/// Audit events tracked by the security subsystem.
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    /// VRAM was allocated.
    Allocated {
        ptr: GpuPtr,
        size: usize,
        owner: OwnerToken,
    },
    /// VRAM was zeroed before free.
    ZeroedAndFreed {
        ptr: GpuPtr,
        size: usize,
        owner: OwnerToken,
    },
    /// A bounds-check violation was caught.
    BoundsViolation {
        ptr: GpuPtr,
        offset: usize,
        len: usize,
        size: usize,
    },
    /// An owner-mismatch was caught.
    OwnerViolation {
        ptr: GpuPtr,
        expected: OwnerToken,
        actual: OwnerToken,
    },
}

/// Fixed-capacity ring buffer for security audit events.
pub struct SecurityAuditLog {
    events: Vec<SecurityEvent>,
    capacity: usize,
    write_pos: usize,
    total_logged: u64,
}

impl SecurityAuditLog {
    /// Create a new audit log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(16);
        Self {
            events: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            total_logged: 0,
        }
    }

    /// Record an event.
    pub fn log(&mut self, event: SecurityEvent) {
        if self.events.len() < self.capacity {
            self.events.push(event);
        } else {
            self.events[self.write_pos] = event;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_logged += 1;
    }

    /// Total number of events ever logged (including overwritten ones).
    pub fn total_logged(&self) -> u64 {
        self.total_logged
    }

    /// Current number of events in the buffer (up to capacity).
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate over events in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &SecurityEvent> {
        let (older, newer) = if self.events.len() < self.capacity {
            // Not yet wrapped — everything is in order.
            (self.events.as_slice(), &[] as &[SecurityEvent])
        } else {
            // Wrapped — older entries start at write_pos.
            let (b, a) = self.events.split_at(self.write_pos);
            (a, b)
        };
        older.iter().chain(newer.iter())
    }
}

// ── Secure Allocator ─────────────────────────────────────────────────────

/// Wrapper around any `GpuHal` that zeroes VRAM before freeing.
///
/// Protocol §14.1 — prevents data leakage between sessions or after OOM
/// evictions by overwriting VRAM with zeros before releasing it.
pub struct SecureAllocator<H: GpuHal> {
    inner: H,
    /// Allocation registry: ptr → (size, owner).
    allocations: HashMap<u64, (usize, OwnerToken)>,
    /// Audit log.
    audit_log: SecurityAuditLog,
}

impl<H: GpuHal> SecureAllocator<H> {
    /// Wrap a HAL backend with secure allocation.
    pub fn new(inner: H) -> Self {
        Self {
            inner,
            allocations: HashMap::new(),
            audit_log: SecurityAuditLog::new(1024),
        }
    }

    /// Allocate VRAM with owner tagging.
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
        owner: OwnerToken,
    ) -> Result<BoundsCheckedPtr, HalError> {
        let ptr = self.inner.allocate_vram(size, alignment)?;
        self.allocations.insert(ptr.0, (size, owner));
        self.audit_log.log(SecurityEvent::Allocated { ptr, size, owner });
        Ok(BoundsCheckedPtr::new(ptr, size, owner))
    }

    /// Zero VRAM content and then free the allocation.
    ///
    /// The zeroing is done via a host→device copy of a zero buffer,
    /// using stream 0 with a synchronous wait.
    pub fn secure_free(&mut self, ptr: GpuPtr) -> Result<(), HalError> {
        if let Some((size, owner)) = self.allocations.remove(&ptr.0) {
            // Zero the VRAM region before freeing.
            let zeros = vec![0u8; size.min(64 * 1024)]; // 64K chunks
            let mut offset = 0usize;
            while offset < size {
                let chunk = (size - offset).min(zeros.len());
                let dst = GpuPtr(ptr.0 + offset as u64);
                self.inner.copy_to_vram(dst, zeros.as_ptr(), chunk, 0)?;
                offset += chunk;
            }
            self.inner.sync_stream(0)?;
            self.inner.free_vram(ptr)?;
            self.audit_log.log(SecurityEvent::ZeroedAndFreed { ptr, size, owner });
            Ok(())
        } else {
            // Unknown pointer — treat as driver error.
            Err(HalError::DriverError {
                code: -100,
                message: format!("secure_free: unknown GpuPtr({:#x})", ptr.0),
            })
        }
    }

    /// Access the inner HAL (e.g. for info queries).
    pub fn inner(&self) -> &H {
        &self.inner
    }

    /// Access the audit log.
    pub fn audit_log(&self) -> &SecurityAuditLog {
        &self.audit_log
    }

    /// Number of active allocations.
    pub fn active_allocations(&self) -> usize {
        self.allocations.len()
    }
}

// Forward GpuHal trait through SecureAllocator so it can be used
// as a drop-in replacement.
impl<H: GpuHal> GpuHal for SecureAllocator<H> {
    fn info(&self) -> Result<GpuInfo, HalError> {
        self.inner.info()
    }

    fn allocate_vram(&self, size: usize, alignment: usize) -> Result<GpuPtr, HalError> {
        self.inner.allocate_vram(size, alignment)
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        // NOTE: prefer `secure_free()` for zeroing. This fallback
        // just forwards to the inner HAL without zeroing.
        self.inner.free_vram(ptr)
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError> {
        self.inner.copy_to_vram(dst, src, size, stream)
    }

    fn copy_from_vram(
        &self,
        dst: *mut u8,
        src: GpuPtr,
        size: usize,
        stream: u32,
    ) -> Result<(), HalError> {
        self.inner.copy_from_vram(dst, src, size, stream)
    }

    fn sync_stream(&self, stream: u32) -> Result<(), HalError> {
        self.inner.sync_stream(stream)
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        self.inner.vram_used()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // --- BoundsCheckedPtr ---

    #[test]
    fn bounds_check_valid_access() {
        let owner = OwnerToken::new(1);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        let result = bcp.validate_access(0, 256, owner);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), GpuPtr(0x1000));
    }

    #[test]
    fn bounds_check_partial_access() {
        let owner = OwnerToken::new(1);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        let result = bcp.validate_access(128, 64, owner);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), GpuPtr(0x1000 + 128));
    }

    #[test]
    fn bounds_check_oob() {
        let owner = OwnerToken::new(1);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        let result = bcp.validate_access(200, 100, owner);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoundsError::OutOfBounds { offset, len, size } => {
                assert_eq!(offset, 200);
                assert_eq!(len, 100);
                assert_eq!(size, 256);
            }
            _ => panic!("expected OutOfBounds"),
        }
    }

    #[test]
    fn bounds_check_overflow() {
        let owner = OwnerToken::new(1);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        let result = bcp.validate_access(usize::MAX, 1, owner);
        assert!(result.is_err());
    }

    #[test]
    fn bounds_check_owner_mismatch() {
        let owner = OwnerToken::new(1);
        let intruder = OwnerToken::new(2);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        let result = bcp.validate_access(0, 1, intruder);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoundsError::OwnerMismatch { expected, actual } => {
                assert_eq!(expected, owner);
                assert_eq!(actual, intruder);
            }
            _ => panic!("expected OwnerMismatch"),
        }
    }

    #[test]
    fn bounds_check_zero_len() {
        let owner = OwnerToken::new(1);
        let bcp = BoundsCheckedPtr::new(GpuPtr(0x1000), 256, owner);
        assert!(bcp.validate_access(256, 0, owner).is_ok());
    }

    // --- ShardedRwLock ---

    #[test]
    fn sharded_lock_read_write() {
        let lock = ShardedRwLock::new(42u64, 4);
        let val = lock.read(0, |v| *v);
        assert_eq!(val, 42);

        lock.write(|v| *v = 100);
        let val = lock.read(3, |v| *v);
        assert_eq!(val, 100);
    }

    #[test]
    fn sharded_lock_concurrent_reads() {
        let lock = Arc::new(ShardedRwLock::new(vec![1, 2, 3], 8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let lock = lock.clone();
                std::thread::spawn(move || lock.read(i as u64, |v| v.len()))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 3);
        }
    }

    #[test]
    fn sharded_lock_shard_count_clamped() {
        let lock = ShardedRwLock::new((), 0);
        assert_eq!(lock.shard_count(), 1);
        let lock = ShardedRwLock::new((), 200);
        assert_eq!(lock.shard_count(), 64);
    }

    // --- SecurityAuditLog ---

    #[test]
    fn audit_log_basic() {
        let mut log = SecurityAuditLog::new(16);
        assert!(log.is_empty());

        log.log(SecurityEvent::Allocated {
            ptr: GpuPtr(0x1000),
            size: 256,
            owner: OwnerToken::new(1),
        });
        assert_eq!(log.len(), 1);
        assert_eq!(log.total_logged(), 1);
    }

    #[test]
    fn audit_log_wraps() {
        let mut log = SecurityAuditLog::new(16);
        for i in 0..32u64 {
            log.log(SecurityEvent::Allocated {
                ptr: GpuPtr(i * 0x100),
                size: 64,
                owner: OwnerToken::new(1),
            });
        }
        assert_eq!(log.len(), 16); // capped at capacity
        assert_eq!(log.total_logged(), 32);

        // Oldest event should be #16 (index 16 in insertion order).
        let events: Vec<_> = log.iter().collect();
        assert_eq!(events.len(), 16);
        match &events[0] {
            SecurityEvent::Allocated { ptr, .. } => {
                assert_eq!(*ptr, GpuPtr(16 * 0x100));
            }
            _ => panic!("expected Allocated"),
        }
    }

    // --- SecureAllocator ---

    use crate::strix::cpu_hal::CpuHal;

    #[test]
    fn secure_allocator_alloc_and_free() {
        let hal = CpuHal::new(1024 * 1024);
        let mut sa = SecureAllocator::new(hal);

        let owner = OwnerToken::new(1);
        let bcp = sa.allocate(256, 64, owner).unwrap();
        assert_eq!(bcp.size(), 256);
        assert_eq!(sa.active_allocations(), 1);

        sa.secure_free(bcp.ptr()).unwrap();
        assert_eq!(sa.active_allocations(), 0);
        assert_eq!(sa.audit_log().total_logged(), 2); // alloc + free
    }

    #[test]
    fn secure_allocator_free_unknown() {
        let hal = CpuHal::new(1024 * 1024);
        let mut sa = SecureAllocator::new(hal);
        let result = sa.secure_free(GpuPtr(0xDEAD));
        assert!(result.is_err());
    }

    #[test]
    fn secure_allocator_zeroing_sets_data() {
        let hal = CpuHal::new(1024 * 1024);
        let mut sa = SecureAllocator::new(hal);
        let owner = OwnerToken::new(1);
        let bcp = sa.allocate(128, 64, owner).unwrap();

        // Write some data via the inner HAL.
        let data = vec![0xFFu8; 128];
        sa.inner.copy_to_vram(bcp.ptr(), data.as_ptr(), 128, 0).unwrap();

        // Secure free should zero then free.
        sa.secure_free(bcp.ptr()).unwrap();

        // The audit log should show both alloc and zeroed-free.
        let events: Vec<_> = sa.audit_log().iter().collect();
        assert!(matches!(events[0], SecurityEvent::Allocated { .. }));
        assert!(matches!(events[1], SecurityEvent::ZeroedAndFreed { .. }));
    }
}
