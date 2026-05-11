//! PagedAttention v2 — v0.5.0
//!
//! Non-contiguous KV block management with copy-on-write for shared prefixes.
//!
//! # Research Basis
//!
//! - **vLLM PagedAttention** (Kwon et al., SOSP 2023): eliminates KV cache
//!   memory fragmentation by mapping logical KV positions to fixed-size
//!   physical blocks. A block table translates sequence positions to physical
//!   block addresses, enabling near-zero memory waste (<0.1% fragmentation).
//!
//! - **PD-Disagg** (Zhong et al., 2024): disaggregated prefill + decode
//!   workers sharing the same block allocator across NVLink/PCIe. Each
//!   decode worker holds a pointer to remote blocks via the block table.
//!
//! # Architecture
//!
//! ```text
//! BlockAllocator
//!   ├─ free_blocks: VecDeque<PhysicalBlockId>
//!   ├─ blocks: [PhysicalBlock; N]        ← fixed pool pre-allocated
//!   └─ ref_counts: [u32; N]             ← for copy-on-write
//!
//! BlockTable (per sequence)
//!   └─ logical_block_id → PhysicalBlockId   (grow on demand)
//!
//! Copy-on-Write (beam search / parallel sampling):
//!   When ref_count[block] > 1 and a writer needs to modify:
//!   1. Allocate a new physical block
//!   2. Copy content from src → dst
//!   3. Decrement ref_count[src], set block_table entry = dst
//! ```

use std::collections::{HashMap, VecDeque};

// ── Constants ──────────────────────────────────────────────────────────────

/// Tokens per physical block (page size). 16 = good balance of granularity and
/// management overhead. vLLM uses 16 or 32.
pub const BLOCK_SIZE: usize = 16;

/// Maximum number of physical blocks in the pool. Each block holds
/// BLOCK_SIZE × n_heads × head_dim × 2 (K+V) × 2 (bf16) bytes.
/// For 128 heads × 128 dim × bf16: 16 × 8 × 128 × 2 × 2 = 65 536 bytes ≈ 64 KB/block.
pub const MAX_BLOCKS: usize = 4096; // 4096 × 64 KB = 256 MB

// ── Types ──────────────────────────────────────────────────────────────────

/// Physical block identifier (index into the block pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub u32);

/// Logical block index within a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalBlockId(pub u32);

/// Sequence identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub u32);

/// A physical KV block — metadata only (actual GPU memory managed externally).
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    pub id: PhysicalBlockId,
    /// Number of valid token slots currently filled in this block.
    pub num_tokens: usize,
    /// True if this block's content is complete (cannot add more tokens).
    pub is_full: bool,
}

impl PhysicalBlock {
    pub fn new(id: PhysicalBlockId) -> Self {
        Self { id, num_tokens: 0, is_full: false }
    }

    /// Remaining free slots in this block.
    pub fn free_slots(&self) -> usize {
        BLOCK_SIZE - self.num_tokens
    }
}

// ── Block Allocator ────────────────────────────────────────────────────────

/// Pre-allocated fixed-size block pool with reference counting.
///
/// Thread-safety: callers must hold an external `Mutex` — the allocator
/// itself is not `Sync` by design (matches vLLM's Python GIL model).
#[derive(Debug)]
pub struct BlockAllocator {
    blocks: Vec<PhysicalBlock>,
    ref_counts: Vec<u32>,
    free_blocks: VecDeque<PhysicalBlockId>,
    /// Total blocks allocated (including currently in use).
    pub total_blocks: usize,
}

impl BlockAllocator {
    /// Create a new allocator with `capacity` blocks.
    pub fn new(capacity: usize) -> Self {
        let blocks: Vec<PhysicalBlock> = (0..capacity)
            .map(|i| PhysicalBlock::new(PhysicalBlockId(i as u32)))
            .collect();
        let free_blocks: VecDeque<PhysicalBlockId> = (0..capacity)
            .map(|i| PhysicalBlockId(i as u32))
            .collect();
        Self {
            total_blocks: capacity,
            ref_counts: vec![0u32; capacity],
            blocks,
            free_blocks,
        }
    }

    /// Allocate one free block. Returns `None` if the pool is exhausted (OOM).
    pub fn allocate(&mut self) -> Option<PhysicalBlockId> {
        let block_id = self.free_blocks.pop_front()?;
        self.ref_counts[block_id.0 as usize] = 1;
        self.blocks[block_id.0 as usize].num_tokens = 0;
        self.blocks[block_id.0 as usize].is_full = false;
        Some(block_id)
    }

    /// Increment the reference count (share a block across sequences).
    pub fn share(&mut self, block_id: PhysicalBlockId) {
        self.ref_counts[block_id.0 as usize] += 1;
    }

    /// Decrement the reference count; free if it reaches zero.
    pub fn release(&mut self, block_id: PhysicalBlockId) {
        let rc = &mut self.ref_counts[block_id.0 as usize];
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.free_blocks.push_back(block_id);
        }
    }

    /// Reference count for a block.
    pub fn ref_count(&self, block_id: PhysicalBlockId) -> u32 {
        self.ref_counts[block_id.0 as usize]
    }

    /// Number of free blocks remaining.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of blocks currently in use.
    pub fn num_used_blocks(&self) -> usize {
        self.total_blocks - self.free_blocks.len()
    }

    /// Perform copy-on-write: if ref_count > 1, allocate a new block, copy
    /// the old block's metadata, decrement the old block's ref_count.
    /// Returns the new (or unchanged) `PhysicalBlockId`.
    pub fn cow_write(&mut self, block_id: PhysicalBlockId) -> Option<PhysicalBlockId> {
        if self.ref_count(block_id) <= 1 {
            return Some(block_id); // sole owner — no copy needed
        }
        let new_block_id = self.allocate()?;
        // Copy metadata (in a real impl this would also copy GPU buffer)
        let src = self.blocks[block_id.0 as usize].clone();
        self.blocks[new_block_id.0 as usize].num_tokens = src.num_tokens;
        self.blocks[new_block_id.0 as usize].is_full = src.is_full;
        // Release old block (decrement rc)
        self.release(block_id);
        Some(new_block_id)
    }

    /// Append one token to a block. Returns `true` if the block is now full.
    pub fn append_token(&mut self, block_id: PhysicalBlockId) -> bool {
        let block = &mut self.blocks[block_id.0 as usize];
        assert!(!block.is_full, "cannot append to a full block");
        block.num_tokens += 1;
        if block.num_tokens == BLOCK_SIZE {
            block.is_full = true;
        }
        block.is_full
    }
}

// ── Block Table ────────────────────────────────────────────────────────────

/// Per-sequence block table: logical_block_id → physical_block_id.
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub seq_id: SeqId,
    /// Ordered list of physical blocks assigned to this sequence.
    pub physical_blocks: Vec<PhysicalBlockId>,
    /// Total tokens appended to this sequence.
    pub num_tokens: usize,
}

impl BlockTable {
    pub fn new(seq_id: SeqId) -> Self {
        Self { seq_id, physical_blocks: Vec::new(), num_tokens: 0 }
    }

    /// Logical block index for the i-th token.
    pub fn logical_block_for_token(token_pos: usize) -> LogicalBlockId {
        LogicalBlockId((token_pos / BLOCK_SIZE) as u32)
    }

    /// Physical block for a given token position (panics if not allocated).
    pub fn physical_block_for_token(&self, token_pos: usize) -> PhysicalBlockId {
        let lbi = token_pos / BLOCK_SIZE;
        self.physical_blocks[lbi]
    }

    /// Total number of logical blocks allocated (including partially filled).
    pub fn num_logical_blocks(&self) -> usize {
        self.physical_blocks.len()
    }

    /// Slot index within the current last block.
    pub fn last_block_offset(&self) -> usize {
        self.num_tokens % BLOCK_SIZE
    }
}

// ── Sequence Manager ───────────────────────────────────────────────────────

/// Manages all active sequences and their block tables.
///
/// Equivalent to vLLM's `BlockSpaceManager`.
#[derive(Debug)]
pub struct SequenceManager {
    pub allocator: BlockAllocator,
    /// Map seq_id → BlockTable
    tables: HashMap<SeqId, BlockTable>,
    next_seq_id: u32,
}

impl SequenceManager {
    pub fn new(block_capacity: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(block_capacity),
            tables: HashMap::new(),
            next_seq_id: 0,
        }
    }

    /// Register a new sequence. Returns its `SeqId`.
    pub fn create_sequence(&mut self) -> SeqId {
        let id = SeqId(self.next_seq_id);
        self.next_seq_id += 1;
        self.tables.insert(id, BlockTable::new(id));
        id
    }

    /// Append a token to the sequence, allocating a new block if needed.
    /// Returns `Err` if out of memory.
    pub fn append_token(&mut self, seq_id: SeqId) -> Result<PhysicalBlockId, &'static str> {
        let table = self.tables.get_mut(&seq_id).ok_or("sequence not found")?;
        let needs_new_block = table.physical_blocks.is_empty()
            || table.last_block_offset() == 0 && table.num_tokens > 0;

        let active_block_id = if needs_new_block {
            let block_id = self.allocator.allocate().ok_or("out of memory — no free blocks")?;
            // Need to re-borrow after allocate (borrow checker)
            self.tables.get_mut(&seq_id).unwrap().physical_blocks.push(block_id);
            block_id
        } else {
            *self.tables[&seq_id].physical_blocks.last().unwrap()
        };

        self.allocator.append_token(active_block_id);
        self.tables.get_mut(&seq_id).unwrap().num_tokens += 1;
        Ok(active_block_id)
    }

    /// Fork a sequence for beam search / parallel sampling.
    /// Increments ref_counts of all blocks in the parent sequence.
    pub fn fork_sequence(&mut self, parent_id: SeqId) -> Result<SeqId, &'static str> {
        let parent_blocks = self.tables
            .get(&parent_id)
            .ok_or("parent sequence not found")?
            .physical_blocks
            .clone();
        let parent_tokens = self.tables[&parent_id].num_tokens;

        // Share all blocks with the forked sequence
        for &bid in &parent_blocks {
            self.allocator.share(bid);
        }

        let child_id = SeqId(self.next_seq_id);
        self.next_seq_id += 1;
        let mut child_table = BlockTable::new(child_id);
        child_table.physical_blocks = parent_blocks;
        child_table.num_tokens = parent_tokens;
        self.tables.insert(child_id, child_table);
        Ok(child_id)
    }

    /// Free all blocks held by a sequence.
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(table) = self.tables.remove(&seq_id) {
            for bid in table.physical_blocks {
                self.allocator.release(bid);
            }
        }
    }

    /// Get the block table for a sequence.
    pub fn table(&self, seq_id: SeqId) -> Option<&BlockTable> {
        self.tables.get(&seq_id)
    }

    /// Perform copy-on-write for the last block of a sequence (before write).
    pub fn cow_last_block(&mut self, seq_id: SeqId) -> Option<PhysicalBlockId> {
        let last_bid = *self.tables.get(&seq_id)?.physical_blocks.last()?;
        let new_bid = self.allocator.cow_write(last_bid)?;
        if new_bid != last_bid {
            let table = self.tables.get_mut(&seq_id)?;
            *table.physical_blocks.last_mut()? = new_bid;
        }
        Some(new_bid)
    }

    /// Number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.tables.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_allocates_and_releases() {
        let mut alloc = BlockAllocator::new(4);
        let b1 = alloc.allocate().unwrap();
        let b2 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free_blocks(), 2);
        alloc.release(b1);
        assert_eq!(alloc.num_free_blocks(), 3);
        alloc.release(b2);
        assert_eq!(alloc.num_free_blocks(), 4);
    }

    #[test]
    fn allocator_exhaustion_returns_none() {
        let mut alloc = BlockAllocator::new(2);
        alloc.allocate().unwrap();
        alloc.allocate().unwrap();
        assert!(alloc.allocate().is_none(), "pool exhausted → None");
    }

    #[test]
    fn ref_count_tracks_sharing() {
        let mut alloc = BlockAllocator::new(4);
        let b = alloc.allocate().unwrap();
        assert_eq!(alloc.ref_count(b), 1);
        alloc.share(b);
        assert_eq!(alloc.ref_count(b), 2);
        alloc.release(b);
        assert_eq!(alloc.ref_count(b), 1);
        alloc.release(b);
        assert_eq!(alloc.ref_count(b), 0);
    }

    #[test]
    fn cow_write_copies_on_shared_block() {
        let mut alloc = BlockAllocator::new(4);
        let b = alloc.allocate().unwrap();
        alloc.share(b); // ref_count = 2
        let new_b = alloc.cow_write(b).unwrap();
        assert_ne!(new_b, b, "CoW should give a different block");
        assert_eq!(alloc.ref_count(b), 1, "old block rc decremented");
        assert_eq!(alloc.ref_count(new_b), 1, "new block rc = 1");
    }

    #[test]
    fn cow_write_no_copy_on_sole_owner() {
        let mut alloc = BlockAllocator::new(4);
        let b = alloc.allocate().unwrap();
        let same = alloc.cow_write(b).unwrap();
        assert_eq!(same, b, "sole owner — no copy");
    }

    #[test]
    fn sequence_manager_create_and_append() {
        let mut mgr = SequenceManager::new(16);
        let sid = mgr.create_sequence();
        for _ in 0..BLOCK_SIZE {
            mgr.append_token(sid).unwrap();
        }
        // After BLOCK_SIZE tokens, 1 block should be filled
        let table = mgr.table(sid).unwrap();
        assert_eq!(table.num_logical_blocks(), 1);
        assert_eq!(table.num_tokens, BLOCK_SIZE);
        // Token BLOCK_SIZE+1 needs a new block
        mgr.append_token(sid).unwrap();
        assert_eq!(mgr.table(sid).unwrap().num_logical_blocks(), 2);
    }

    #[test]
    fn fork_shares_parent_blocks() {
        let mut mgr = SequenceManager::new(16);
        let parent = mgr.create_sequence();
        for _ in 0..BLOCK_SIZE {
            mgr.append_token(parent).unwrap();
        }
        let child = mgr.fork_sequence(parent).unwrap();
        // Child sees the same physical block
        assert_eq!(
            mgr.table(parent).unwrap().physical_blocks[0],
            mgr.table(child).unwrap().physical_blocks[0]
        );
        // ref_count of shared block is 2
        let bid = mgr.table(parent).unwrap().physical_blocks[0];
        assert_eq!(mgr.allocator.ref_count(bid), 2);
    }

    #[test]
    fn free_sequence_releases_blocks() {
        let mut mgr = SequenceManager::new(4);
        let sid = mgr.create_sequence();
        mgr.append_token(sid).unwrap();
        let free_before = mgr.allocator.num_free_blocks();
        mgr.free_sequence(sid);
        assert_eq!(mgr.allocator.num_free_blocks(), free_before + 1);
    }

    #[test]
    fn append_token_oom_error() {
        let mut mgr = SequenceManager::new(1);
        let sid = mgr.create_sequence();
        // Fill the only block
        for _ in 0..BLOCK_SIZE {
            mgr.append_token(sid).unwrap();
        }
        // Next token needs a new block but pool is empty
        let result = mgr.append_token(sid);
        assert!(result.is_err(), "should OOM");
    }

    #[test]
    fn block_manager_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BlockAllocator>();
        assert_send::<SequenceManager>();
    }

    #[test]
    fn logical_block_for_token_correct() {
        assert_eq!(BlockTable::logical_block_for_token(0), LogicalBlockId(0));
        assert_eq!(BlockTable::logical_block_for_token(BLOCK_SIZE - 1), LogicalBlockId(0));
        assert_eq!(BlockTable::logical_block_for_token(BLOCK_SIZE), LogicalBlockId(1));
    }
}
