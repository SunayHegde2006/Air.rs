use air_core::error::{AirError, Result};
use std::collections::HashMap;

pub const PAGE_BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalBlockId(pub usize);

pub struct PagedKvCache {
    block_size: usize,
    num_physical_blocks: usize,
    free_blocks: Vec<PhysicalBlockId>,
    block_tables: HashMap<usize, Vec<PhysicalBlockId>>, // Sequence ID -> list of blocks
}

impl PagedKvCache {
    pub fn new(num_physical_blocks: usize) -> Self {
        let free_blocks = (0..num_physical_blocks)
            .map(PhysicalBlockId)
            .collect();

        Self {
            block_size: PAGE_BLOCK_SIZE,
            num_physical_blocks,
            free_blocks,
            block_tables: HashMap::new(),
        }
    }

    pub fn allocate_block(&mut self, seq_id: usize) -> Result<PhysicalBlockId> {
        let block = self
            .free_blocks
            .pop()
            .ok_or_else(|| AirError::Ice("Out of physical KV cache blocks".into()))?;

        self.block_tables
            .entry(seq_id)
            .or_insert_with(Vec::new)
            .push(block);

        Ok(block)
    }

    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(blocks) = self.block_tables.remove(&seq_id) {
            self.free_blocks.extend(blocks);
        }
    }

    pub fn get_block_table(&self, seq_id: usize) -> Option<&Vec<PhysicalBlockId>> {
        self.block_tables.get(&seq_id)
    }
}
