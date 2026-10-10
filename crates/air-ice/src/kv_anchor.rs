use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct KVAnchor {
    current_hash: u64,
}

impl KVAnchor {
    pub fn new() -> Self {
        Self { current_hash: 0 }
    }

    pub fn update(&mut self, token_id: u32, k_bytes: &[u8]) {
        let mut hasher = DefaultHasher::new();
        self.current_hash.hash(&mut hasher);
        token_id.hash(&mut hasher);
        k_bytes.hash(&mut hasher);
        self.current_hash = hasher.finish();
    }

    pub fn current_hash(&self) -> u64 {
        self.current_hash
    }
}

impl Default for KVAnchor {
    fn default() -> Self {
        Self::new()
    }
}
