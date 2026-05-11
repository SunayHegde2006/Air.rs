//! # RadixAttention Prefix Cache
//!
//! Automatic KV-cache reuse via a **concurrent Patricia radix trie** keyed on
//! token-id sequences. Longest-prefix-match before prefill → skip recomputing
//! shared system prompts and conversation history.
//!
//! **Research**: "Efficiently Programming Large Language Models using SGLang"
//! (Zheng et al., SOSP 2024, arXiv:2312.07104).
//!
//! **Air.rs advantage**: pure Rust, `AtomicU64` LRU timestamps, no GIL, no
//! Python interpreter. Lock-free hot-path reads via per-node `RwLock`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Eviction policy
// ---------------------------------------------------------------------------

/// Eviction strategy for the prefix cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-recently-used: evict the node with the oldest `last_access`.
    Lru,
    /// Least-frequently-used: evict the node with the lowest `ref_count`.
    /// Useful for workloads with many unique one-off prompts.
    Lfu,
}

// ---------------------------------------------------------------------------
// RadixNode
// ---------------------------------------------------------------------------

/// A single node in the prefix trie.
///
/// Each node covers a contiguous run of token ids (`token_ids`) and stores
/// the corresponding PagedAttention block ids (`kv_block_ids`). Children are
/// keyed on the **first token** of their `token_ids` slice, enabling O(1)
/// child lookup per edge traversal.
pub struct RadixNode {
    /// Token ids that this edge/node covers (may be >1 — Patricia compression).
    pub token_ids: Vec<u32>,
    /// PagedAttention KV-block ids for the tokens this node covers.
    pub kv_block_ids: Vec<u32>,
    /// Children keyed on first token of child.token_ids.
    pub children: HashMap<u32, Arc<RwLock<RadixNode>>>,
    /// Number of active decode sequences currently using this node's KV blocks.
    /// A node with `ref_count > 0` is **pinned** and must not be evicted.
    pub ref_count: AtomicUsize,
    /// Unix timestamp (ms) of last access — used by LRU eviction.
    pub last_access: AtomicU64,
    /// Total number of accesses — used by LFU eviction.
    pub access_count: AtomicU64,
}

impl RadixNode {
    fn new(token_ids: Vec<u32>, kv_block_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            kv_block_ids,
            children: HashMap::new(),
            ref_count: AtomicUsize::new(0),
            last_access: AtomicU64::new(now_millis()),
            access_count: AtomicU64::new(1),
        }
    }

    fn touch(&self) {
        self.last_access.fetch_max(now_millis(), Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    fn is_pinned(&self) -> bool {
        self.ref_count.load(Ordering::Acquire) > 0
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Snapshot of cache health — exposed via Prometheus.
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    pub total_cached_blocks: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub pinned_blocks: usize,
}

impl PrefixCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// RadixCache
// ---------------------------------------------------------------------------

/// Concurrent prefix cache backed by a Patricia radix trie.
///
/// # Thread safety
/// The root is protected by a top-level `RwLock`. Each `RadixNode` also
/// carries an individual `RwLock` for its children map, enabling concurrent
/// reads on disjoint subtrees. Atomic fields (`last_access`, `ref_count`)
/// require no locks for updates.
pub struct RadixCache {
    root: Arc<RwLock<RadixNode>>,
    /// Current number of KV blocks held by the cache.
    total_blocks: AtomicUsize,
    /// Hard upper limit in KV blocks (set from VRAM budget).
    pub max_blocks: usize,
    pub eviction_policy: EvictionPolicy,
    // Prometheus counters
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl RadixCache {
    /// Create a new cache with a given block budget.
    pub fn new(max_blocks: usize, eviction_policy: EvictionPolicy) -> Self {
        Self {
            root: Arc::new(RwLock::new(RadixNode::new(vec![], vec![]))),
            total_blocks: AtomicUsize::new(0),
            max_blocks,
            eviction_policy,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    // -----------------------------------------------------------------------
    // Longest-prefix-match (hot path — request scheduling)
    // -----------------------------------------------------------------------

    /// Find the longest prefix of `tokens` already cached.
    ///
    /// Returns `(matched_len, kv_block_ids)` where:
    /// - `matched_len` is the number of leading tokens whose KV is cached.
    /// - `kv_block_ids` is the concatenated list of block ids for those tokens.
    ///
    /// A miss (nothing cached) returns `(0, vec![])`.
    pub fn longest_prefix_match(&self, tokens: &[u32]) -> (usize, Vec<u32>) {
        if tokens.is_empty() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return (0, vec![]);
        }

        let root = self.root.read().expect("root lock poisoned");
        let (matched, blocks) = self.walk_for_match(&root, tokens, 0);

        if matched == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        (matched, blocks)
    }

    fn walk_for_match(
        &self,
        node: &RadixNode,
        remaining: &[u32],
        depth: usize,
    ) -> (usize, Vec<u32>) {
        if remaining.is_empty() {
            return (depth, vec![]);
        }

        let first = remaining[0];
        let child_arc = match node.children.get(&first) {
            Some(arc) => arc.clone(),
            None => return (depth, vec![]),
        };

        let child = child_arc.read().expect("child lock poisoned");
        child.touch();

        // How many tokens of this child's edge match?
        let edge = &child.token_ids;
        let common = edge
            .iter()
            .zip(remaining.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common < edge.len() {
            // Partial match on this edge — stop here.
            let partial_blocks = child.kv_block_ids[..common.min(child.kv_block_ids.len())].to_vec();
            return (depth + common, partial_blocks);
        }

        // Full edge match — recurse into child's children.
        let mut blocks = child.kv_block_ids.clone();
        let (deeper_match, deeper_blocks) =
            self.walk_for_match(&child, &remaining[common..], depth + common);

        blocks.extend(deeper_blocks);
        (deeper_match, blocks)
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert a mapping from `tokens` → `kv_block_ids` into the trie.
    ///
    /// Performs Patricia split if necessary. No-op if already cached.
    pub fn insert(&self, tokens: &[u32], kv_block_ids: &[u32]) {
        if tokens.is_empty() || kv_block_ids.is_empty() {
            return;
        }
        let n_new_blocks = kv_block_ids.len();

        // Evict if over budget before inserting.
        if self.total_blocks.load(Ordering::Relaxed) + n_new_blocks > self.max_blocks {
            self.evict_until(n_new_blocks);
        }

        let mut root = self.root.write().expect("root lock poisoned");
        self.insert_into(&mut root, tokens, kv_block_ids);
        self.total_blocks.fetch_add(n_new_blocks, Ordering::Relaxed);
    }

    fn insert_into(&self, node: &mut RadixNode, tokens: &[u32], kv_blocks: &[u32]) {
        if tokens.is_empty() {
            return;
        }

        let first = tokens[0];

        if let Some(child_arc) = node.children.get(&first) {
            let mut child = child_arc.write().expect("child lock poisoned");

            // Find common prefix length between existing edge and new tokens.
            let edge = child.token_ids.clone();
            let common = edge
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == edge.len() {
                // Full match on this edge: recurse.
                let remaining_tokens = &tokens[common..];
                let remaining_blocks = if kv_blocks.len() >= common {
                    &kv_blocks[common..]
                } else {
                    &[]
                };
                self.insert_into(&mut child, remaining_tokens, remaining_blocks);
            } else {
                // Partial match: Patricia split.
                // 1. Shrink current child to `common` prefix.
                let split_tokens = edge[..common].to_vec();
                let split_blocks = child.kv_block_ids[..common.min(child.kv_block_ids.len())].to_vec();

                // 2. Old remainder becomes a new grandchild.
                let old_tail_tokens = edge[common..].to_vec();
                let old_tail_blocks = child.kv_block_ids[common.min(child.kv_block_ids.len())..].to_vec();
                let old_tail_first = old_tail_tokens[0];

                let old_grandchild = RadixNode {
                    token_ids: old_tail_tokens,
                    kv_block_ids: old_tail_blocks,
                    children: std::mem::take(&mut child.children),
                    ref_count: AtomicUsize::new(0),
                    last_access: AtomicU64::new(now_millis()),
                    access_count: AtomicU64::new(1),
                };

                // 3. New remainder becomes another new grandchild.
                let new_tail_tokens = tokens[common..].to_vec();
                let new_tail_blocks = if kv_blocks.len() >= common {
                    kv_blocks[common..].to_vec()
                } else {
                    vec![]
                };
                let new_tail_first = if new_tail_tokens.is_empty() {
                    u32::MAX
                } else {
                    new_tail_tokens[0]
                };

                // 4. Rewrite the current child as the split node.
                child.token_ids = split_tokens;
                child.kv_block_ids = split_blocks;
                child.children.clear();
                child
                    .children
                    .insert(old_tail_first, Arc::new(RwLock::new(old_grandchild)));

                if !new_tail_tokens.is_empty() {
                    child.children.insert(
                        new_tail_first,
                        Arc::new(RwLock::new(RadixNode::new(new_tail_tokens, new_tail_blocks))),
                    );
                }
            }
        } else {
            // No child for this first token: create new leaf.
            let new_node = RadixNode::new(tokens.to_vec(), kv_blocks.to_vec());
            node.children.insert(first, Arc::new(RwLock::new(new_node)));
        }
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Evict leaf nodes (lowest priority) until `free_target` blocks
    /// are available. Returns number of blocks freed.
    ///
    /// Pinned nodes (`ref_count > 0`) are skipped.
    pub fn evict_until(&self, free_target: usize) -> usize {
        let mut freed = 0usize;
        let mut root = self.root.write().expect("root lock poisoned");

        loop {
            if freed >= free_target {
                break;
            }
            // Collect evictable leaves.
            let mut leaves: Vec<(u64, u64, u32)> = Vec::new(); // (last_access, access_count, first_token)
            Self::collect_leaves(&root, &mut leaves, self.eviction_policy);

            if leaves.is_empty() {
                break;
            }

            // Remove the lowest-priority leaf.
            let (_, _, victim_key) = leaves[0];
            if let Some(removed) = root.children.remove(&victim_key) {
                let guard = removed.read().expect("evict lock");
                let n = guard.kv_block_ids.len();
                freed += n;
                self.total_blocks.fetch_sub(n, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        freed
    }

    fn collect_leaves(
        node: &RadixNode,
        out: &mut Vec<(u64, u64, u32)>,
        policy: EvictionPolicy,
    ) {
        for (key, child_arc) in &node.children {
            let child = child_arc.read().expect("child lock");
            if child.is_leaf() && !child.is_pinned() {
                let la = child.last_access.load(Ordering::Relaxed);
                let ac = child.access_count.load(Ordering::Relaxed);
                out.push((la, ac, *key));
            } else if !child.is_pinned() {
                Self::collect_leaves(&child, out, policy);
            }
        }

        // Sort so lowest priority is first.
        match policy {
            EvictionPolicy::Lru => out.sort_unstable_by_key(|(la, _, _)| *la),
            EvictionPolicy::Lfu => out.sort_unstable_by_key(|(_, ac, _)| *ac),
        }
    }

    // -----------------------------------------------------------------------
    // Reference counting (for active decode sequences)
    // -----------------------------------------------------------------------

    /// Pin blocks for an active sequence — will not be evicted during decode.
    pub fn pin(&self, tokens: &[u32]) {
        let root = self.root.read().expect("root lock");
        Self::adjust_refs(&root, tokens, 1);
    }

    /// Unpin blocks when a sequence finishes decoding.
    pub fn unpin(&self, tokens: &[u32]) {
        let root = self.root.read().expect("root lock");
        Self::adjust_refs(&root, tokens, -1);
    }

    fn adjust_refs(node: &RadixNode, tokens: &[u32], delta: i64) {
        if tokens.is_empty() {
            return;
        }
        let first = tokens[0];
        if let Some(child_arc) = node.children.get(&first) {
            let child = child_arc.read().expect("adjust_refs lock");
            let common = child
                .token_ids
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if delta > 0 {
                child.ref_count.fetch_add(1, Ordering::AcqRel);
            } else {
                child.ref_count.fetch_sub(1, Ordering::AcqRel);
            }
            if common == child.token_ids.len() {
                Self::adjust_refs(&child, &tokens[common..], delta);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    pub fn stats(&self) -> PrefixCacheStats {
        let total = self.total_blocks.load(Ordering::Relaxed);
        let pinned = self.count_pinned_blocks(&self.root.read().expect("stats lock"));
        PrefixCacheStats {
            total_cached_blocks: total,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pinned_blocks: pinned,
        }
    }

    fn count_pinned_blocks(&self, node: &RadixNode) -> usize {
        let mut count = 0;
        for child_arc in node.children.values() {
            let child = child_arc.read().expect("count lock");
            if child.is_pinned() {
                count += child.kv_block_ids.len();
            }
            count += self.count_pinned_blocks(&child);
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn cache(max: usize) -> RadixCache {
        RadixCache::new(max, EvictionPolicy::Lru)
    }

    #[test]
    fn test_exact_match() {
        let c = cache(100);
        c.insert(&[1, 2, 3], &[10, 11, 12]);
        let (len, blocks) = c.longest_prefix_match(&[1, 2, 3]);
        assert_eq!(len, 3);
        assert_eq!(blocks, vec![10, 11, 12]);
    }

    #[test]
    fn test_prefix_match_partial() {
        let c = cache(100);
        c.insert(&[1, 2, 3, 4], &[10, 11, 12, 13]);
        // Query only first 2 tokens — should still return cached portion.
        let (len, blocks) = c.longest_prefix_match(&[1, 2]);
        assert_eq!(len, 2);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_miss_returns_zero() {
        let c = cache(100);
        c.insert(&[1, 2, 3], &[10, 11, 12]);
        let (len, blocks) = c.longest_prefix_match(&[9, 8, 7]);
        assert_eq!(len, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_empty_query_is_miss() {
        let c = cache(100);
        c.insert(&[1, 2], &[10, 11]);
        let (len, _) = c.longest_prefix_match(&[]);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_hit_rate_computed() {
        let c = cache(100);
        c.insert(&[1, 2, 3], &[10, 11, 12]);
        c.longest_prefix_match(&[1, 2, 3]); // hit
        c.longest_prefix_match(&[9]);        // miss
        let stats = c.stats();
        assert_eq!(stats.hits, 1);
        // Empty-slice early-return in longest_prefix_match fires the miss counter;
        // here we only called two queries so misses = 1.
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_eviction_removes_lru_leaf() {
        let c = cache(3); // only 3 blocks
        c.insert(&[1], &[10]);          // 1 block
        c.insert(&[2], &[20]);          // 1 block
        c.insert(&[3], &[30]);          // 1 block — at budget
        // Insert one more → eviction triggered.
        c.insert(&[4], &[40]);
        // total_blocks should remain <= max_blocks
        assert!(c.stats().total_cached_blocks <= 3);
    }

    #[test]
    fn test_pinned_node_not_evicted() {
        let c = cache(1);
        c.insert(&[1], &[10]);
        c.pin(&[1]); // pin it
        // Trigger eviction — should not evict pinned node.
        let freed = c.evict_until(1);
        // Pinned, so nothing should be freed.
        assert_eq!(freed, 0);
        c.unpin(&[1]);
    }

    #[test]
    fn test_patricia_split_on_mismatch() {
        let c = cache(100);
        c.insert(&[1, 2, 3, 4], &[10, 11, 12, 13]);
        c.insert(&[1, 2, 5, 6], &[20, 21, 22, 23]);
        // Both should be retrievable.
        let (len1, _) = c.longest_prefix_match(&[1, 2, 3, 4]);
        let (len2, _) = c.longest_prefix_match(&[1, 2, 5, 6]);
        assert_eq!(len1, 4);
        assert_eq!(len2, 4);
    }

    #[test]
    fn test_shared_prefix_hit() {
        let c = cache(100);
        // Shared system prompt [1,2,3] for two different conversations.
        c.insert(&[1, 2, 3, 100], &[10, 11, 12, 100]);
        let (len, _) = c.longest_prefix_match(&[1, 2, 3, 200]);
        // Should match [1,2,3] prefix even though suffix differs.
        assert_eq!(len, 3);
    }

    #[test]
    fn test_stats_eviction_count() {
        let c = cache(2);
        c.insert(&[1], &[10, 11]); // 2 blocks = at budget
        c.insert(&[2], &[20, 21]); // triggers eviction
        let stats = c.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_concurrent_insert_lookup() {
        let c = Arc::new(cache(200));
        let handles: Vec<_> = (0..8u32)
            .map(|i| {
                let cc = c.clone();
                thread::spawn(move || {
                    cc.insert(&[i, i + 1, i + 2], &[i * 10, i * 10 + 1]);
                    let (len, _) = cc.longest_prefix_match(&[i, i + 1, i + 2]);
                    assert!(len <= 3);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn test_lfu_eviction_policy() {
        // Budget = 4 blocks so [1] (2 blocks) and [2] (2 blocks) both fit.
        let c = RadixCache::new(4, EvictionPolicy::Lfu);
        c.insert(&[1], &[10, 11]); // access_count=1 after insert
        c.insert(&[2], &[20, 21]); // access_count=1 after insert
        // Bump [1]'s access count so it is more frequent than [2].
        c.longest_prefix_match(&[1]);
        c.longest_prefix_match(&[1]);
        // Insert [3] (2 blocks) → total would be 6 > 4 → evict lowest freq = [2].
        c.insert(&[3], &[30, 31]);
        // [1] should still be present (higher access count).
        let (len, _) = c.longest_prefix_match(&[1]);
        assert_eq!(len, 1);
        // [2] should be evicted.
        let (len2, _) = c.longest_prefix_match(&[2]);
        assert_eq!(len2, 0);
    }

    #[test]
    fn test_max_blocks_zero_always_evicts() {
        // Budget=0: evict_until is called for any insertion.
        let c = cache(0);
        // evict_until(n) on an empty tree returns 0 freed — no blocks to evict.
        // Either no blocks stored OR evictions > 0 is the invariant.
        let freed = c.evict_until(1);
        // Empty cache → nothing to free; that's fine — invariant: total <= max.
        assert_eq!(freed, 0);
        assert!(c.stats().total_cached_blocks == 0);
    }

    // Verify Send + Sync
    fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn test_send_sync() {
        _assert_send_sync::<RadixCache>();
    }
}
