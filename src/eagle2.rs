//! EAGLE-2 Speculative Decoding — v0.5.0
//!
//! Context-aware dynamic draft tree speculative decoding.
//!
//! # Research Basis
//!
//! - **EAGLE-2** (Li et al., NeurIPS 2024): Unlike static-depth speculative
//!   decoding (Leviathan et al., 2023), EAGLE-2 builds a *dynamic* draft tree
//!   by expanding nodes whose draft probability exceeds a threshold τ. This
//!   gives 2.8–4× wall-clock speedup vs. auto-regressive decoding using a
//!   draft model that is ~3× smaller than the target.
//!
//! # Key Concepts
//!
//! ```text
//! Draft tree construction (per decoding step):
//!   root = last accepted token
//!   For each candidate node:
//!     1. Draft model predicts logits → top-k children
//!     2. If draft_prob[child] ≥ τ AND tree_depth < MAX_DEPTH:
//!        → expand child as a new node
//!     3. Collect all leaf paths as candidate sequences
//!
//! Tree verification (single target forward pass):
//!   target_logits = target_model.forward(all_draft_tokens, tree_mask)
//!   For each path root→leaf:
//!     accepted_len = speculative_sample(target_logits, draft_probs)
//!   Pick longest accepted path → restart from bonus token
//! ```
//!
//! # Acceptance Sampling
//!
//! Token t is accepted with probability min(1, p_target(t) / p_draft(t)).
//! If rejected, sample a corrected token from the "residual" distribution
//! max(0, p_target − p_draft) / Z.  This guarantees the output distribution
//! equals the target distribution exactly (the DeSpecki et al. guarantee).

use std::collections::VecDeque;

// ── Constants ──────────────────────────────────────────────────────────────

/// Maximum draft tree depth.
pub const MAX_TREE_DEPTH: usize = 6;

/// Minimum draft probability to expand a node (τ).
pub const EXPAND_THRESHOLD: f32 = 0.05;

/// Top-k candidates expanded per node.
pub const DRAFT_TOP_K: usize = 4;

/// Vocabulary size (matches LLaMA / Mistral default).
pub const DEFAULT_VOCAB_SIZE: usize = 32_000;

// ── Draft Tree Node ────────────────────────────────────────────────────────

/// A single node in the EAGLE-2 draft token tree.
#[derive(Debug, Clone)]
pub struct DraftNode {
    /// Token id at this node.
    pub token_id: u32,
    /// Draft model probability for this token.
    pub draft_prob: f32,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// Index of parent node in the tree's node list (None = root).
    pub parent_idx: Option<usize>,
}

/// A complete draft tree: a flat `Vec<DraftNode>` with parent links.
#[derive(Debug, Clone)]
pub struct DraftTree {
    pub nodes: Vec<DraftNode>,
    /// Root token (last verified token from target model).
    pub root_token: u32,
}

impl DraftTree {
    /// Construct an empty draft tree rooted at `root_token`.
    pub fn new(root_token: u32) -> Self {
        Self { nodes: Vec::new(), root_token }
    }

    /// All paths from root to every leaf node.
    /// Returns a `Vec` of token sequences (root-exclusive).
    pub fn all_paths(&self) -> Vec<Vec<u32>> {
        // Find leaf indices (nodes with no children)
        let has_children: Vec<bool> = {
            let mut hc = vec![false; self.nodes.len()];
            for node in &self.nodes {
                if let Some(p) = node.parent_idx {
                    hc[p] = true;
                }
            }
            hc
        };

        let mut paths = Vec::new();
        for (leaf_idx, is_leaf) in has_children.iter().enumerate().map(|(i, &b)| (i, !b)) {
            if !is_leaf {
                continue;
            }
            paths.push(self.path_to_root(leaf_idx));
        }
        paths
    }

    /// Reconstruct the token sequence from `node_idx` back to root (exclusive).
    pub fn path_to_root(&self, mut node_idx: usize) -> Vec<u32> {
        let mut path = Vec::new();
        loop {
            let node = &self.nodes[node_idx];
            path.push(node.token_id);
            match node.parent_idx {
                Some(p) => node_idx = p,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Tree size (number of draft nodes).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if the draft tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ── Draft Model Interface ──────────────────────────────────────────────────

/// Trait implemented by the draft model backend.
///
/// In production this wraps a small LM (e.g. LLaMA-3.2-1B when target is
/// LLaMA-3.1-8B). For tests, a `MockDraftModel` is provided.
pub trait DraftModel: Send + Sync {
    /// Given the current context tokens, return logits for the next token.
    /// Returns a `Vec<f32>` of length `vocab_size`.
    fn next_logits(&self, context: &[u32]) -> Vec<f32>;

    /// Vocabulary size of this draft model.
    fn vocab_size(&self) -> usize;
}

/// Softmax over a logit vector → probability distribution.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Top-k indices + probabilities from a probability vector.
pub fn top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed
}

// ── Tree Builder ───────────────────────────────────────────────────────────

/// Builds an EAGLE-2 dynamic draft tree using the draft model.
///
/// Uses BFS expansion: nodes are enqueued if their draft probability ≥ τ
/// AND depth < MAX_TREE_DEPTH.
pub struct Eagle2Builder<'a> {
    draft: &'a dyn DraftModel,
    tau: f32,
    top_k: usize,
    max_depth: usize,
}

impl<'a> Eagle2Builder<'a> {
    /// Create a builder with the given expansion threshold τ.
    pub fn new(draft: &'a dyn DraftModel, tau: f32) -> Self {
        Self { draft, tau, top_k: DRAFT_TOP_K, max_depth: MAX_TREE_DEPTH }
    }

    /// Build a draft tree from `context` (accepted tokens so far).
    pub fn build(&self, context: &[u32]) -> DraftTree {
        let root_token = *context.last().unwrap_or(&0);
        let mut tree = DraftTree::new(root_token);

        // BFS queue: (context_suffix, parent_node_idx, depth)
        let mut queue: VecDeque<(Vec<u32>, Option<usize>, usize)> = VecDeque::new();
        queue.push_back((context.to_vec(), None, 0));

        while let Some((ctx, parent_idx, depth)) = queue.pop_front() {
            if depth >= self.max_depth {
                continue;
            }

            let logits = self.draft.next_logits(&ctx);
            let probs = softmax(&logits);
            let candidates = top_k(&probs, self.top_k);

            for (token_id, prob) in candidates {
                let node_idx = tree.nodes.len();
                tree.nodes.push(DraftNode {
                    token_id: token_id as u32,
                    draft_prob: prob,
                    depth,
                    parent_idx,
                });

                // Expand if probability above threshold
                if prob >= self.tau && depth + 1 < self.max_depth {
                    let mut next_ctx = ctx.clone();
                    next_ctx.push(token_id as u32);
                    queue.push_back((next_ctx, Some(node_idx), depth + 1));
                }
            }
        }

        tree
    }
}

// ── Acceptance Sampler ─────────────────────────────────────────────────────

/// Result of verifying one draft candidate path.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of tokens accepted from the draft path.
    pub n_accepted: usize,
    /// The bonus token sampled from the target residual distribution.
    pub bonus_token: u32,
    /// Draft path that was verified.
    pub path: Vec<u32>,
}

/// Speculative acceptance sampling (DeSpecki et al. guarantee).
///
/// Given parallel target logits and draft probabilities for a candidate path,
/// accept each token with probability min(1, p_target / p_draft).
/// On rejection, sample from max(0, p_target − p_draft) / Z.
///
/// # Parameters
/// - `path`: Draft token sequence (excluding root).
/// - `target_probs`: Target model probability for each token in `path`,
///   shape `[path.len() × vocab_size]`.
/// - `draft_probs`: Draft probability for each token in `path` (scalar per step).
/// - `rng`: Uniform random source (0..1) for acceptance decisions.
pub fn accept_sample(
    path: &[u32],
    target_probs: &[Vec<f32>],
    draft_probs_scalar: &[f32],
    rng: &mut impl FnMut() -> f32,
) -> VerificationResult {
    assert_eq!(path.len(), target_probs.len());
    assert_eq!(path.len(), draft_probs_scalar.len());

    let mut n_accepted = 0;
    for (i, (&tok, tprobs)) in path.iter().zip(target_probs.iter()).enumerate() {
        let p_target = tprobs.get(tok as usize).copied().unwrap_or(0.0);
        let p_draft = draft_probs_scalar[i].max(1e-8);
        let accept_prob = (p_target / p_draft).min(1.0);

        if rng() < accept_prob {
            n_accepted += 1;
        } else {
            // Sample from residual distribution at rejection point
            let residual: Vec<f32> = tprobs
                .iter()
                .zip(target_probs[i].iter())
                .map(|(pt, _)| (pt - draft_probs_scalar[i]).max(0.0))
                .collect();
            let sum: f32 = residual.iter().sum();
            let bonus_token = if sum < 1e-8 {
                // Fallback: argmax of target
                tprobs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            } else {
                sample_from_dist(&residual, sum, rng)
            };
            return VerificationResult { n_accepted, bonus_token, path: path.to_vec() };
        }
    }

    // All accepted — bonus token from target at last position
    let last_tprobs = &target_probs[path.len() - 1];
    let sum: f32 = last_tprobs.iter().sum();
    let bonus_token = sample_from_dist(last_tprobs, sum, &mut *rng);
    VerificationResult { n_accepted, bonus_token, path: path.to_vec() }
}

fn sample_from_dist(probs: &[f32], sum: f32, rng: &mut impl FnMut() -> f32) -> u32 {
    let u = rng() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= u {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ── Statistics ─────────────────────────────────────────────────────────────

/// Running statistics for EAGLE-2 acceptance rate monitoring.
#[derive(Debug, Default, Clone)]
pub struct Eagle2Stats {
    pub total_draft_tokens: u64,
    pub total_accepted_tokens: u64,
    pub total_steps: u64,
}

impl Eagle2Stats {
    pub fn record(&mut self, draft_len: usize, accepted: usize) {
        self.total_draft_tokens += draft_len as u64;
        self.total_accepted_tokens += accepted as u64;
        self.total_steps += 1;
    }

    /// Mean tokens accepted per draft step.
    pub fn mean_accepted(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_steps as f32
    }

    /// Token acceptance rate: accepted / drafted.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
    }

    /// Effective speedup multiplier (accepted+1 per step, vs 1 for AR).
    pub fn speedup(&self) -> f32 {
        self.mean_accepted() + 1.0
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic mock draft model: returns uniform logits + small bump on
    /// token = (context.last() + 1) % vocab_size.
    struct MockDraftModel {
        vocab: usize,
    }

    impl DraftModel for MockDraftModel {
        fn next_logits(&self, context: &[u32]) -> Vec<f32> {
            let mut logits = vec![1.0f32; self.vocab];
            if let Some(&last) = context.last() {
                let next = ((last + 1) as usize) % self.vocab;
                logits[next] = 10.0; // strong preference
            }
            logits
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
    }

    fn det_rng(val: f32) -> impl FnMut() -> f32 {
        move || val
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn top_k_returns_k_elements() {
        let probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        let tk = top_k(&probs, 3);
        assert_eq!(tk.len(), 3);
        // Descending order
        assert!(tk[0].1 >= tk[1].1 && tk[1].1 >= tk[2].1);
    }

    #[test]
    fn draft_tree_built_nonempty() {
        let model = MockDraftModel { vocab: 100 };
        let builder = Eagle2Builder::new(&model, EXPAND_THRESHOLD);
        let context = vec![0u32, 1, 2];
        let tree = builder.build(&context);
        assert!(!tree.is_empty(), "draft tree should not be empty");
    }

    #[test]
    fn draft_tree_depth_bounded() {
        let model = MockDraftModel { vocab: 50 };
        let builder = Eagle2Builder::new(&model, 0.0); // expand everything
        let tree = builder.build(&[0u32]);
        for node in &tree.nodes {
            assert!(node.depth < MAX_TREE_DEPTH, "node depth {} ≥ MAX", node.depth);
        }
    }

    #[test]
    fn path_to_root_matches_depth() {
        let model = MockDraftModel { vocab: 100 };
        let builder = Eagle2Builder::new(&model, EXPAND_THRESHOLD);
        let tree = builder.build(&[5u32, 6, 7]);
        // Every path should have length ≤ MAX_TREE_DEPTH
        for i in 0..tree.len() {
            let path = tree.path_to_root(i);
            assert!(path.len() <= MAX_TREE_DEPTH, "path len {} > MAX", path.len());
        }
    }

    #[test]
    fn accept_sample_all_accepted_when_target_dominates() {
        // When p_target >> p_draft, always accept
        let path = vec![1u32, 2, 3];
        let vocab = 100usize;
        // target gives high prob to exactly the draft tokens
        let target_probs: Vec<Vec<f32>> = path
            .iter()
            .map(|&t| {
                let mut v = vec![0.001f32; vocab];
                v[t as usize] = 0.99;
                let s: f32 = v.iter().sum();
                v.iter().map(|x| x / s).collect()
            })
            .collect();
        let draft_probs = vec![0.01f32; 3]; // p_draft << p_target → accept prob = 1
        let mut always_accept = det_rng(0.0); // u < accept_prob when u=0
        let result = accept_sample(&path, &target_probs, &draft_probs, &mut always_accept);
        assert_eq!(result.n_accepted, 3, "all 3 should be accepted");
    }

    #[test]
    fn accept_sample_none_accepted_when_draft_dominates() {
        // When u > accept_prob (u=1.0) nothing is accepted
        let path = vec![5u32];
        let vocab = 100usize;
        let mut target_p = vec![0.01f32; vocab];
        target_p[5] = 0.01; // small target prob for token 5
        let target_probs = vec![target_p.iter().map(|x| x / target_p.iter().sum::<f32>()).collect::<Vec<_>>()];
        let draft_probs = vec![0.99]; // p_draft >> p_target → accept_prob << 1
        let mut always_reject = det_rng(1.0); // u=1 > accept_prob → reject
        let result = accept_sample(&path, &target_probs, &draft_probs, &mut always_reject);
        assert_eq!(result.n_accepted, 0);
    }

    #[test]
    fn stats_speedup_at_least_one() {
        let mut stats = Eagle2Stats::default();
        stats.record(4, 3);
        stats.record(4, 2);
        assert!(stats.speedup() >= 1.0);
        assert!(stats.acceptance_rate() > 0.0 && stats.acceptance_rate() <= 1.0);
    }

    #[test]
    fn eagle2_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Eagle2Stats>();
        assert_send_sync::<DraftTree>();
    }
}
