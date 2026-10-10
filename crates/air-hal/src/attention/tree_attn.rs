#[derive(Debug, Clone)]
pub struct TreeMask {
    pub num_candidates: usize,
    pub mask: Vec<bool>, // num_candidates x num_candidates
}

impl TreeMask {
    pub fn new(num_candidates: usize, mask: Vec<bool>) -> Self {
        Self {
            num_candidates,
            mask,
        }
    }

    pub fn is_allowed(&self, q_idx: usize, k_idx: usize) -> bool {
        if q_idx >= self.num_candidates || k_idx >= self.num_candidates {
            return false;
        }
        self.mask[q_idx * self.num_candidates + k_idx]
    }
}

pub struct TreeAttention;

impl TreeAttention {
    pub fn verify_draft(
        q_candidates: &[f32],
        k: &[f32],
        v: &[f32],
        tree_mask: &TreeMask,
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let num_cand = tree_mask.num_candidates;
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..num_heads {
            for cand in 0..num_cand {
                let q_offset = (cand * num_heads + h) * head_dim;
                let q_slice = &q_candidates[q_offset..q_offset + head_dim];

                let out_offset = (cand * num_heads + h) * head_dim;
                let out_slice = &mut out[out_offset..out_offset + head_dim];
                out_slice.fill(0.0);

                let mut scores = vec![0.0_f32; seq_len + num_cand];
                let mut max_val = f32::NEG_INFINITY;

                // Past sequence tokens
                for t in 0..seq_len {
                    let k_offset = (t * num_heads + h) * head_dim;
                    let k_slice = &k[k_offset..k_offset + head_dim];
                    let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
                    let s = dot * scale;
                    scores[t] = s;
                    if s > max_val {
                        max_val = s;
                    }
                }

                // Candidate tree tokens
                for k_cand in 0..num_cand {
                    if tree_mask.is_allowed(cand, k_cand) {
                        let k_offset = ((seq_len + k_cand) * num_heads + h) * head_dim;
                        let k_slice = &k[k_offset..k_offset + head_dim];
                        let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
                        let s = dot * scale;
                        scores[seq_len + k_cand] = s;
                        if s > max_val {
                            max_val = s;
                        }
                    } else {
                        scores[seq_len + k_cand] = f32::NEG_INFINITY;
                    }
                }

                let mut sum_exp = 0.0_f32;
                for t in 0..(seq_len + num_cand) {
                    if scores[t] > f32::NEG_INFINITY {
                        scores[t] = (scores[t] - max_val).exp();
                        sum_exp += scores[t];
                    } else {
                        scores[t] = 0.0;
                    }
                }

                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                for t in 0..seq_len {
                    let p = scores[t] * inv_sum;
                    let v_offset = (t * num_heads + h) * head_dim;
                    let v_slice = &v[v_offset..v_offset + head_dim];
                    for d in 0..head_dim {
                        out_slice[d] += p * v_slice[d];
                    }
                }

                for k_cand in 0..num_cand {
                    if tree_mask.is_allowed(cand, k_cand) {
                        let p = scores[seq_len + k_cand] * inv_sum;
                        let v_offset = ((seq_len + k_cand) * num_heads + h) * head_dim;
                        let v_slice = &v[v_offset..v_offset + head_dim];
                        for d in 0..head_dim {
                            out_slice[d] += p * v_slice[d];
                        }
                    }
                }
            }
        }
    }
}
