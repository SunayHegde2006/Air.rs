pub struct FlashAttention3;

impl FlashAttention3 {
    pub fn prefill_sm90(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        // Fallback to FA2 logic on non-Hopper hardware
        crate::attention::fa2::FlashAttention2::prefill(q, k, v, out, seq_len, num_heads, head_dim);
    }
}
