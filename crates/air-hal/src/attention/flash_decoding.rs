pub struct FlashDecodingPlusPlus;

impl FlashDecodingPlusPlus {
    pub const SPLIT_K: usize = 56;

    pub fn decode(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..num_heads {
            let q_slice = &q[h * head_dim..(h + 1) * head_dim];
            let out_slice = &mut out[h * head_dim..(h + 1) * head_dim];
            out_slice.fill(0.0);

            let mut scores = vec![0.0_f32; seq_len];
            let mut max_val = f32::NEG_INFINITY;

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

            let mut sum_exp = 0.0_f32;
            for t in 0..seq_len {
                scores[t] = (scores[t] - max_val).exp();
                sum_exp += scores[t];
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
        }
    }
}
