pub struct FlashAttention2;

impl FlashAttention2 {
    pub fn prefill(
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
            for i in 0..seq_len {
                let q_offset = (i * num_heads + h) * head_dim;
                let q_slice = &q[q_offset..q_offset + head_dim];

                let mut max_val = f32::NEG_INFINITY;
                let mut scores = vec![0.0_f32; seq_len];

                for j in 0..=i {
                    let k_offset = (j * num_heads + h) * head_dim;
                    let k_slice = &k[k_offset..k_offset + head_dim];
                    let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max_val {
                        max_val = score;
                    }
                }

                let mut sum_exp = 0.0_f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_val).exp();
                    sum_exp += scores[j];
                }

                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                let out_offset = (i * num_heads + h) * head_dim;
                let out_slice = &mut out[out_offset..out_offset + head_dim];
                out_slice.fill(0.0);

                for j in 0..=i {
                    let p = scores[j] * inv_sum;
                    let v_offset = (j * num_heads + h) * head_dim;
                    let v_slice = &v[v_offset..v_offset + head_dim];
                    for d in 0..head_dim {
                        out_slice[d] += p * v_slice[d];
                    }
                }
            }
        }
    }
}
