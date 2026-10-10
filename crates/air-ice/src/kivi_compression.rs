pub struct KiviCompressor;

impl KiviCompressor {
    pub fn compress_k_per_channel(k_uncompressed: &[f32], head_dim: usize, compressed: &mut [u8], scales: &mut [f32]) {
        let seq_len = k_uncompressed.len() / head_dim;
        for c in 0..head_dim {
            let mut max_val = 0.0_f32;
            for t in 0..seq_len {
                let val = k_uncompressed[t * head_dim + c].abs();
                if val > max_val {
                    max_val = val;
                }
            }

            let scale = max_val / 7.0;
            scales[c] = scale;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            for t in (0..seq_len).step_by(2) {
                let q0 = ((k_uncompressed[t * head_dim + c] * inv_scale).round() as i8 + 8).clamp(0, 15) as u8;
                let q1 = if t + 1 < seq_len {
                    ((k_uncompressed[(t + 1) * head_dim + c] * inv_scale).round() as i8 + 8).clamp(0, 15) as u8
                } else {
                    8
                };
                let packed_idx = (t / 2) * head_dim + c;
                if packed_idx < compressed.len() {
                    compressed[packed_idx] = q0 | (q1 << 4);
                }
            }
        }
    }

    pub fn compress_v_per_token(v_uncompressed: &[f32], head_dim: usize, compressed: &mut [u8], scales: &mut [f32]) {
        let seq_len = v_uncompressed.len() / head_dim;
        for t in 0..seq_len {
            let mut max_val = 0.0_f32;
            for c in 0..head_dim {
                let val = v_uncompressed[t * head_dim + c].abs();
                if val > max_val {
                    max_val = val;
                }
            }

            let scale = max_val / 7.0;
            scales[t] = scale;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            for c in (0..head_dim).step_by(2) {
                let q0 = ((v_uncompressed[t * head_dim + c] * inv_scale).round() as i8 + 8).clamp(0, 15) as u8;
                let q1 = ((v_uncompressed[t * head_dim + c + 1] * inv_scale).round() as i8 + 8).clamp(0, 15) as u8;
                let packed_idx = t * (head_dim / 2) + (c / 2);
                if packed_idx < compressed.len() {
                    compressed[packed_idx] = q0 | (q1 << 4);
                }
            }
        }
    }
}
