pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    pos: usize,
    rope_theta: f32,
) {
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
        let val = pos as f32 * freq;
        let cos = val.cos();
        let sin = val.sin();

        // Process Q
        for head_q in q.chunks_exact_mut(head_dim) {
            let q0 = head_q[i];
            let q1 = head_q[i + half_dim];
            head_q[i] = q0 * cos - q1 * sin;
            head_q[i + half_dim] = q0 * sin + q1 * cos;
        }

        // Process K
        for head_k in k.chunks_exact_mut(head_dim) {
            let k0 = head_k[i];
            let k1 = head_k[i + half_dim];
            head_k[i] = k0 * cos - k1 * sin;
            head_k[i + half_dim] = k0 * sin + k1 * cos;
        }
    }
}
