pub fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let dim = input.len();
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let scale = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

    for i in 0..dim {
        output[i] = input[i] * scale * weight[i];
    }
}

pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;

    for val in logits.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in logits.iter_mut() {
            *val *= inv_sum;
        }
    }
}
