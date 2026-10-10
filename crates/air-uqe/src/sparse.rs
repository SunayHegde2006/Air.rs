pub struct SparseQuantizer;

impl SparseQuantizer {
    pub fn dequantize_block_sparse(
        non_zero_values: &[f32],
        block_indices: &[u32],
        block_size: usize,
        output: &mut [f32],
    ) {
        output.fill(0.0);
        for (b_idx, &target_block) in block_indices.iter().enumerate() {
            let src_offset = b_idx * block_size;
            let dst_offset = target_block as usize * block_size;

            if dst_offset + block_size <= output.len() && src_offset + block_size <= non_zero_values.len() {
                output[dst_offset..dst_offset + block_size]
                    .copy_from_slice(&non_zero_values[src_offset..src_offset + block_size]);
            }
        }
    }
}
