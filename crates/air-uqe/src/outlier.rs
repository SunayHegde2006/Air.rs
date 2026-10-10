pub struct OutlierSplitQuantizer;

impl OutlierSplitQuantizer {
    pub fn dequantize_int8_outlier(
        bulk_int8: &[i8],
        bulk_scales: &[f32],
        outlier_indices: &[usize],
        outlier_values: &[f32],
        output: &mut [f32],
    ) {
        for (i, &val) in bulk_int8.iter().enumerate() {
            output[i] = val as f32 * bulk_scales[i % bulk_scales.len()];
        }
        for (&idx, &val) in outlier_indices.iter().zip(outlier_values.iter()) {
            if idx < output.len() {
                output[idx] = val;
            }
        }
    }
}
