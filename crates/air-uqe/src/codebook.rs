pub struct CodebookQuantizer;

impl CodebookQuantizer {
    pub fn dequantize_iq4_nl(packed: &[u8], scales: &[f32], lut: &[f32; 16], output: &mut [f32]) {
        let block_size = 32;
        let num_blocks = packed.len() / (block_size / 2);

        for b in 0..num_blocks {
            let scale = scales[b];
            let block_packed = &packed[b * 16..(b + 1) * 16];
            let out_block = &mut output[b * block_size..(b + 1) * block_size];

            for i in 0..16 {
                let byte = block_packed[i];
                let idx0 = (byte & 0x0F) as usize;
                let idx1 = ((byte >> 4) & 0x0F) as usize;
                out_block[i] = lut[idx0] * scale;
                out_block[i + 16] = lut[idx1] * scale;
            }
        }
    }
}
