pub fn dequantize_q4_0(packed: &[u8], scales: &[f32], output: &mut [f32]) {
    let block_size = 32;
    let num_blocks = packed.len() / (block_size / 2);

    for b in 0..num_blocks {
        let scale = scales[b];
        let block_packed = &packed[b * 16..(b + 1) * 16];
        let out_block = &mut output[b * block_size..(b + 1) * block_size];

        for i in 0..16 {
            let byte = block_packed[i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            out_block[i] = q0 as f32 * scale;
            out_block[i + 16] = q1 as f32 * scale;
        }
    }
}

pub fn dequantize_q8_0(packed: &[i8], scales: &[f32], output: &mut [f32]) {
    let block_size = 32;
    let num_blocks = packed.len() / block_size;

    for b in 0..num_blocks {
        let scale = scales[b];
        let block_packed = &packed[b * block_size..(b + 1) * block_size];
        let out_block = &mut output[b * block_size..(b + 1) * block_size];

        for i in 0..block_size {
            out_block[i] = block_packed[i] as f32 * scale;
        }
    }
}
