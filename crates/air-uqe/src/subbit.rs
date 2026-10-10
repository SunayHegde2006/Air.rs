pub struct SubBitQuantizer;

impl SubBitQuantizer {
    pub fn dequantize_bitnet_b158(packed_trits: &[u8], scale: f32, output: &mut [f32]) {
        // BitNet b1.58 packs 5 trits (-1, 0, +1) into 1 byte (3^5 = 243 < 256)
        let mut out_idx = 0;
        for &byte in packed_trits {
            let mut val = byte as u32;
            for _ in 0..5 {
                if out_idx < output.len() {
                    let trit = (val % 3) as i8 - 1;
                    output[out_idx] = trit as f32 * scale;
                    out_idx += 1;
                    val /= 3;
                }
            }
        }
    }

    pub fn dequantize_ternary_2bit(packed: &[u8], scale: f32, output: &mut [f32]) {
        for (i, &byte) in packed.iter().enumerate() {
            for shift in (0..8).step_by(2) {
                let out_i = i * 4 + (shift / 2);
                if out_i < output.len() {
                    let code = (byte >> shift) & 0x03;
                    let val = match code {
                        0 => 0.0,
                        1 => 1.0,
                        2 => -1.0,
                        _ => 0.0,
                    };
                    output[out_i] = val * scale;
                }
            }
        }
    }
}
