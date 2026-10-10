pub struct FpMicroscaleQuantizer;

impl FpMicroscaleQuantizer {
    pub fn dequantize_fp8_e4m3(packed: &[u8], scale: f32, output: &mut [f32]) {
        for (i, &byte) in packed.iter().enumerate() {
            let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
            let exp = ((byte >> 3) & 0x0F) as i32 - 7;
            let mant = (byte & 0x07) as f32 / 8.0;
            let val = if exp == -7 {
                sign * mant * 2.0_f32.powi(-6)
            } else {
                sign * (1.0 + mant) * 2.0_f32.powi(exp)
            };
            output[i] = val * scale;
        }
    }
}
