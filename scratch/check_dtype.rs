fn main() {
    // Try converting u32 to GgmlDType using TryFrom or from_u32 if they exist
    // Let's just try to compile a mock reference to see if we can find if it's supported.
    let _t = candle_core::quantized::GgmlDType::Q4_0;
    println!("Q4_0 exists");
}
