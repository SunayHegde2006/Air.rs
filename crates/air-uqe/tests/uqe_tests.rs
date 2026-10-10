use air_uqe::format_map::{map_format, FamilyId};
use air_uqe::affine_block::dequantize_q4_0;

#[test]
fn test_format_mapping() {
    assert_eq!(map_format("Q4_0"), FamilyId::F1);
    assert_eq!(map_format("IQ2_XXS"), FamilyId::F2);
    assert_eq!(map_format("FP8-E4M3"), FamilyId::F3);
}

#[test]
fn test_q4_0_dequant() {
    let raw_packed = vec![0u8; 16]; // 1 block of 32 values
    let scales = vec![1.0f32];
    let mut out = vec![0.0_f32; 32];
    dequantize_q4_0(&raw_packed, &scales, &mut out);
    assert_eq!(out.len(), 32);
}
