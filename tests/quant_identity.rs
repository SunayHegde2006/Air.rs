// Verification of Quantization Bit-Identity (Wave 1)
//
// This test ensures that the new production-ready dequantization kernels
// for IQ2_XXS and EXL2 produce bit-exact results for known patterns.

use air_rs::iq_quant::{dequantize_iq, IqFormat};
use air_rs::alt_quant::{Exl2Layer};
use candle_core::{Device, Tensor, DType};

#[test]
fn test_iq2_xxs_identity() {
    let device = Device::Cpu;
    // IQ2_XXS: 256 weights, 66 bytes
    // scale (f16: 1.0) = 0x3C00
    let mut data = vec![0x00, 0x3C];
    // Fill with a pattern of indices (0..64)
    for i in 0..64 {
        data.push(i as u8);
    }
    
    let t = dequantize_iq(&data, 1, 256, IqFormat::Iq2xxs, &device).unwrap();
    let vals = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    
    // First group (idx 0) should use IQ2XXS_GRID[0] = [-1, -1, -1, -1]
    // with scale 1.0 -> [-1.0, -1.0, -1.0, -1.0]
    assert_eq!(vals[0], -1.0);
    assert_eq!(vals[1], -1.0);
    assert_eq!(vals[2], -1.0);
    assert_eq!(vals[3], -1.0);
}

#[test]
fn test_exl2_variable_bit_identity() {
    let device = Device::Cpu;
    
    // Mock EXL2 layer: 32 rows, 1 col, 4-bit weights
    // q_groups: [offset=0, bits=4]
    let q_groups = Tensor::from_vec(vec![0i64, 4i64], (2,), &device).unwrap();
    // q_weight: 4 entries (4 bits/weight * 32 weights = 128 bits = 4 * u32)
    // pattern: all 15 (max 4-bit)
    let q_weight = Tensor::from_vec(vec![
        0xFFFFFFFFu32 as i64, 
        0xFFFFFFFFu32 as i64, 
        0xFFFFFFFFu32 as i64, 
        0xFFFFFFFFu32 as i64
    ], (4, 1), &device).unwrap()
        .to_dtype(DType::I64).unwrap();
    // q_scale: 1.0
    let q_scale = Tensor::from_vec(vec![1.0f32], (1,), &device).unwrap();
    // q_scale_max: 256.0
    let q_scale_max = Tensor::from_vec(vec![256.0f32], (1,), &device).unwrap();
    
    let layer = Exl2Layer {
        q_weight,
        q_scale,
        q_scale_max,
        q_groups,
        q_perm: None,
    };
    
    let t = layer.dequantize().unwrap();
    let vals = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    
    // bits=4, center=8.0, scale=1.0*256/256 = 1.0
    // q=15 -> w = 1.0 * (15 - 8) = 7.0
    for v in vals.iter() {
        assert_eq!(*v, 7.0);
    }
}
