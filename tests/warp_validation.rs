use air_rs::pd_disagg::KvBlock;
use air_rs::warp_protocol::{WarpHeader, WarpFeature};
use candle_core::{Device, DType};

#[test]
fn test_warp_p2p_int8_quant() -> Result<(), Box<dyn std::error::Error>> {
    let _device = Device::Cpu;
    
    // 1. Create a dummy f16 block (bf16 bytes)
    let data = vec![0.5f32, -1.0, 2.0, 0.0];
    let bf16_data: Vec<u8> = data.iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
        
    let block = KvBlock {
        block_id: 42,
        layer_idx: 1,
        data: bf16_data.clone(),
    };

    // 2. Quantize
    let (q_data, scale) = KvBlock::int8_quantize(&block.data);
    assert_eq!(q_data.len(), 4);
    assert!(scale > 0.0);
    
    // 3. Serialise/Deserialise Header
    let header = WarpHeader {
        features: WarpFeature::Int8Quantization as u64 | WarpFeature::AsyncPipelining as u64,
        body_len: 1024,
    };
    let buf = header.serialise();
    let decoded = WarpHeader::deserialise(&buf[..])?;
    assert_eq!(decoded.features, header.features);
    assert_eq!(decoded.body_len, 1024);

    Ok(())
}

#[test]
fn test_ghost_gemma_draft_envelope() -> Result<(), Box<dyn std::error::Error>> {
    use air_rs::medusa_heads::{MedusaHeads, MedusaConfig};
    use candle_core::Tensor;
    
    let device = Device::Cpu;
    let medusa = MedusaHeads::new_random(MedusaConfig {
        n_heads: 2,
        hidden_dim: 128,
        vocab_size: 1000,
        ..Default::default()
    }, &device)?;
    
    let h = Tensor::randn(0.0f32, 1.0f32, (1, 128), &device)?.to_dtype(DType::F16)?;
    let lm_head = Tensor::randn(0.0f32, 1.0f32, (1000, 128), &device)?.to_dtype(DType::F16)?;
    
    let envelope = medusa.draft_envelope(&h, &lm_head)?;
    assert_eq!(envelope.positions.len(), 2);
    assert_eq!(envelope.positions[0].candidates.len(), 5);
    
    // Verify sorting
    let c = envelope.positions[0].candidates;
    assert!(c[0].1 >= c[1].1);
    
    Ok(())
}
