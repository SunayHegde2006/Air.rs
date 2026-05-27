use air_rs::generator::InferenceGenerator;
use air_rs::model::ModelConfig;
use air_rs::medusa_heads::MedusaHeads;
use air_rs::sparsity_predictor::SparsityPredictorBank;
use air_rs::wavefront::compute_union_mask;
use candle_core::{Device, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wcpsr_pipeline_integration() -> anyhow::Result<()> {
        let device = Device::Cpu;
        
        // 1. Setup Mock Config
        let config = ModelConfig {
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 2,
            hidden_dim: 128,
            intermediate_dim: 512,
            vocab_size: 100,
            rope_theta: 1000000.0,
            context_length: 2048,
            rms_norm_eps: 1e-6,
            head_dim: 128,
            arch: air_rs::model_variant::ModelVariant::Qwen3_6,
            norm_type: air_rs::model_variant::NormType::RmsNorm,
            ffn_type: air_rs::model_variant::FfnType::SwiGlu,
            sliding_window: None,
            partial_rope_factor: None,
            attn_router: air_rs::attention_backend::HybridAttentionRouter::uniform(2, air_rs::attention_backend::AttentionBackend::Softmax),
            n_experts: 0,
            moe_top_k: 0,
            eos_token_id: 2,
        };

        // 2. Initialize Components
        let sampler = air_rs::sampler::SamplerConfig::default();
        let _generator = InferenceGenerator::with_device(config.clone(), sampler, device.clone())?;
        
        // 3. Mock a Hidden State [1, 128]
        let h = Tensor::randn(0.0f32, 1.0f32, (1, 128), &device)?.to_dtype(candle_core::DType::F16)?;
        let lm_head = Tensor::randn(0.0f32, 1.0f32, (100, 128), &device)?.to_dtype(candle_core::DType::F16)?;

        // 4. Test Draft Cycle
        let medusa = MedusaHeads::new_random(air_rs::medusa_heads::MedusaConfig {
            n_heads: 4,
            hidden_dim: 128,
            vocab_size: 100,
            ..Default::default()
        }, &device)?;
        
        let bundle = medusa.draft(&h, &lm_head, &[])?;
        assert_eq!(bundle.n_heads, 4);

        // 5. Test Sparsity Planning
        let bank = SparsityPredictorBank::new(2, air_rs::sparsity_predictor::SparsityConfig {
            hidden_dim: 128,
            intermediate_dim: 512,
            ..Default::default()
        });
        
        let mut layer_preds = Vec::new();
        for i in 0..2 {
            let mask = bank.predict_mask(&h, i)?;
            layer_preds.push(mask.active_indices);
        }
        let union_mask = compute_union_mask(&layer_preds, 512);
        assert!(union_mask.density >= 0.0);

        Ok(())
    }
}
