//! P8 — Vision Encoder (SigLIP / CLIP)
//!
//! Vision encoders for multimodal LLMs — embeds image patches into the same
//! hidden dim as text tokens so they can be fused with the language model.
//!
//! # Supported Architectures
//!
//! | Model | Encoder | Patch | Resolution | Output dim |
//! |-------|---------|-------|------------|------------|
//! | LLaVA 1.5/1.6 | CLIP ViT-L/14 | 14px | 336×336 | 1024 |
//! | PaliGemma | SigLIP ViT-So/14 | 14px | 224×224 | 1152 |
//! | Gemma 3 Vision | SigLIP 2 | 16px | 896×896 | 1152 |
//! | Qwen2-VL | ViT + NaViT | 14px | dynamic | 1536 |
//!
//! # Architecture (ViT / SigLIP)
//! ```text
//! image → patch_embed (conv 14×14) → flatten → + pos_embed → transformer → proj → [patch_tokens]
//! ```
//!
//! The patch tokens are then projected to the language model's hidden_dim
//! and concatenated with text token embeddings.
//!
//! # SigLIP vs CLIP
//! - CLIP uses softmax contrastive loss (min cosine sim across batch)
//! - SigLIP uses sigmoid binary classification loss (SigmoidLoss)
//! - Architecture is nearly identical — same ViT + projection; only training loss differs
//! - At inference, no difference: both produce patch embeddings via ViT forward pass

use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor, D};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Vision encoder variant
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VisionEncoderType {
    /// OpenAI CLIP ViT (LLaVA 1.5/1.6)
    ClipVit,
    /// Google SigLIP ViT (PaliGemma, Gemma 3 Vision)
    SigLip,
    /// Qwen2-VL dynamic resolution ViT
    QwenVl,
}

/// Configuration for one vision encoder.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub encoder_type: VisionEncoderType,
    /// Input image resolution (height = width for square)
    pub image_size: usize,
    /// Patch size in pixels
    pub patch_size: usize,
    /// Number of visual tokens per image = (image_size / patch_size)²
    pub num_patches: usize,
    /// Number of transformer layers in the vision encoder
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Hidden dimension of the vision transformer
    pub hidden_dim: usize,
    /// MLP intermediate dimension (usually hidden * 4)
    pub mlp_dim: usize,
    /// Projection output dimension (language model hidden dim)
    pub proj_dim: usize,
    /// Whether to include CLS token
    pub use_cls_token: bool,
}

impl VisionConfig {
    /// CLIP ViT-L/14 @ 336px (LLaVA 1.5 / 1.6)
    pub fn clip_vit_l_336() -> Self {
        let image_size: usize = 336;
        let patch_size: usize = 14;
        let n = image_size / patch_size;
        let num_patches = n * n;
        Self {
            encoder_type: VisionEncoderType::ClipVit,
            image_size,
            patch_size,
            num_patches,
            n_layers: 24,
            n_heads: 16,
            hidden_dim: 1024,
            mlp_dim: 4096,
            proj_dim: 4096, // LLaVA projects to language model hidden (Vicuna 7B = 4096)
            use_cls_token: true,
        }
    }

    /// SigLIP ViT-So/14 @ 224px (PaliGemma 3B)
    pub fn siglip_so14_224() -> Self {
        let image_size: usize = 224;
        let patch_size: usize = 14;
        let n = image_size / patch_size;
        let num_patches = n * n;
        Self {
            encoder_type: VisionEncoderType::SigLip,
            image_size,
            patch_size,
            num_patches,
            n_layers: 27,
            n_heads: 16,
            hidden_dim: 1152,
            mlp_dim: 4304,
            proj_dim: 2048, // PaliGemma 3B language model hidden
            use_cls_token: false, // SigLIP does not use CLS
        }
    }

    /// SigLIP 2 @ 896px (Gemma 3 Vision 27B)
    pub fn siglip2_896() -> Self {
        let image_size: usize = 896;
        let patch_size: usize = 16;
        let n = image_size / patch_size;
        let num_patches = n * n;
        Self {
            encoder_type: VisionEncoderType::SigLip,
            image_size,
            patch_size,
            num_patches,
            n_layers: 27,
            n_heads: 16,
            hidden_dim: 1152,
            mlp_dim: 4304,
            proj_dim: 3072,
            use_cls_token: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Patch Embedding
// ---------------------------------------------------------------------------

/// Patch embedding: converts image to flat sequence of patch tokens.
///
/// Equivalent to Conv2d(3, hidden_dim, kernel=patch_size, stride=patch_size)
/// followed by flatten and transpose.
pub struct PatchEmbedding {
    /// Convolution weights: [hidden_dim, 3, patch_size, patch_size]
    pub weight: Tensor,
    /// Bias: [hidden_dim]
    pub bias: Option<Tensor>,
    pub patch_size: usize,
}

impl PatchEmbedding {
    /// Embed image patches.
    ///
    /// # Arguments
    /// * `image` — `[batch, 3, H, W]` (float, 0..1 normalized)
    ///
    /// # Returns
    /// `[batch, num_patches, hidden_dim]`
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let (batch, _c, h, w) = image.dims4()?;
        let ps = self.patch_size;
        let n_h = h / ps;
        let n_w = w / ps;
        let hidden_dim = self.weight.dim(0)?;

        // Manual 2D patch extraction using reshape + contiguous
        // image: [b, 3, h, w] → [b, 3, n_h, ps, n_w, ps]
        let im_r = image.reshape((batch, 3usize, n_h, ps, n_w, ps))?;
        // → [b, n_h, n_w, 3, ps, ps]
        let im_r = im_r.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
        // → [b, n_h*n_w, 3*ps*ps]
        let n_patches = n_h * n_w;
        let patch_flat = im_r.reshape((batch, n_patches, 3 * ps * ps))?;

        // Linear projection: weight [hidden, 3*ps*ps]
        let w_flat = self.weight.reshape((hidden_dim, 3 * ps * ps))?;
        // [b, n_patches, 3*ps*ps] × [3*ps*ps, hidden] → [b, n_patches, hidden]
        let w_t = w_flat.t()?;
        let out = patch_flat.broadcast_matmul(&w_t)?;

        if let Some(ref b) = self.bias {
            out.broadcast_add(b)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Vision Transformer Block Weights
// ---------------------------------------------------------------------------

/// Weights for one ViT transformer layer.
pub struct VitLayerWeights {
    // Self-attention
    pub attn_norm_w: Tensor,
    pub attn_norm_b: Option<Tensor>,
    pub q: QMatMul,
    pub k: QMatMul,
    pub v: QMatMul,
    pub attn_out: QMatMul,
    // MLP (GELU)
    pub mlp_norm_w: Tensor,
    pub mlp_norm_b: Option<Tensor>,
    pub mlp_fc1: QMatMul,
    pub mlp_fc2: QMatMul,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn layer_norm_vit(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let diff = (x - &mean)?;
    let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = (&diff / (var + eps)?.sqrt()?)?;
    let scaled = normed.broadcast_mul(w)?;
    if let Some(bias) = b {
        scaled.broadcast_add(bias)
    } else {
        Ok(scaled)
    }
}

fn gelu_approx(x: &Tensor) -> Result<Tensor> {
    // tanh approximation for GELU: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let x3 = (x.sqr()? * x)?;
    let inner = ((x + (x3 * 0.044715f64)?)? * (2.0f64 / std::f64::consts::PI).sqrt())?;
    let tanh_inner = inner.tanh()?;
    x * ((tanh_inner + 1.0f64)? * 0.5f64)?
}

// ---------------------------------------------------------------------------
// ViT Forward Pass
// ---------------------------------------------------------------------------

/// Run self-attention for one ViT layer.
fn vit_self_attention(
    x: &Tensor,
    w: &VitLayerWeights,
    n_heads: usize,
) -> Result<Tensor> {
    let (batch, seq, hidden) = x.dims3()?;
    let head_dim = hidden / n_heads;
    let scale = (head_dim as f64).sqrt().recip();

    let q = w.q.forward(x)?; // [b, seq, hidden]
    let k = w.k.forward(x)?;
    let v = w.v.forward(x)?;

    let reshape_heads = |t: Tensor| -> Result<Tensor> {
        t.reshape((batch, seq, n_heads, head_dim))?
            .permute((0, 2, 1, 3)) // [b, heads, seq, head_dim]
    };

    let q = reshape_heads(q)?;
    let k = reshape_heads(k)?;
    let v = reshape_heads(v)?;

    // Scaled dot-product attention (no causal mask for vision — bidirectional)
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    // Numerically stable softmax (no candle_nn dep)
    let score_max = scores.max_keepdim(D::Minus1)?;
    let score_exp = (scores.broadcast_sub(&score_max)?.exp())?;
    let score_sum = score_exp.sum_keepdim(D::Minus1)?;
    let attn = score_exp.broadcast_div(&score_sum)?;
    let ctx = attn.matmul(&v)?; // [b, heads, seq, head_dim]

    let ctx = ctx.permute((0, 2, 1, 3))?.contiguous()?
        .reshape((batch, seq, hidden))?;
    w.attn_out.forward(&ctx)
}

/// Run one ViT transformer layer.
pub fn vit_layer_forward(x: &Tensor, w: &VitLayerWeights, n_heads: usize) -> Result<Tensor> {
    let dtype = x.dtype();
    let nw = w.attn_norm_w.to_dtype(dtype)?;
    let nb = w.attn_norm_b.as_ref().map(|b| b.to_dtype(dtype).unwrap());

    // Pre-norm attention
    let normed = layer_norm_vit(x, &nw, nb.as_ref(), 1e-6)?;
    let attn_out = vit_self_attention(&normed, w, n_heads)?;
    let x = (x + attn_out)?;

    // Pre-norm MLP
    let mw = w.mlp_norm_w.to_dtype(dtype)?;
    let mb = w.mlp_norm_b.as_ref().map(|b| b.to_dtype(dtype).unwrap());
    let normed2 = layer_norm_vit(&x, &mw, mb.as_ref(), 1e-6)?;
    let fc1 = gelu_approx(&w.mlp_fc1.forward(&normed2)?)?;
    let fc2 = w.mlp_fc2.forward(&fc1)?;
    (x + fc2)
}

// ---------------------------------------------------------------------------
// Complete Vision Encoder
// ---------------------------------------------------------------------------

/// Full vision encoder weights.
pub struct VisionEncoderWeights {
    pub patch_embed: PatchEmbedding,
    /// Positional embedding table: [1, num_patches (+1 for CLS), hidden_dim]
    pub pos_embed: Tensor,
    /// Optional CLS token: [1, 1, hidden_dim]
    pub cls_token: Option<Tensor>,
    pub layers: Vec<VitLayerWeights>,
    /// Final LayerNorm (CLIP uses post-norm; SigLIP might not)
    pub post_norm_w: Option<Tensor>,
    pub post_norm_b: Option<Tensor>,
    /// Projection from vision hidden to language model hidden
    pub proj: Option<QMatMul>,
}

/// Encode one image batch through the vision encoder.
///
/// # Arguments
/// * `image`   — `[batch, 3, H, W]`
/// * `weights` — vision encoder weights
/// * `cfg`     — vision config
///
/// # Returns
/// Vision patch tokens `[batch, num_visual_tokens, proj_dim]`
pub fn vision_encode(
    image: &Tensor,
    weights: &VisionEncoderWeights,
    cfg: &VisionConfig,
) -> Result<Tensor> {
    let dtype = image.dtype();

    // ── 1. Patch embedding ────────────────────────────────────────────────
    let patches = weights.patch_embed.forward(image)?; // [b, n_patches, hidden]

    // ── 2. Prepend CLS token if needed ───────────────────────────────────
    let mut x = if let Some(ref cls) = weights.cls_token {
        let batch = patches.dim(0)?;
        let cls_expanded = cls.to_dtype(dtype)?.expand((batch, 1, cfg.hidden_dim))?;
        Tensor::cat(&[&cls_expanded, &patches], 1)?
    } else {
        patches
    };

    // ── 3. Add positional embeddings ──────────────────────────────────────
    let pos = weights.pos_embed.to_dtype(dtype)?;
    x = x.broadcast_add(&pos)?;

    // ── 4. Transformer blocks ─────────────────────────────────────────────
    for layer in &weights.layers {
        x = vit_layer_forward(&x, layer, cfg.n_heads)?;
    }

    // ── 5. Post-norm ──────────────────────────────────────────────────────
    if let Some(ref nw) = weights.post_norm_w {
        let nw = nw.to_dtype(dtype)?;
        let nb = weights.post_norm_b.as_ref().map(|b| b.to_dtype(dtype).unwrap());
        x = layer_norm_vit(&x, &nw, nb.as_ref(), 1e-6)?;
    }

    // ── 6. Drop CLS token for patch-only models ───────────────────────────
    let patch_tokens = if cfg.use_cls_token {
        // For CLIP: some models use CLS, some use patch average
        // LLaVA uses all patch tokens (skip CLS for grid features)
        x.narrow(1, 1, cfg.num_patches)?
    } else {
        x
    };

    // ── 7. Vision → Language projection ──────────────────────────────────
    if let Some(ref proj) = weights.proj {
        proj.forward(&patch_tokens)
    } else {
        Ok(patch_tokens)
    }
}

/// Pre-process image tensor for ViT inference.
///
/// Normalizes to zero mean, unit variance using ImageNet stats (CLIP)
/// or SigLIP normalization (mean=0.5, std=0.5).
///
/// # Arguments
/// * `image`      — `[batch, 3, H, W]` in range [0, 1]
/// * `use_siglip` — if true, uses SigLIP norm (0.5/0.5); else CLIP ImageNet norm
pub fn normalize_image(image: &Tensor, use_siglip: bool) -> Result<Tensor> {
    let dtype = image.dtype();
    let device = image.device();

    let (mean, std) = if use_siglip {
        (vec![0.5f32, 0.5, 0.5], vec![0.5f32, 0.5, 0.5])
    } else {
        // ImageNet stats (CLIP)
        (
            vec![0.48145466f32, 0.4578275, 0.40821073],
            vec![0.26862954f32, 0.26130258, 0.27577711],
        )
    };

    let mean_t = Tensor::from_vec(mean, (3usize,), device)?
        .to_dtype(dtype)?
        .reshape((1usize, 3, 1, 1))?;
    let std_t = Tensor::from_vec(std, (3usize,), device)?
        .to_dtype(dtype)?
        .reshape((1usize, 3, 1, 1))?;

    (image.broadcast_sub(&mean_t)?.broadcast_div(&std_t))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_clip() {
        let cfg = VisionConfig::clip_vit_l_336();
        assert_eq!(cfg.num_patches, (336/14) * (336/14));
        assert_eq!(cfg.num_patches, 576);
        assert_eq!(cfg.hidden_dim, 1024);
        assert!(cfg.use_cls_token);
    }

    #[test]
    fn test_vision_config_siglip() {
        let cfg = VisionConfig::siglip_so14_224();
        assert_eq!(cfg.num_patches, (224/14) * (224/14));
        assert_eq!(cfg.num_patches, 256);
        assert!(!cfg.use_cls_token);
    }

    #[test]
    fn test_vision_config_siglip2_896() {
        let cfg = VisionConfig::siglip2_896();
        assert_eq!(cfg.num_patches, (896/16) * (896/16));
        assert_eq!(cfg.num_patches, 3136);
    }

    #[test]
    fn test_patch_embed_shape() {
        use candle_core::Device;
        let dev = &Device::Cpu;
        let batch = 1;
        let patch_size = 14;
        let hidden = 64;
        let image_size = 28; // 2×2 patches

        let image = Tensor::zeros((batch, 3usize, image_size, image_size), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((hidden, 3usize, patch_size, patch_size), DType::F32, dev).unwrap();
        let embed = PatchEmbedding { weight, bias: None, patch_size };
        let out = embed.forward(&image).unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]); // 4 patches, 64 hidden
    }

    #[test]
    fn test_normalize_clip() {
        use candle_core::Device;
        let dev = &Device::Cpu;
        let img = Tensor::ones((1usize, 3, 4, 4), DType::F32, dev).unwrap();
        let normed = normalize_image(&img, false).unwrap();
        let vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        // (1 - 0.48145) / 0.26863 ≈ 1.93
        assert!((vals[0] - 1.93).abs() < 0.05);
    }
}
