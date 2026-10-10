use air_core::error::{AirError, Result};

pub struct VisionPatchExtractor {
    patch_size: usize,
    image_size: usize,
}

impl VisionPatchExtractor {
    pub fn new(patch_size: usize, image_size: usize) -> Self {
        Self {
            patch_size,
            image_size,
        }
    }

    pub fn extract_patches_rgb(&self, rgb_data: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        if width != self.image_size || height != self.image_size {
            return Err(AirError::Mie(format!(
                "Image size mismatch: expected {}x{}, got {}x{}",
                self.image_size, self.image_size, width, height
            )));
        }

        let patches_per_dim = self.image_size / self.patch_size;
        let total_patches = patches_per_dim * patches_per_dim;
        let patch_pixels = self.patch_size * self.patch_size * 3;
        let mut patch_tensor = vec![0.0_f32; total_patches * patch_pixels];

        for p_y in 0..patches_per_dim {
            for p_x in 0..patches_per_dim {
                let patch_idx = p_y * patches_per_dim + p_x;
                let patch_out = &mut patch_tensor[patch_idx * patch_pixels..(patch_idx + 1) * patch_pixels];

                let mut pixel_idx = 0;
                for y in 0..self.patch_size {
                    for x in 0..self.patch_size {
                        let img_x = p_x * self.patch_size + x;
                        let img_y = p_y * self.patch_size + y;
                        let src_idx = (img_y * width + img_x) * 3;

                        patch_out[pixel_idx] = rgb_data[src_idx] as f32 / 255.0;
                        patch_out[pixel_idx + 1] = rgb_data[src_idx + 1] as f32 / 255.0;
                        patch_out[pixel_idx + 2] = rgb_data[src_idx + 2] as f32 / 255.0;
                        pixel_idx += 3;
                    }
                }
            }
        }

        Ok(patch_tensor)
    }
}
