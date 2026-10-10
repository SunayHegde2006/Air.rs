use crate::vision::VisionPatchExtractor;
use air_core::error::Result;

pub struct VideoProcessor {
    vision_extractor: VisionPatchExtractor,
    max_frames: usize,
}

impl VideoProcessor {
    pub fn new(patch_size: usize, image_size: usize, max_frames: usize) -> Self {
        Self {
            vision_extractor: VisionPatchExtractor::new(patch_size, image_size),
            max_frames,
        }
    }

    pub fn process_keyframes(
        &self,
        frames: &[Vec<u8>],
        width: usize,
        height: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let frame_count = frames.len().min(self.max_frames);
        let mut processed_frames = Vec::with_capacity(frame_count);

        for frame in frames.iter().take(frame_count) {
            let patches = self.vision_extractor.extract_patches_rgb(frame, width, height)?;
            processed_frames.push(patches);
        }

        Ok(processed_frames)
    }
}
