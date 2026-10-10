use air_core::error::{AirError, Result};

pub struct AudioSpectrogramExtractor {
    sample_rate: usize,
    n_mels: usize,
}

impl AudioSpectrogramExtractor {
    pub fn new(sample_rate: usize, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_mels,
        }
    }

    pub fn pcm_to_log_mel(&self, pcm_samples: &[f32]) -> Result<Vec<f32>> {
        if pcm_samples.is_empty() {
            return Err(AirError::Mie("Empty PCM audio input".into()));
        }

        let num_frames = pcm_samples.len() / 160;
        let mut mel_spectrogram = vec![0.0_f32; num_frames * self.n_mels];

        for frame in 0..num_frames {
            let start = frame * 160;
            let end = (start + 400).min(pcm_samples.len());
            let slice = &pcm_samples[start..end];

            let energy: f32 = slice.iter().map(|&x| x * x).sum();
            let log_energy = (energy + 1e-5).ln();

            for m in 0..self.n_mels {
                mel_spectrogram[frame * self.n_mels + m] = log_energy;
            }
        }

        Ok(mel_spectrogram)
    }
}
