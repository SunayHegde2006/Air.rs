use crate::error::{AirError, Result};
use std::path::{Path, PathBuf};

pub struct ModelHub {
    cache_dir: PathBuf,
}

impl ModelHub {
    pub fn new() -> Result<Self> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| AirError::Hub("Could not find user home directory".to_string()))?;
        let cache_dir = PathBuf::from(home).join(".air-rs").join("hub");
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    pub fn resolve_model(&self, model_spec: &str) -> Result<PathBuf> {
        let path = Path::new(model_spec);
        if path.exists() {
            return Ok(path.to_path_buf());
        }

        // Cache filename lookup
        let filename = model_spec.replace('/', "--");
        let cached_file = self.cache_dir.join(&filename);
        if cached_file.exists() {
            return Ok(cached_file);
        }

        if model_spec.starts_with("http://") || model_spec.starts_with("https://") {
            let file_name = model_spec
                .split('/')
                .next_back()
                .unwrap_or("downloaded_model.gguf");
            let target_path = self.cache_dir.join(file_name);
            self.download_file(model_spec, &target_path)?;
            return Ok(target_path);
        }

        Err(AirError::Hub(format!(
            "Model '{model_spec}' not found locally or in cache"
        )))
    }

    fn download_file(&self, url: &str, target_path: &Path) -> Result<()> {
        let response = reqwest::blocking::get(url)
            .map_err(|e| AirError::Hub(format!("Download request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(AirError::Hub(format!(
                "HTTP error downloading model: {}",
                response.status()
            )));
        }

        let mut file = std::fs::File::create(target_path)?;
        let mut content = std::io::Cursor::new(
            response
                .bytes()
                .map_err(|e| AirError::Hub(format!("Failed to read bytes: {e}")))?,
        );
        std::io::copy(&mut content, &mut file)?;
        Ok(())
    }
}
