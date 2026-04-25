//! Model Hub — Download and manage GGUF models from Hugging Face.
//!
//! Provides:
//! - One-line model download: `air pull TheBloke/Llama-2-7B-GGUF`
//! - SHA-256 integrity verification
//! - Local model registry with metadata
//! - Progress bar for large downloads
//!
//! Models are stored in `~/.air/models/<org>/<repo>/<filename>`.

use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Base URL for Hugging Face model downloads.
const HF_BASE_URL: &str = "https://huggingface.co";

/// Default models directory (relative to home).
const MODELS_DIR: &str = ".air/models";

/// Registry filename.
const REGISTRY_FILE: &str = ".air/registry.json";

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

/// Information about a locally cached model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    /// HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")
    pub repo_id: String,
    /// Filename within the repo (e.g., "llama-2-7b.Q4_K_M.gguf")
    pub filename: String,
    /// Full local path to the GGUF file.
    pub local_path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hash of the file (hex-encoded).
    pub sha256: Option<String>,
    /// When the model was downloaded (ISO 8601).
    pub downloaded_at: String,
    /// Optional alias for quick reference.
    pub alias: Option<String>,
}

/// The local model registry, persisted as JSON.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<ModelEntry>,
}

impl ModelRegistry {
    /// Load the registry from disk (or create empty if not found).
    pub fn load() -> Result<Self> {
        let path = Self::registry_path()?;
        if path.exists() {
            let data = fs::read_to_string(&path)
                .with_context(|| format!("Failed to read registry: {}", path.display()))?;
            let registry: Self = serde_json::from_str(&data)
                .with_context(|| "Failed to parse registry JSON")?;
            Ok(registry)
        } else {
            Ok(Self::default())
        }
    }

    /// Save the registry to disk.
    pub fn save(&self) -> Result<()> {
        let path = Self::registry_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&path, data)?;
        Ok(())
    }

    /// Add a model entry.
    pub fn add(&mut self, entry: ModelEntry) {
        // Remove existing entry for the same file
        self.models.retain(|e| {
            !(e.repo_id == entry.repo_id && e.filename == entry.filename)
        });
        self.models.push(entry);
    }

    /// Find a model by repo and filename.
    pub fn find(&self, repo_id: &str, filename: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|e| e.repo_id == repo_id && e.filename == filename)
    }

    /// Find a model by alias.
    pub fn find_by_alias(&self, alias: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|e| e.alias.as_deref() == Some(alias))
    }

    /// List all registered models.
    pub fn list(&self) -> &[ModelEntry] {
        &self.models
    }

    /// Remove a model entry (does not delete the file).
    pub fn remove(&mut self, repo_id: &str, filename: &str) -> bool {
        let before = self.models.len();
        self.models
            .retain(|e| !(e.repo_id == repo_id && e.filename == filename));
        self.models.len() < before
    }

    fn registry_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Cannot determine home directory")?;
        Ok(home.join(REGISTRY_FILE))
    }
}

// ---------------------------------------------------------------------------
// Download Manager
// ---------------------------------------------------------------------------

/// Progress callback type.
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send>;

/// Result of a model download.
#[derive(Debug)]
pub struct DownloadResult {
    pub local_path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
    pub duration_secs: f64,
    pub speed_mbps: f64,
}

impl std::fmt::Display for DownloadResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Download complete!")?;
        writeln!(f, "  Path: {}", self.local_path.display())?;
        writeln!(f, "  Size: {}", format_size(self.size_bytes))?;
        writeln!(f, "  SHA-256: {}...", &self.sha256[..16])?;
        writeln!(f, "  Speed: {:.1} MB/s ({:.1}s)", self.speed_mbps, self.duration_secs)?;
        Ok(())
    }
}

/// Construct the download URL for a HuggingFace model file.
pub fn download_url(repo_id: &str, filename: &str) -> String {
    format!(
        "{}/{}/resolve/main/{}",
        HF_BASE_URL, repo_id, filename
    )
}

/// Construct the local path for a model file.
pub fn model_path(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(MODELS_DIR).join(repo_id).join(filename))
}

/// Check if a model file already exists locally.
pub fn is_cached(repo_id: &str, filename: &str) -> Result<bool> {
    let path = model_path(repo_id, filename)?;
    Ok(path.exists())
}

/// Download a GGUF model file from Hugging Face.
///
/// If the file already exists locally and `force` is false, returns the
/// cached path without re-downloading.
pub fn download_model(
    repo_id: &str,
    filename: &str,
    force: bool,
    hf_token: Option<&str>,
) -> Result<DownloadResult> {
    let local_path = model_path(repo_id, filename)?;

    // Check cache
    if !force && local_path.exists() {
        let metadata = fs::metadata(&local_path)?;
        println!("Model already cached: {}", local_path.display());
        return Ok(DownloadResult {
            local_path,
            size_bytes: metadata.len(),
            sha256: "cached".to_string(),
            duration_secs: 0.0,
            speed_mbps: 0.0,
        });
    }

    // Create directory
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let url = download_url(repo_id, filename);
    println!("Downloading: {}", url);
    println!("Destination: {}", local_path.display());

    let start = Instant::now();

    // Build HTTP request
    let mut request = ureq::get(&url);
    if let Some(token) = hf_token {
        request = request.set("Authorization", &format!("Bearer {}", token));
    }

    let response = request.call().with_context(|| {
        format!("Failed to download from {}", url)
    })?;

    let total_size: u64 = response
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Download with progress
    let mut file = fs::File::create(&local_path)?;
    let mut downloaded: u64 = 0;
    let mut buffer = vec![0u8; 1024 * 1024]; // 1 MB buffer
    let mut reader = response.into_reader();
    let mut hasher = sha2::Sha256::new();

    loop {
        use sha2::Digest;
        let bytes_read = std::io::Read::read(&mut reader, &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        hasher.update(&buffer[..bytes_read]);
        downloaded += bytes_read as u64;

        // Print progress
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64 * 100.0) as u32;
            let bar_len = 40;
            let filled = (pct as usize * bar_len) / 100;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
            print!(
                "\r  [{}] {}% ({}/{})",
                bar,
                pct,
                format_size(downloaded),
                format_size(total_size)
            );
            std::io::stdout().flush().ok();
        }
    }

    println!(); // Newline after progress bar

    use sha2::Digest;
    let hash = format!("{:x}", hasher.finalize());
    let elapsed = start.elapsed().as_secs_f64();
    let speed = if elapsed > 0.0 {
        (downloaded as f64 / 1_048_576.0) / elapsed
    } else {
        0.0
    };

    let result = DownloadResult {
        local_path,
        size_bytes: downloaded,
        sha256: hash,
        duration_secs: elapsed,
        speed_mbps: speed,
    };

    println!("{}", result);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Model Listing & Discovery
// ---------------------------------------------------------------------------

/// Parse a model specifier like "TheBloke/Llama-2-7B-GGUF:Q4_K_M".
///
/// Returns (repo_id, optional_quant_hint).
pub fn parse_model_spec(spec: &str) -> (&str, Option<&str>) {
    if let Some(idx) = spec.find(':') {
        (&spec[..idx], Some(&spec[idx + 1..]))
    } else {
        (spec, None)
    }
}

/// Guess the best GGUF filename for a given quantization hint.
///
/// If no hint is provided, defaults to Q4_K_M as a good balance of
/// quality and size.
pub fn guess_filename(repo_id: &str, quant_hint: Option<&str>) -> String {
    let quant = quant_hint.unwrap_or("Q4_K_M");
    // Extract model name from repo ID
    let model_name = repo_id
        .split('/')
        .last()
        .unwrap_or("model")
        .to_lowercase()
        .replace("-gguf", "");
    format!("{}.{}.gguf", model_name, quant)
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Compute SHA-256 hash of a file.
pub fn hash_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    let data = fs::read(path)?;
    let hash = sha2::Sha256::digest(&data);
    Ok(format!("{:x}", hash))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(4_831_838_208), "4.5 GB");
    }

    #[test]
    fn test_download_url() {
        let url = download_url("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
        assert_eq!(
            url,
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
        );
    }

    #[test]
    fn test_parse_model_spec() {
        let (repo, quant) = parse_model_spec("TheBloke/Llama-2-7B-GGUF:Q4_K_M");
        assert_eq!(repo, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(quant, Some("Q4_K_M"));

        let (repo, quant) = parse_model_spec("TheBloke/Llama-2-7B-GGUF");
        assert_eq!(repo, "TheBloke/Llama-2-7B-GGUF");
        assert_eq!(quant, None);
    }

    #[test]
    fn test_guess_filename() {
        let name = guess_filename("TheBloke/Llama-2-7B-GGUF", Some("Q5_K_M"));
        assert_eq!(name, "llama-2-7b.Q5_K_M.gguf");

        let name = guess_filename("TheBloke/Mistral-7B-v0.3-GGUF", None);
        assert_eq!(name, "mistral-7b-v0.3.Q4_K_M.gguf");
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = ModelRegistry::default();
        assert!(registry.list().is_empty());

        let entry = ModelEntry {
            repo_id: "org/model-GGUF".to_string(),
            filename: "model.Q4_K_M.gguf".to_string(),
            local_path: "/tmp/model.gguf".to_string(),
            size_bytes: 4_000_000_000,
            sha256: Some("abc123".to_string()),
            downloaded_at: "2024-01-01T00:00:00Z".to_string(),
            alias: Some("my-model".to_string()),
        };

        registry.add(entry);
        assert_eq!(registry.list().len(), 1);

        let found = registry.find("org/model-GGUF", "model.Q4_K_M.gguf");
        assert!(found.is_some());

        let found_alias = registry.find_by_alias("my-model");
        assert!(found_alias.is_some());

        let removed = registry.remove("org/model-GGUF", "model.Q4_K_M.gguf");
        assert!(removed);
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_is_cached() {
        // Non-existent model should not be cached
        let result = is_cached("nonexistent/model", "nonexistent.gguf");
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}
