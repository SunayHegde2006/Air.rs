use air_core::error::{AirError, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

pub struct NvmeSpillManager {
    spill_file_path: PathBuf,
    file: File,
}

impl NvmeSpillManager {
    pub fn new(spill_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&spill_dir)?;
        let spill_file_path = spill_dir.join("kv_cache_spill.bin");
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&spill_file_path)?;

        Ok(Self {
            spill_file_path,
            file,
        })
    }

    pub fn write_block_dma(&self, block_offset_bytes: u64, data: &[u8]) -> Result<()> {
        self.file
            .write_all_at(data, block_offset_bytes)
            .map_err(|e| AirError::Ice(format!("Spill write failed: {e}")))?;
        Ok(())
    }

    pub fn read_block_dma(&self, block_offset_bytes: u64, buffer: &mut [u8]) -> Result<()> {
        self.file
            .read_exact_at(buffer, block_offset_bytes)
            .map_err(|e| AirError::Ice(format!("Spill read failed: {e}")))?;
        Ok(())
    }
}
