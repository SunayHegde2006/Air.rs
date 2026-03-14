use anyhow::Result;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use std::sync::Arc;

// In cudarc 0.13, the device type is `CudaDevice` (not CudaStream/CudaContext).
type CudarCudaDevice = candle_core::cuda_backend::cudarc::driver::CudaDevice;

pub struct LayerKvCache {
    pub layer_id: usize,
    pub k_cache_ram: Vec<u8>,
    pub v_cache_ram: Vec<u8>,
}

pub struct KvCacheManager {
    device: Arc<CudarCudaDevice>,
    layers: Vec<LayerKvCache>,
}

impl KvCacheManager {
    pub fn new(device: Arc<CudarCudaDevice>, num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for id in 0..num_layers {
            layers.push(LayerKvCache {
                layer_id: id,
                k_cache_ram: Vec::new(),
                v_cache_ram: Vec::new(),
            });
        }
        
        Self { device, layers }
    }

    /// Loads a specific layer's KV-cache into VRAM.
    pub fn load_to_vram(&self, layer_id: usize) -> Result<(CudaSlice<u8>, CudaSlice<u8>)> {
        let layer = &self.layers[layer_id];
        
        let k_vram = if layer.k_cache_ram.is_empty() {
            self.device.alloc_zeros::<u8>(1)
                .map_err(|e| anyhow::anyhow!("k alloc failed: {e}"))?
        } else {
            self.device.htod_sync_copy(&layer.k_cache_ram)
                .map_err(|e| anyhow::anyhow!("k htod failed: {e}"))?
        };
        
        let v_vram = if layer.v_cache_ram.is_empty() {
            self.device.alloc_zeros::<u8>(1)
                .map_err(|e| anyhow::anyhow!("v alloc failed: {e}"))?
        } else {
            self.device.htod_sync_copy(&layer.v_cache_ram)
                .map_err(|e| anyhow::anyhow!("v htod failed: {e}"))?
        };
        
        Ok((k_vram, v_vram))
    }

    /// Saves a newly computed/updated KV-cache back to System RAM.
    pub fn save_from_vram(&mut self, layer_id: usize, k_vram: &CudaSlice<u8>, v_vram: &CudaSlice<u8>) -> Result<()> {
        let layer = &mut self.layers[layer_id];
        
        // Download from VRAM to RAM
        layer.k_cache_ram = self.device.dtoh_sync_copy(k_vram)
            .map_err(|e| anyhow::anyhow!("k dtoh failed: {e}"))?;
        layer.v_cache_ram = self.device.dtoh_sync_copy(v_vram)
            .map_err(|e| anyhow::anyhow!("v dtoh failed: {e}"))?;
        
        Ok(())
    }
}
