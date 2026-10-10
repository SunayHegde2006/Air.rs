use crate::buffer::DeviceBuffer;
use air_core::error::Result;
use air_core::types::DeviceInfo;

pub trait ComputeDevice: Send + Sync {
    fn info(&self) -> DeviceInfo;
    fn allocate(&self, size_bytes: usize) -> Result<DeviceBuffer>;
    fn synchronize(&self) -> Result<()>;
}

pub struct CpuDevice {
    info: DeviceInfo,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            info: DeviceInfo {
                kind: air_core::types::DeviceKind::Cpu,
                name: "Host CPU".to_string(),
                total_vram_bytes: 64 * 1024 * 1024 * 1024,
                free_vram_bytes: 32 * 1024 * 1024 * 1024,
                compute_capability: 0,
                sm_count: 16,
                memory_bus_bits: 256,
                max_bandwidth_gbps: 50.0,
            },
        }
    }
}

impl ComputeDevice for CpuDevice {
    fn info(&self) -> DeviceInfo {
        self.info.clone()
    }

    fn allocate(&self, size_bytes: usize) -> Result<DeviceBuffer> {
        DeviceBuffer::alloc_cpu(size_bytes)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
