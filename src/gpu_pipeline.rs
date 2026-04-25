//! Multi-GPU Pipeline Parallelism.
//!
//! For systems with multiple CUDA GPUs, this module partitions transformer
//! layers across devices and manages inter-device tensor transfers.
//!
//! Strategy:
//! ```text
//! GPU 0: layers  0..N/2   (first half of transformer)
//! GPU 1: layers N/2..N    (second half of transformer)
//! ...
//! ```
//!
//! Hidden states are transferred between GPUs at partition boundaries via
//! `tensor.to_device()`. Each GPU maintains its own KV cache for its layers.

use candle_core::Device;

// ---------------------------------------------------------------------------
// GPU Topology Discovery
// ---------------------------------------------------------------------------

/// Represents an available compute device with its properties.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device index (0, 1, 2, ...)
    pub device_id: usize,
    /// Candle device handle
    pub device: Device,
    /// Human-readable name
    pub name: String,
    /// Available memory in bytes (estimated)
    pub memory_bytes: u64,
}

/// Discover available GPU devices.
pub struct GpuTopology {
    pub devices: Vec<GpuInfo>,
}

impl GpuTopology {
    /// Probe the system for available CUDA devices.
    pub fn discover() -> Self {
        let mut devices = Vec::new();

        // Try to create CUDA devices incrementally
        for id in 0..8 {
            // Reasonable max of 8 GPUs
            match Device::new_cuda(id) {
                Ok(device) => {
                    devices.push(GpuInfo {
                        device_id: id,
                        device,
                        name: format!("CUDA:{}", id),
                        // Memory detection would require cuMemGetInfo via CUDA bindings,
                        // which candle doesn't expose directly. We use a placeholder.
                        memory_bytes: 0,
                    });
                }
                Err(_) => break, // No more GPUs
            }
        }

        if devices.is_empty() {
            // Fall back to CPU
            devices.push(GpuInfo {
                device_id: 0,
                device: Device::Cpu,
                name: "CPU".to_string(),
                memory_bytes: 0,
            });
        }

        Self { devices }
    }

    /// Number of available compute devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Check if we have multiple GPUs (pipeline parallelism is beneficial).
    pub fn is_multi_gpu(&self) -> bool {
        self.devices.len() > 1
            && self.devices.iter().all(|d| !matches!(d.device, Device::Cpu))
    }
}

impl std::fmt::Display for GpuTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPU Topology ({} device(s)):", self.devices.len())?;
        for gpu in &self.devices {
            writeln!(f, "  {} [{}]", gpu.name, gpu.device_id)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline Partitioning
// ---------------------------------------------------------------------------

/// A partition assigns a contiguous range of transformer layers to a device.
#[derive(Debug, Clone)]
pub struct PipelinePartition {
    pub device_id: usize,
    pub device: Device,
    pub start_layer: usize,
    pub end_layer: usize, // exclusive
}

impl PipelinePartition {
    /// Number of layers assigned to this partition.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }

    /// Check if a given layer belongs to this partition.
    pub fn contains_layer(&self, layer_id: usize) -> bool {
        layer_id >= self.start_layer && layer_id < self.end_layer
    }
}

/// Create balanced pipeline partitions across available GPUs.
///
/// Distributes layers as evenly as possible. Extra layers go to the last GPU.
pub fn partition_layers(n_layers: usize, topology: &GpuTopology) -> Vec<PipelinePartition> {
    let n_gpus = topology.device_count();

    if n_gpus <= 1 {
        return vec![PipelinePartition {
            device_id: 0,
            device: topology.devices[0].device.clone(),
            start_layer: 0,
            end_layer: n_layers,
        }];
    }

    let layers_per_gpu = n_layers / n_gpus;
    let mut partitions = Vec::with_capacity(n_gpus);
    let mut start = 0;

    for (i, gpu) in topology.devices.iter().enumerate() {
        let end = if i == n_gpus - 1 {
            n_layers // Last GPU gets any remaining layers
        } else {
            start + layers_per_gpu
        };

        partitions.push(PipelinePartition {
            device_id: gpu.device_id,
            device: gpu.device.clone(),
            start_layer: start,
            end_layer: end,
        });

        start = end;
    }

    partitions
}

/// Find which partition a given layer belongs to.
pub fn layer_device(layer_id: usize, partitions: &[PipelinePartition]) -> &PipelinePartition {
    partitions
        .iter()
        .find(|p| p.contains_layer(layer_id))
        .unwrap_or(&partitions[0]) // Safety fallback
}

/// Check if a layer transition requires a device-to-device transfer.
pub fn needs_transfer(
    current_layer: usize,
    next_layer: usize,
    partitions: &[PipelinePartition],
) -> bool {
    let current = layer_device(current_layer, partitions);
    let next = layer_device(next_layer, partitions);
    current.device_id != next.device_id
}

// ---------------------------------------------------------------------------
// Pipeline Executor
// ---------------------------------------------------------------------------

/// Manages execution across a multi-GPU pipeline.
///
/// The executor keeps track of which device is currently active and handles
/// tensor transfers between devices at partition boundaries.
pub struct PipelineExecutor {
    pub partitions: Vec<PipelinePartition>,
    pub topology: GpuTopology,
    /// Total cross-device transfers performed
    pub transfer_count: usize,
}

impl PipelineExecutor {
    /// Create a new pipeline executor for the given model.
    pub fn new(n_layers: usize) -> Self {
        let topology = GpuTopology::discover();
        let partitions = partition_layers(n_layers, &topology);

        println!("{}", topology);
        println!("Pipeline partitions:");
        for p in &partitions {
            println!(
                "  GPU {} → layers {}..{} ({} layers)",
                p.device_id,
                p.start_layer,
                p.end_layer,
                p.num_layers()
            );
        }

        Self {
            partitions,
            topology,
            transfer_count: 0,
        }
    }

    /// Get the device for a specific layer.
    pub fn device_for_layer(&self, layer_id: usize) -> &Device {
        &layer_device(layer_id, &self.partitions).device
    }

    /// Transfer a tensor to the correct device for the target layer.
    ///
    /// Returns the tensor on the new device if transfer was needed,
    /// or the original tensor if already on the correct device.
    pub fn transfer_if_needed(
        &mut self,
        tensor: candle_core::Tensor,
        target_layer: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let target_device = layer_device(target_layer, &self.partitions).device.clone();

        if tensor.device().same_device(&target_device) {
            Ok(tensor)
        } else {
            self.transfer_count += 1;
            tensor.to_device(&target_device)
        }
    }

    /// Get the first (primary) device for embedding and output layers.
    pub fn primary_device(&self) -> &Device {
        &self.partitions[0].device
    }

    /// Check if pipeline parallelism is active.
    pub fn is_parallel(&self) -> bool {
        self.partitions.len() > 1
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_device_partition() {
        // On a machine with no CUDA, we get a single CPU partition
        let topo = GpuTopology {
            devices: vec![GpuInfo {
                device_id: 0,
                device: Device::Cpu,
                name: "CPU".to_string(),
                memory_bytes: 0,
            }],
        };

        let partitions = partition_layers(32, &topo);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].start_layer, 0);
        assert_eq!(partitions[0].end_layer, 32);
        assert_eq!(partitions[0].num_layers(), 32);
    }

    #[test]
    fn test_multi_device_partition() {
        let topo = GpuTopology {
            devices: vec![
                GpuInfo {
                    device_id: 0,
                    device: Device::Cpu,
                    name: "GPU0".to_string(),
                    memory_bytes: 0,
                },
                GpuInfo {
                    device_id: 1,
                    device: Device::Cpu, // Using CPU for test
                    name: "GPU1".to_string(),
                    memory_bytes: 0,
                },
            ],
        };

        let partitions = partition_layers(32, &topo);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].start_layer, 0);
        assert_eq!(partitions[0].end_layer, 16);
        assert_eq!(partitions[1].start_layer, 16);
        assert_eq!(partitions[1].end_layer, 32);
    }

    #[test]
    fn test_odd_layer_distribution() {
        let topo = GpuTopology {
            devices: vec![
                GpuInfo {
                    device_id: 0,
                    device: Device::Cpu,
                    name: "GPU0".to_string(),
                    memory_bytes: 0,
                },
                GpuInfo {
                    device_id: 1,
                    device: Device::Cpu,
                    name: "GPU1".to_string(),
                    memory_bytes: 0,
                },
            ],
        };

        // 33 layers: GPU0 gets 16, GPU1 gets 17 (the remainder)
        let partitions = partition_layers(33, &topo);
        assert_eq!(partitions[0].num_layers(), 16);
        assert_eq!(partitions[1].num_layers(), 17);
    }

    #[test]
    fn test_layer_device_lookup() {
        let topo = GpuTopology {
            devices: vec![
                GpuInfo {
                    device_id: 0,
                    device: Device::Cpu,
                    name: "GPU0".to_string(),
                    memory_bytes: 0,
                },
                GpuInfo {
                    device_id: 1,
                    device: Device::Cpu,
                    name: "GPU1".to_string(),
                    memory_bytes: 0,
                },
            ],
        };

        let partitions = partition_layers(32, &topo);
        assert_eq!(layer_device(0, &partitions).device_id, 0);
        assert_eq!(layer_device(15, &partitions).device_id, 0);
        assert_eq!(layer_device(16, &partitions).device_id, 1);
        assert_eq!(layer_device(31, &partitions).device_id, 1);
    }

    #[test]
    fn test_needs_transfer() {
        let topo = GpuTopology {
            devices: vec![
                GpuInfo {
                    device_id: 0,
                    device: Device::Cpu,
                    name: "GPU0".to_string(),
                    memory_bytes: 0,
                },
                GpuInfo {
                    device_id: 1,
                    device: Device::Cpu,
                    name: "GPU1".to_string(),
                    memory_bytes: 0,
                },
            ],
        };

        let partitions = partition_layers(32, &topo);
        assert!(!needs_transfer(0, 1, &partitions));
        assert!(!needs_transfer(14, 15, &partitions));
        assert!(needs_transfer(15, 16, &partitions));  // Crosses boundary
        assert!(!needs_transfer(16, 17, &partitions));
    }

    #[test]
    fn test_topology_display() {
        let topo = GpuTopology {
            devices: vec![GpuInfo {
                device_id: 0,
                device: Device::Cpu,
                name: "CPU".to_string(),
                memory_bytes: 0,
            }],
        };
        let display = format!("{}", topo);
        assert!(display.contains("CPU"));
        assert!(display.contains("1 device"));
    }
}
