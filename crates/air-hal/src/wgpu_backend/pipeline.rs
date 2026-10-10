use air_core::error::{AirError, Result};
use std::sync::Arc;

pub struct WgpuPipeline {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl WgpuPipeline {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| AirError::Hal("No suitable WebGPU adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| AirError::Hal(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    pub fn dispatch_wgsl_shader(&self, shader_code: &str, entry_point: &str, workgroups: (u32, u32, u32)) -> Result<()> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Air.rs WGSL Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Air.rs WGSL Compute Pipeline"),
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Air.rs WGSL Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Air.rs Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compute_pipeline);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
