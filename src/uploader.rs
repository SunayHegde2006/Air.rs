#![cfg(feature = "cuda")]

use crate::manifest::Manifest;
use cudarc::driver::CudaSlice;
use memmap2::Mmap;
use std::sync::Arc;
use tokio::sync::mpsc;

// In cudarc 0.19, the device type is `CudaContext`.
type CudarCudaDevice = cudarc::driver::CudaContext;

#[cfg(unix)]
fn prefetch_page_cache(mmap: &Mmap, offset: usize, len: usize) {
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *mut libc::c_void;
        libc::madvise(ptr, len, libc::MADV_WILLNEED);
    }
}
#[cfg(not(unix))]
fn prefetch_page_cache(_mmap: &Mmap, _offset: usize, _len: usize) {
    // Windows: handled via PrefetchVirtualMemory in weight_streamer.rs
}

pub struct VramBuffer {
    pub chunk_id: usize,
    pub slice: CudaSlice<u8>,
}

pub struct TransferEngine {
    receiver: mpsc::Receiver<VramBuffer>,
}

impl TransferEngine {
    pub fn start(device: Arc<CudarCudaDevice>, mmap: Arc<Mmap>, manifest: Arc<Manifest>) -> Self {
        // Triple buffering - we keep a few buffers in flight to hide PCIe latency
        let (tx, rx) = mpsc::channel(3);

        let device_clone = Arc::clone(&device);
        let mmap_clone = Arc::clone(&mmap);
        
        tokio::spawn(async move {
            for chunk in &manifest.chunks {
                let offset = chunk.dma_start_offset as usize;
                let len = chunk.dma_transfer_size as usize;
                
                // Background async prefetch of the next chunk into OS page cache
                prefetch_page_cache(&mmap_clone, offset, len);

                // Copy host slice to VRAM via cudarc 0.19 clone_htod API and synchronize
                let host_slice = &mmap_clone[offset..offset + len];
                let stream = device_clone.default_stream();
                let vram_buffer = stream.clone_htod(host_slice)
                    .expect("Failed to copy to VRAM buffer");
                stream.synchronize().expect("Failed to sync stream");

                let completed_buffer = VramBuffer {
                    chunk_id: chunk.id,
                    slice: vram_buffer,
                };
                
                if tx.send(completed_buffer).await.is_err() {
                    break;
                }
            }
        });

        Self {
            receiver: rx,
        }
    }

    /// Block and wait for the next chunk to be fully written to VRAM.
    pub async fn get_next_buffer(&mut self) -> Option<VramBuffer> {
        self.receiver.recv().await
    }
}
