//! # Distributed Inference Core
//!
//! Provides traits and logic for multi-node Tensor Parallel (TP) and
//! Pipeline Parallel (PP) execution. Synchronized via the STRIX timeline semaphores.

use crate::strix::hal::HalError;
use async_trait::async_trait;

/// Abstract communication backend for distributed orchestration.
#[async_trait]
pub trait Communicator: Send + Sync + std::fmt::Debug {
    /// Rank of the current node in the world.
    fn rank(&self) -> usize;
    /// Total number of nodes in the world.
    fn world_size(&self) -> usize;
    
    /// Blocking all-reduce operation (sum).
    async fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), HalError>;
    
    /// Synchronize all nodes.
    async fn barrier(&self) -> Result<(), HalError>;
    
    /// Send a tensor shard to a specific rank.
    async fn send(&self, to_rank: usize, data: &[u8]) -> Result<(), HalError>;
    
    /// Receive a tensor shard from a specific rank.
    async fn recv(&self, from_rank: usize, data: &mut [u8]) -> Result<(), HalError>;
}

/// Software-agnostic TCP implementation of the [`Communicator`] trait.
#[derive(Debug)]
pub struct TcpCommunicator {
    rank: usize,
    world_size: usize,
    /// Mutex per stream to allow concurrent send/recv across different ranks
    /// while maintaining thread-safety for each individual socket.
    streams: Vec<Option<tokio::sync::Mutex<tokio::net::TcpStream>>>,
}

impl TcpCommunicator {
    /// Initialize a distributed cluster and perform handshake.
    ///
    /// This is a blocking (async) call that waits for all nodes to connect.
    pub async fn new(rank: usize, addresses: &[String]) -> Result<Self, HalError> {
        let world_size = addresses.len();
        let mut streams = Vec::with_capacity(world_size);
        for _ in 0..world_size {
            streams.push(None);
        }

        // 1. Start listener (Server)
        let listener = tokio::net::TcpListener::bind(&addresses[rank]).await
            .map_err(|e| HalError::IoError(e))?;

        // 2. Connect to higher ranks, accept from lower ranks
        for i in 0..world_size {
            if i == rank {
                continue;
            }
            if i < rank {
                // Accept connection from lower rank
                let (stream, _) = listener.accept().await
                    .map_err(|e| HalError::IoError(e))?;
                streams[i] = Some(tokio::sync::Mutex::new(stream));
            } else {
                // Connect to higher rank
                loop {
                    match tokio::net::TcpStream::connect(&addresses[i]).await {
                        Ok(stream) => {
                            streams[i] = Some(tokio::sync::Mutex::new(stream));
                            break;
                        }
                        Err(_) => tokio::time::sleep(tokio::time::Duration::from_millis(500)).await,
                    }
                }
            }
        }

        Ok(Self { rank, world_size, streams })
    }
}

#[async_trait]
impl Communicator for TcpCommunicator {
    fn rank(&self) -> usize { self.rank }
    fn world_size(&self) -> usize { self.world_size }

    async fn all_reduce_sum(&self, data: &mut [f32]) -> Result<(), HalError> {
        if self.world_size == 1 { return Ok(()); }
        
        let chunk_size = (data.len() + self.world_size - 1) / self.world_size;
        
        // Ring Reduce-Scatter
        for i in 0..self.world_size - 1 {
            let send_chunk_id = (self.rank + self.world_size - i) % self.world_size;
            let recv_chunk_id = (self.rank + self.world_size - i - 1) % self.world_size;
            
            let send_start = send_chunk_id * chunk_size;
            let send_end = std::cmp::min(send_start + chunk_size, data.len());
            let recv_start = recv_chunk_id * chunk_size;
            let recv_end = std::cmp::min(recv_start + chunk_size, data.len());

            let send_to = (self.rank + 1) % self.world_size;
            let recv_from = (self.rank + self.world_size - 1) % self.world_size;

            if send_start < data.len() {
                let send_slice = &data[send_start..send_end];
                let send_bytes = unsafe { std::slice::from_raw_parts(send_slice.as_ptr() as *const u8, send_slice.len() * 4) };
                
                let mut recv_buf = vec![0u8; (recv_end - recv_start) * 4];
                
                // Concurrent send and receive to prevent deadlock in the ring
                let (s_res, r_res) = tokio::join!(
                    self.send(send_to, send_bytes),
                    self.recv(recv_from, &mut recv_buf)
                );
                s_res?; r_res?;

                let recv_vals = unsafe { std::slice::from_raw_parts(recv_buf.as_ptr() as *const f32, recv_end - recv_start) };
                for (idx, &val) in recv_vals.iter().enumerate() {
                    data[recv_start + idx] += val;
                }
            }
        }

        // Ring All-Gather
        for i in 0..self.world_size - 1 {
            let send_chunk_id = (self.rank + self.world_size - i + 1) % self.world_size;
            let recv_chunk_id = (self.rank + self.world_size - i) % self.world_size;
            
            let send_start = send_chunk_id * chunk_size;
            let send_end = std::cmp::min(send_start + chunk_size, data.len());
            let recv_start = recv_chunk_id * chunk_size;
            let recv_end = std::cmp::min(recv_start + chunk_size, data.len());

            let send_to = (self.rank + 1) % self.world_size;
            let recv_from = (self.rank + self.world_size - 1) % self.world_size;

            if send_start < data.len() {
                let send_slice = &data[send_start..send_end];
                let send_bytes = unsafe { std::slice::from_raw_parts(send_slice.as_ptr() as *const u8, send_slice.len() * 4) };
                let mut recv_buf = vec![0u8; (recv_end - recv_start) * 4];

                let (s_res, r_res) = tokio::join!(
                    self.send(send_to, send_bytes),
                    self.recv(recv_from, &mut recv_buf)
                );
                s_res?; r_res?;

                let recv_vals = unsafe { std::slice::from_raw_parts(recv_buf.as_ptr() as *const f32, recv_end - recv_start) };
                for (idx, &val) in recv_vals.iter().enumerate() {
                    data[recv_start + idx] = val;
                }
            }
        }

        Ok(())
    }

    async fn barrier(&self) -> Result<(), HalError> {
        if self.world_size == 1 { return Ok(()); }
        let mut msg = [0u8; 1];
        if self.rank == 0 {
            for i in 1..self.world_size {
                self.recv(i, &mut msg).await?;
            }
            for i in 1..self.world_size {
                self.send(i, &msg).await?;
            }
        } else {
            self.send(0, &msg).await?;
            self.recv(0, &mut msg).await?;
        }
        Ok(())
    }

    async fn send(&self, to_rank: usize, data: &[u8]) -> Result<(), HalError> {
        use tokio::io::AsyncWriteExt;
        if let Some(mutex) = &self.streams[to_rank] {
            let mut stream = mutex.lock().await;
            
            // 1. Send size header
            let len = data.len() as u64;
            stream.write_all(&len.to_le_bytes()).await
                .map_err(HalError::IoError)?;
            
            // 2. Send payload
            stream.write_all(data).await
                .map_err(HalError::IoError)?;
            
            stream.flush().await
                .map_err(HalError::IoError)?;
        }
        Ok(())
    }

    async fn recv(&self, from_rank: usize, data: &mut [u8]) -> Result<(), HalError> {
        use tokio::io::AsyncReadExt;
        if let Some(mutex) = &self.streams[from_rank] {
            let mut stream = mutex.lock().await;
            
            // 1. Read size header
            let mut len_bytes = [0u8; 8];
            stream.read_exact(&mut len_bytes).await
                .map_err(HalError::IoError)?;
            let len = u64::from_le_bytes(len_bytes) as usize;
            
            if len != data.len() {
                return Err(HalError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other, 
                    format!("Distributed recv size mismatch: expected {}, got {}", data.len(), len)
                )));
            }

            // 2. Read payload
            stream.read_exact(data).await
                .map_err(HalError::IoError)?;
        }
        Ok(())
    }
}

/// Shard strategy for distributing model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Replicate full weights on all nodes.
    Replicated,
    /// Shard weights along the hidden dimension (Tensor Parallel).
    TensorParallel,
    /// Shard weights along the layer dimension (Pipeline Parallel).
    PipelineParallel,
}

/// A node in the distributed cluster.
pub struct DistributedNode {
    pub id: String,
    pub address: String,
    pub primary_gpu: usize,
}
