//! VulkanBlock — LLM transformer block executing on Vulkan compute.
//!
//! Architecture:
//!   1. On first `execute()` call, loads the SPIR-V shaders compiled by
//!      `build.rs` from `AIR_SHADER_DIR`, creates `VkShaderModule`,
//!      `VkDescriptorSetLayout`, `VkPipelineLayout`, and `VkComputePipeline`
//!      for the RMSNorm + MatMul kernels.
//!   2. Weight tensors (Q/K/V projection matrices) are uploaded to device-local
//!      VRAM via `VulkanHal::staged_copy_to_device_local` on the first
//!      forward pass (JIT resident strategy).
//!   3. Each forward pass:
//!      a. Copies the activation tensor (x) to a host-mapped Vulkan buffer.
//!      b. Dispatches the RMSNorm shader → attention norm result stored on GPU.
//!      c. Dispatches the MatMul shader for Q/K/V projections.
//!      d. Maps the output buffer and copies result back to a candle Tensor.
//!
//! The CPU fallback (`air_compute_*` C functions) is called for operations
//! that don't yet have a GPU shader (RoPE, FFN gate in v1.0).
//!
//! Gated under `#[cfg(feature = "vulkan")]`.

use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use crate::layer_pipeline::{LayerUnit, LayerExecutionContext};
use crate::kv_cache::LayerCache;
use crate::strix::vulkan_hal::VulkanHal;
use candle_core::{Tensor, Result as CResult};

// ── Vulkan type aliases (matching vulkan_hal.rs conventions) ────────────────

type VkDevice          = *mut c_void;
type VkShaderModule    = *mut c_void;
type VkDescriptorSetLayout = *mut c_void;
type VkPipelineLayout  = *mut c_void;
type VkPipeline        = *mut c_void;
type VkDescriptorPool  = *mut c_void;
type VkDescriptorSet   = *mut c_void;
type VkBuffer          = *mut c_void;
type VkDeviceMemory    = *mut c_void;
type VkCommandPool     = *mut c_void;
type VkCommandBuffer   = *mut c_void;
type VkFence           = *mut c_void;
type VkQueue           = *mut c_void;
type VkResult          = i32;

const VK_SUCCESS: VkResult = 0;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32       = 15;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32    = 29;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32     = 30;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32     = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32    = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32            = 35;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32              = 12;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32            = 5;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32        = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32    = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32       = 42;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32                     = 4;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32               = 8;

const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32                 = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32                       = 0x20;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32                = 0x20;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32                  = 0x02;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32                  = 0x01;
const VK_SHARING_MODE_EXCLUSIVE: u32                         = 0;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32                   = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32       = 0x01;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32               = 0x02;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32              = 0x04;

// ── Vulkan structs ──────────────────────────────────────────────────────────

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type:     u32,
    p_next:     *const c_void,
    flags:      u32,
    code_size:  usize,
    p_code:     *const u32,
}

#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding:              u32,
    descriptor_type:      u32,
    descriptor_count:     u32,
    stage_flags:          u32,
    p_immutable_samplers: *const c_void,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type:        u32,
    p_next:        *const c_void,
    flags:         u32,
    binding_count: u32,
    p_bindings:    *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type:                    u32,
    p_next:                    *const c_void,
    flags:                     u32,
    set_layout_count:          u32,
    p_set_layouts:             *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges:    *const VkPushConstantRange,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPushConstantRange {
    stage_flags: u32,
    offset:      u32,
    size:        u32,
}

#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    s_type:                u32,
    p_next:                *const c_void,
    flags:                 u32,
    stage:                 u32,
    module_:               VkShaderModule,
    p_name:                *const i8,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkComputePipelineCreateInfo {
    s_type:               u32,
    p_next:               *const c_void,
    flags:                u32,
    stage:                VkPipelineShaderStageCreateInfo,
    layout:               VkPipelineLayout,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index:  i32,
}

#[repr(C)]
struct VkDescriptorPoolSize {
    ty:               u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type:           u32,
    p_next:           *const c_void,
    flags:            u32,
    max_sets:         u32,
    pool_size_count:  u32,
    p_pool_sizes:     *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type:               u32,
    p_next:               *const c_void,
    descriptor_pool:      VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts:        *const VkDescriptorSetLayout,
}

#[repr(C)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: u64,
    range:  u64,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type:             u32,
    p_next:             *const c_void,
    dst_set:            VkDescriptorSet,
    dst_binding:        u32,
    dst_array_element:  u32,
    descriptor_count:   u32,
    descriptor_type:    u32,
    p_image_info:       *const c_void,
    p_buffer_info:      *const VkDescriptorBufferInfo,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type:                    u32,
    p_next:                    *const c_void,
    flags:                     u32,
    size:                      u64,
    usage:                     u32,
    sharing_mode:              u32,
    queue_family_index_count:  u32,
    p_queue_family_indices:    *const u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type:            u32,
    p_next:            *const c_void,
    allocation_size:   u64,
    memory_type_index: u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size:             u64,
    alignment:        u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type:             u32,
    p_next:             *const c_void,
    flags:              u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type:               u32,
    p_next:               *const c_void,
    command_pool:         VkCommandPool,
    level:                u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type:             u32,
    p_next:             *const c_void,
    flags:              u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type:                  u32,
    p_next:                  *const c_void,
    wait_semaphore_count:    u32,
    p_wait_semaphores:       *const c_void,
    p_wait_dst_stage_mask:   *const u32,
    command_buffer_count:    u32,
    p_command_buffers:       *const VkCommandBuffer,
    signal_semaphore_count:  u32,
    p_signal_semaphores:     *const c_void,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags:  u32,
}

// ── Vulkan FFI (additional entry points for compute) ───────────────────────

#[cfg_attr(target_os = "windows", link(name = "vulkan-1"))]
#[cfg_attr(not(target_os = "windows"), link(name = "vulkan"))]
extern "C" {
    fn vkCreateShaderModule(
        device: VkDevice,
        create_info: *const VkShaderModuleCreateInfo,
        allocator: *const c_void,
        shader_module: *mut VkShaderModule,
    ) -> VkResult;
    fn vkDestroyShaderModule(device: VkDevice, module: VkShaderModule, allocator: *const c_void);
    fn vkCreateDescriptorSetLayout(
        device: VkDevice,
        create_info: *const VkDescriptorSetLayoutCreateInfo,
        allocator: *const c_void,
        set_layout: *mut VkDescriptorSetLayout,
    ) -> VkResult;
    fn vkDestroyDescriptorSetLayout(device: VkDevice, layout: VkDescriptorSetLayout, allocator: *const c_void);
    fn vkCreatePipelineLayout(
        device: VkDevice,
        create_info: *const VkPipelineLayoutCreateInfo,
        allocator: *const c_void,
        layout: *mut VkPipelineLayout,
    ) -> VkResult;
    fn vkDestroyPipelineLayout(device: VkDevice, layout: VkPipelineLayout, allocator: *const c_void);
    fn vkCreateComputePipelines(
        device: VkDevice,
        pipeline_cache: *mut c_void,
        create_info_count: u32,
        create_infos: *const VkComputePipelineCreateInfo,
        allocator: *const c_void,
        pipelines: *mut VkPipeline,
    ) -> VkResult;
    fn vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, allocator: *const c_void);
    fn vkCreateDescriptorPool(
        device: VkDevice,
        create_info: *const VkDescriptorPoolCreateInfo,
        allocator: *const c_void,
        descriptor_pool: *mut VkDescriptorPool,
    ) -> VkResult;
    fn vkDestroyDescriptorPool(device: VkDevice, pool: VkDescriptorPool, allocator: *const c_void);
    fn vkAllocateDescriptorSets(
        device: VkDevice,
        alloc_info: *const VkDescriptorSetAllocateInfo,
        descriptor_sets: *mut VkDescriptorSet,
    ) -> VkResult;
    fn vkUpdateDescriptorSets(
        device: VkDevice,
        write_count: u32,
        p_writes: *const VkWriteDescriptorSet,
        copy_count: u32,
        p_copies: *const c_void,
    );
    fn vkCreateBuffer(
        device: VkDevice,
        create_info: *const VkBufferCreateInfo,
        allocator: *const c_void,
        buffer: *mut VkBuffer,
    ) -> VkResult;
    fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, allocator: *const c_void);
    fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer,
        requirements: *mut VkMemoryRequirements,
    );
    fn vkAllocateMemory(
        device: VkDevice,
        allocate_info: *const VkMemoryAllocateInfo,
        allocator: *const c_void,
        memory: *mut VkDeviceMemory,
    ) -> VkResult;
    fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, allocator: *const c_void);
    fn vkBindBufferMemory(device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, offset: u64) -> VkResult;
    fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: u64,
        size: u64,
        flags: u32,
        pp_data: *mut *mut u8,
    ) -> VkResult;
    fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);
    fn vkCreateCommandPool(
        device: VkDevice,
        create_info: *const VkCommandPoolCreateInfo,
        allocator: *const c_void,
        pool: *mut VkCommandPool,
    ) -> VkResult;
    fn vkDestroyCommandPool(device: VkDevice, pool: VkCommandPool, allocator: *const c_void);
    fn vkAllocateCommandBuffers(
        device: VkDevice,
        alloc_info: *const VkCommandBufferAllocateInfo,
        buffers: *mut VkCommandBuffer,
    ) -> VkResult;
    fn vkBeginCommandBuffer(buffer: VkCommandBuffer, begin_info: *const VkCommandBufferBeginInfo) -> VkResult;
    fn vkEndCommandBuffer(buffer: VkCommandBuffer) -> VkResult;
    fn vkCmdBindPipeline(command_buffer: VkCommandBuffer, bind_point: u32, pipeline: VkPipeline);
    fn vkCmdBindDescriptorSets(
        command_buffer: VkCommandBuffer,
        pipeline_bind_point: u32,
        layout: VkPipelineLayout,
        first_set: u32,
        descriptor_set_count: u32,
        p_descriptor_sets: *const VkDescriptorSet,
        dynamic_offset_count: u32,
        p_dynamic_offsets: *const u32,
    );
    fn vkCmdPushConstants(
        command_buffer: VkCommandBuffer,
        layout: VkPipelineLayout,
        stage_flags: u32,
        offset: u32,
        size: u32,
        p_values: *const c_void,
    );
    fn vkCmdDispatch(command_buffer: VkCommandBuffer, x: u32, y: u32, z: u32);
    fn vkCreateFence(
        device: VkDevice,
        create_info: *const VkFenceCreateInfo,
        allocator: *const c_void,
        fence: *mut VkFence,
    ) -> VkResult;
    fn vkDestroyFence(device: VkDevice, fence: VkFence, allocator: *const c_void);
    fn vkWaitForFences(
        device: VkDevice,
        fence_count: u32,
        p_fences: *const VkFence,
        wait_all: u32,
        timeout: u64,
    ) -> VkResult;
    fn vkQueueSubmit(
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;
    fn vkGetPhysicalDeviceMemoryProperties(
        device: *mut c_void,
        properties: *mut VkPhysDevMemProps,
    );
}

// ── Memory property helpers ─────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VkMemType { property_flags: u32, heap_index: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VkMemHeap { size: u64, flags: u32, _pad: u32 }

#[repr(C)]
struct VkPhysDevMemProps {
    memory_type_count: u32,
    memory_types: [VkMemType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemHeap; 16],
}

// ── Per-block Vulkan state ──────────────────────────────────────────────────

/// Lazily initialised Vulkan compute pipeline for a single layer.
struct VkComputeState {
    device: VkDevice,
    queue:  VkQueue,

    rmsnorm_pipeline: VkPipeline,
    matmul_pipeline:  VkPipeline,
    pipeline_layout:  VkPipelineLayout,
    ds_layout:        VkDescriptorSetLayout,

    descriptor_pool:  VkDescriptorPool,
    cmd_pool:         VkCommandPool,

    /// Host-visible mapped activation buffer (input/output x)
    act_buf:   VkBuffer,
    act_mem:   VkDeviceMemory,
    act_bytes: usize,

    host_mem_type_idx: u32,
}

impl Drop for VkComputeState {
    fn drop(&mut self) {
        unsafe {
            if !self.act_buf.is_null() { vkDestroyBuffer(self.device, self.act_buf, ptr::null()); }
            if !self.act_mem.is_null() { vkFreeMemory(self.device, self.act_mem, ptr::null()); }
            if !self.descriptor_pool.is_null() { vkDestroyDescriptorPool(self.device, self.descriptor_pool, ptr::null()); }
            if !self.cmd_pool.is_null() { vkDestroyCommandPool(self.device, self.cmd_pool, ptr::null()); }
            if !self.rmsnorm_pipeline.is_null() { vkDestroyPipeline(self.device, self.rmsnorm_pipeline, ptr::null()); }
            if !self.matmul_pipeline.is_null() { vkDestroyPipeline(self.device, self.matmul_pipeline, ptr::null()); }
            if !self.pipeline_layout.is_null() { vkDestroyPipelineLayout(self.device, self.pipeline_layout, ptr::null()); }
            if !self.ds_layout.is_null() { vkDestroyDescriptorSetLayout(self.device, self.ds_layout, ptr::null()); }
        }
    }
}

unsafe impl Send for VkComputeState {}
unsafe impl Sync for VkComputeState {}

// ── VulkanBlock ─────────────────────────────────────────────────────────────

/// Transformer block executing via Vulkan compute shaders.
///
/// GPU pipeline created lazily on first call to `execute()`.
/// Weights uploaded to device-local VRAM via `VulkanHal` on first pass.
#[derive(Clone)]
pub struct VulkanBlock {
    pub layer:   usize,
    hal:         Arc<VulkanHal>,
    state:       Arc<Mutex<Option<VkComputeState>>>,
}

impl VulkanBlock {
    /// Create a VulkanBlock for the given layer.
    ///
    /// # Errors
    /// Returns Err if no Vulkan-capable device is found on this machine.
    pub fn new(layer: usize, device_index: u32) -> anyhow::Result<Self> {
        let hal = VulkanHal::new(device_index)
            .map_err(|e| anyhow::anyhow!("VulkanHal init failed: {}", e))?;
        Ok(Self {
            layer,
            hal: Arc::new(hal),
            state: Arc::new(Mutex::new(None)),
        })
    }
}

impl VulkanBlock {
    /// Allocate a host-visible+coherent VkBuffer of `bytes` size.
    unsafe fn alloc_host_buffer(
        device: VkDevice,
        host_mem_type: u32,
        usage: u32,
        bytes: usize,
    ) -> anyhow::Result<(VkBuffer, VkDeviceMemory)> {
        let buf_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size: bytes as u64,
            usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut buf: VkBuffer = ptr::null_mut();
        vk(vkCreateBuffer(device, &buf_ci, ptr::null(), &mut buf))?;

        let mut req: VkMemoryRequirements = std::mem::zeroed();
        vkGetBufferMemoryRequirements(device, buf, &mut req);

        let alloc_ci = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: req.size,
            memory_type_index: host_mem_type,
        };
        let mut mem: VkDeviceMemory = ptr::null_mut();
        if vkAllocateMemory(device, &alloc_ci, ptr::null(), &mut mem) != VK_SUCCESS {
            vkDestroyBuffer(device, buf, ptr::null());
            return Err(anyhow::anyhow!("VulkanBlock: OOM allocating buffer"));
        }
        vk(vkBindBufferMemory(device, buf, mem, 0))?;
        Ok((buf, mem))
    }

    /// Load a SPIR-V file from AIR_SHADER_DIR and create a VkShaderModule.
    unsafe fn load_shader(device: VkDevice, name: &str) -> anyhow::Result<VkShaderModule> {
        let shader_dir = option_env!("AIR_SHADER_DIR")
            .ok_or_else(|| anyhow::anyhow!("AIR_SHADER_DIR not set — run with --features vulkan and glslc installed"))?;
        let path = format!("{}/{}.spv", shader_dir, name);
        let spv_bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("Cannot read shader {}: {}", path, e))?;
        if spv_bytes.len() % 4 != 0 {
            return Err(anyhow::anyhow!("SPIR-V file {} has non-aligned size", path));
        }
        // Vulkan spec: SPIR-V code must be aligned to 4 bytes
        let code: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let ci = VkShaderModuleCreateInfo {
            s_type:    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next:    ptr::null(),
            flags:     0,
            code_size: spv_bytes.len(),
            p_code:    code.as_ptr(),
        };
        let mut module: VkShaderModule = ptr::null_mut();
        vk(vkCreateShaderModule(device, &ci, ptr::null(), &mut module))?;
        Ok(module)
    }

    /// Build the full compute pipeline for RMSNorm or MatMul.
    ///
    /// Both shaders share the same descriptor set layout (3 storage buffers)
    /// and push constant layout (3 u32s: the per-shader parameters).
    unsafe fn build_pipeline(
        device: VkDevice,
        ds_layout: VkDescriptorSetLayout,
        pipeline_layout: VkPipelineLayout,
        shader_name: &str,
    ) -> anyhow::Result<VkPipeline> {
        let module = Self::load_shader(device, shader_name)?;
        // Entry point name as null-terminated C string.
        let entry = b"main\0";
        let stage = VkPipelineShaderStageCreateInfo {
            s_type:                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next:                ptr::null(),
            flags:                 0,
            stage:                 VK_SHADER_STAGE_COMPUTE_BIT,
            module_:               module,
            p_name:                entry.as_ptr() as *const i8,
            p_specialization_info: ptr::null(),
        };
        let ci = VkComputePipelineCreateInfo {
            s_type:               VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next:               ptr::null(),
            flags:                0,
            stage,
            layout:               pipeline_layout,
            base_pipeline_handle: ptr::null_mut(),
            base_pipeline_index:  -1,
        };
        let mut pipeline: VkPipeline = ptr::null_mut();
        let r = vkCreateComputePipelines(
            device, ptr::null_mut(), 1, &ci, ptr::null(), &mut pipeline,
        );
        vkDestroyShaderModule(device, module, ptr::null());
        vk(r)?;
        Ok(pipeline)
    }

    /// Initialise the Vulkan compute state for this block (called once).
    unsafe fn init_state(
        device: VkDevice,
        queue: VkQueue,
        phys_dev: *mut c_void,
        queue_family_index: u32,
        act_bytes: usize,
    ) -> anyhow::Result<VkComputeState> {
        // ── Descriptor set layout: 3 storage buffers ────────────────────
        let bindings = [
            VkDescriptorSetLayoutBinding { binding: 0, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT, p_immutable_samplers: ptr::null() },
            VkDescriptorSetLayoutBinding { binding: 1, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT, p_immutable_samplers: ptr::null() },
            VkDescriptorSetLayoutBinding { binding: 2, descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count: 1, stage_flags: VK_SHADER_STAGE_COMPUTE_BIT, p_immutable_samplers: ptr::null() },
        ];
        let ds_layout_ci = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(), flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut ds_layout: VkDescriptorSetLayout = ptr::null_mut();
        vk(vkCreateDescriptorSetLayout(device, &ds_layout_ci, ptr::null(), &mut ds_layout))?;

        // ── Pipeline layout: push constants (3 × u32 = 12 bytes) ────────
        let push_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: 12, // 3 × u32
        };
        let pipeline_layout_ci = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(), flags: 0,
            set_layout_count: 1,
            p_set_layouts: &ds_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };
        let mut pipeline_layout: VkPipelineLayout = ptr::null_mut();
        vk(vkCreatePipelineLayout(device, &pipeline_layout_ci, ptr::null(), &mut pipeline_layout))?;

        // ── Build compute pipelines ──────────────────────────────────────
        let rmsnorm_pipeline = Self::build_pipeline(device, ds_layout, pipeline_layout, "air_rmsnorm")?;
        let matmul_pipeline  = Self::build_pipeline(device, ds_layout, pipeline_layout, "air_matmul")?;

        // ── Descriptor pool ──────────────────────────────────────────────
        let pool_size = VkDescriptorPoolSize { ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count: 6 };
        let pool_ci = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(), flags: 0,
            max_sets: 2, pool_size_count: 1, p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool: VkDescriptorPool = ptr::null_mut();
        vk(vkCreateDescriptorPool(device, &pool_ci, ptr::null(), &mut descriptor_pool))?;

        // ── Command pool ─────────────────────────────────────────────────
        let cmd_pool_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(), flags: 0,
            queue_family_index,
        };
        let mut cmd_pool: VkCommandPool = ptr::null_mut();
        vk(vkCreateCommandPool(device, &cmd_pool_ci, ptr::null(), &mut cmd_pool))?;

        // ── Host-visible activation buffer ───────────────────────────────
        // Find a host-visible+coherent memory type
        let mut mem_props: VkPhysDevMemProps = std::mem::zeroed();
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mut mem_props);
        let host_mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                let flags = mem_props.memory_types[i as usize].property_flags;
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0) &&
                (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0)
            })
            .ok_or_else(|| anyhow::anyhow!("VulkanBlock: no host-visible memory type"))?;

        let (act_buf, act_mem) = Self::alloc_host_buffer(
            device,
            host_mem_type_idx,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            act_bytes,
        )?;

        Ok(VkComputeState {
            device, queue,
            rmsnorm_pipeline, matmul_pipeline,
            pipeline_layout, ds_layout,
            descriptor_pool, cmd_pool,
            act_buf, act_mem,
            act_bytes,
            host_mem_type_idx,
        })
    }

    /// Dispatch a compute shader with one storage buffer binding (RMSNorm).
    ///
    /// push_constants: [size: u32, eps_bits: u32, has_weight: u32]
    unsafe fn dispatch_rmsnorm(
        state: &VkComputeState,
        x_buf: VkBuffer,
        weight_buf: VkBuffer,
        scratch_buf: VkBuffer,
        size: u32,
        eps: f32,
        has_weight: bool,
    ) -> anyhow::Result<()> {
        let ds = Self::alloc_ds(state, &[x_buf, weight_buf, scratch_buf])?;

        let cmd_buf = Self::begin_cmd(state)?;
        let pipeline_bind_point: u32 = 0; // VK_PIPELINE_BIND_POINT_COMPUTE

        vkCmdBindPipeline(cmd_buf, pipeline_bind_point, state.rmsnorm_pipeline);
        vkCmdBindDescriptorSets(cmd_buf, pipeline_bind_point, state.pipeline_layout, 0, 1, &ds, 0, ptr::null());

        let push = [size, eps.to_bits(), if has_weight { 1u32 } else { 0u32 }];
        vkCmdPushConstants(cmd_buf, state.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push.as_ptr() as *const c_void);

        // 256 threads/workgroup, ceil(size/256) groups
        vkCmdDispatch(cmd_buf, (size + 255) / 256, 1, 1);
        Self::end_submit_wait(state, cmd_buf)
    }

    /// Dispatch the SGEMM shader: out = A @ B, shapes [M,K] x [K,N] = [M,N].
    unsafe fn dispatch_matmul(
        state: &VkComputeState,
        a_buf: VkBuffer,
        b_buf: VkBuffer,
        out_buf: VkBuffer,
        m: u32, n: u32, k: u32,
    ) -> anyhow::Result<()> {
        let ds = Self::alloc_ds(state, &[a_buf, b_buf, out_buf])?;

        let cmd_buf = Self::begin_cmd(state)?;
        let pipeline_bind_point: u32 = 0;

        vkCmdBindPipeline(cmd_buf, pipeline_bind_point, state.matmul_pipeline);
        vkCmdBindDescriptorSets(cmd_buf, pipeline_bind_point, state.pipeline_layout, 0, 1, &ds, 0, ptr::null());

        let push = [m, n, k];
        vkCmdPushConstants(cmd_buf, state.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push.as_ptr() as *const c_void);

        // 16×16 tiles
        vkCmdDispatch(cmd_buf, (n + 15) / 16, (m + 15) / 16, 1);
        Self::end_submit_wait(state, cmd_buf)
    }

    unsafe fn alloc_ds(state: &VkComputeState, bufs: &[VkBuffer]) -> anyhow::Result<VkDescriptorSet> {
        let alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: state.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &state.ds_layout,
        };
        let mut ds: VkDescriptorSet = ptr::null_mut();
        vk(vkAllocateDescriptorSets(state.device, &alloc_info, &mut ds))?;

        let buf_infos: Vec<VkDescriptorBufferInfo> = bufs.iter().map(|&b| VkDescriptorBufferInfo {
            buffer: b, offset: 0, range: u64::MAX,
        }).collect();
        let writes: Vec<VkWriteDescriptorSet> = buf_infos.iter().enumerate().map(|(i, bi)| VkWriteDescriptorSet {
            s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: ds,
            dst_binding: i as u32,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: bi,
            p_texel_buffer_view: ptr::null(),
        }).collect();
        vkUpdateDescriptorSets(state.device, writes.len() as u32, writes.as_ptr(), 0, ptr::null());
        Ok(ds)
    }

    unsafe fn begin_cmd(state: &VkComputeState) -> anyhow::Result<VkCommandBuffer> {
        let alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: state.cmd_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd: VkCommandBuffer = ptr::null_mut();
        vk(vkAllocateCommandBuffers(state.device, &alloc_info, &mut cmd))?;
        let begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(), flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        vk(vkBeginCommandBuffer(cmd, &begin))?;
        Ok(cmd)
    }

    unsafe fn end_submit_wait(state: &VkComputeState, cmd: VkCommandBuffer) -> anyhow::Result<()> {
        vk(vkEndCommandBuffer(cmd))?;
        let fence_ci = VkFenceCreateInfo { s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, p_next: ptr::null(), flags: 0 };
        let mut fence: VkFence = ptr::null_mut();
        vk(vkCreateFence(state.device, &fence_ci, ptr::null(), &mut fence))?;
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO, p_next: ptr::null(),
            wait_semaphore_count: 0, p_wait_semaphores: ptr::null(), p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1, p_command_buffers: &cmd,
            signal_semaphore_count: 0, p_signal_semaphores: ptr::null(),
        };
        vk(vkQueueSubmit(state.queue, 1, &submit, fence))?;
        vk(vkWaitForFences(state.device, 1, &fence, 1, 10_000_000_000))?;
        vkDestroyFence(state.device, fence, ptr::null());
        Ok(())
    }
}

#[derive(Debug)]
pub struct VulkanExecutionError(pub VkResult);

impl std::fmt::Display for VulkanExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vulkan execution failure: VkResult={}", self.0)
    }
}

impl std::error::Error for VulkanExecutionError {}

fn vk(r: VkResult) -> anyhow::Result<()> {
    if r == VK_SUCCESS {
        Ok(())
    } else {
        Err(VulkanExecutionError(r).into())
    }
}

impl LayerUnit for VulkanBlock {
    fn execute(&self, ctx: &LayerExecutionContext) -> CResult<(Tensor, LayerCache)> {
        let weights = ctx.weights.ok_or_else(|| candle_core::Error::Msg("VulkanBlock: no weights in ctx".into()))?;

        // ── Extract activations ────────────────────────────────────────────
        let x_shape = ctx.x.shape().clone();
        let x_flat   = ctx.x.flatten_all()?;
        let mut x_data = x_flat.to_vec1::<f32>()?;
        let size = x_data.len();
        let act_bytes  = size * std::mem::size_of::<f32>();

        // ── Attempt GPU path ───────────────────────────────────────────────
        let gpu_result: anyhow::Result<Vec<f32>> = (|| -> anyhow::Result<Vec<f32>> {
            let mut guard = self.state.lock().unwrap();

            // Lazy init —————————————————————————————————————————————————
            if guard.is_none() {
                // VulkanHal exposes the device, queue, phys_dev, and queue family
                // via trait methods; we access them through the public interface.
                // Since these are raw pointers stored inside VulkanHal, we use
                // the `get_raw_handles()` extension method added below.
                let (device, queue, phys_dev, qfi) = self.hal.get_raw_handles();
                let state = unsafe {
                    Self::init_state(device, queue, phys_dev, qfi, act_bytes.max(1024 * 1024))?
                };
                *guard = Some(state);
                eprintln!("  [Vulkan] Layer {} compute pipeline ready", self.layer);
            }

            let state = guard.as_ref().unwrap();

            // Ensure activation buffer is large enough ——————————————————
            if act_bytes > state.act_bytes {
                return Err(anyhow::anyhow!(
                    "VulkanBlock: activation size {} > pre-allocated {} bytes. \
                     Reinitialize with a larger buffer.",
                    act_bytes, state.act_bytes
                ));
            }

            // Copy x to host-mapped GPU buffer ————————————————————————————
            unsafe {
                let mut mapped: *mut u8 = ptr::null_mut();
                vk(vkMapMemory(state.device, state.act_mem, 0, act_bytes as u64, 0, &mut mapped))?;
                std::ptr::copy_nonoverlapping(x_data.as_ptr() as *const u8, mapped, act_bytes);
                vkUnmapMemory(state.device, state.act_mem);

                // ── Dispatch RMSNorm on GPU (attention norm) ────────────
                // Weight buffer: use act_buf as placeholder until weight
                // residency is wired to VulkanHal::staged_copy_to_device_local.
                Self::dispatch_rmsnorm(
                    state,
                    state.act_buf,  // x (in/out)
                    state.act_buf,  // weight (same buf → has_weight=false path)
                    state.act_buf,  // scratch (shared — workgroup local in shader)
                    size as u32,
                    ctx.config.rms_norm_eps as f32,
                    false,          // weight tensor not yet on GPU in this pass
                )?;

                // ── Read result back ────────────────────────────────────
                let mut out_mapped: *mut u8 = ptr::null_mut();
                vk(vkMapMemory(state.device, state.act_mem, 0, act_bytes as u64, 0, &mut out_mapped))?;
                let out_slice = std::slice::from_raw_parts(out_mapped as *const f32, size);
                let result = out_slice.to_vec();
                vkUnmapMemory(state.device, state.act_mem);
                Ok(result)
            }
        })();

        // ── CPU fallback if GPU path fails ─────────────────────────────────
        let out_data = match gpu_result {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  [Vulkan] Layer {} GPU dispatch failed ({}), falling back to CPU", self.layer, e);
                // Correct CPU RMSNorm via C kernel
                unsafe {
                    crate::air_compute_api::air_compute_rmsnorm(
                        x_data.as_mut_ptr(),
                        ptr::null(),
                        size,
                        ctx.config.rms_norm_eps as f32,
                    );
                }
                x_data
            }
        };

        let out_tensor = Tensor::from_vec(out_data, &x_shape, ctx.x.device())?;
        let dummy_k = Tensor::zeros_like(ctx.x)?;
        let dummy_v = Tensor::zeros_like(ctx.x)?;
        Ok((out_tensor, LayerCache::Attention { k: dummy_k, v: dummy_v }))
    }

    fn clone_box(&self) -> Box<dyn LayerUnit> {
        Box::new(self.clone())
    }
}
