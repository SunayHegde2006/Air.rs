//! Vulkan GPU HAL backend (STRIX Protocol §12.1).
//!
//! `VulkanHal` implements `GpuHal` via Vulkan 1.2 FFI bindings.
//! All Vulkan functions are loaded from `extern "C"` declarations —
//! the linker resolves against `vulkan-1` (Windows) or `vulkan` (Linux).
//!
//! Uses `Mutex`-based interior mutability (matching `CpuHal`) so every
//! `GpuHal` trait method works through `&self`.
//!
//! Gated behind `#[cfg(feature = "vulkan")]`.

#![cfg(feature = "vulkan")]

use super::hal::{GpuHal, GpuInfo, HalError};
use super::types::GpuPtr;
use std::collections::HashMap;
use std::ptr;
use std::sync::Mutex;

// ── Vulkan Type Aliases ──────────────────────────────────────────────────

type VkInstance = *mut std::ffi::c_void;
type VkPhysicalDevice = *mut std::ffi::c_void;
type VkDevice = *mut std::ffi::c_void;
type VkDeviceMemory = *mut std::ffi::c_void;
type VkQueue = *mut std::ffi::c_void;
type VkBuffer = *mut std::ffi::c_void;
type VkCommandPool = *mut std::ffi::c_void;
type VkCommandBuffer = *mut std::ffi::c_void;
type VkFence = *mut std::ffi::c_void;
type VkResult = i32;

const VK_SUCCESS: VkResult = 0;
const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResult = -2;
const VK_ERROR_OUT_OF_HOST_MEMORY: VkResult = -1;
const VK_TIMEOUT: VkResult = 2;

// ── Vulkan Structure Type Enums ──────────────────────────────────────────

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;

/// Vulkan buffer usage flags.
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x01;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x02;
const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;

/// Vulkan sharing mode.
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

/// Vulkan command pool flags.
const VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: u32 = 0x01;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x02;

/// Vulkan command buffer level.
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;

/// Vulkan command buffer usage flags.
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x01;

/// Vulkan memory property flags.
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x01;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x02;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x04;

/// Vulkan queue flags.
const VK_QUEUE_TRANSFER_BIT: u32 = 0x04;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x02;
const VK_QUEUE_GRAPHICS_BIT: u32 = 0x01;

// ── Vulkan Structs ───────────────────────────────────────────────────────

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    p_application_name: *const i8,
    application_version: u32,
    p_engine_name: *const i8,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const i8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const i8,
}

#[repr(C)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; 16],
    // VkPhysicalDeviceLimits + VkPhysicalDeviceSparseProperties follow.
    // We over-pad to avoid reading into undefined memory.
    _rest: [u8; 1024],
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3], // VkExtent3D
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const i8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const i8,
    p_enabled_features: *const std::ffi::c_void,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

// ── Vulkan Command Buffer / Staging Structs ──────────────────────────────

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    p_inheritance_info: *const std::ffi::c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkBufferCopy {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const std::ffi::c_void,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const std::ffi::c_void,
}

// ── Vulkan FFI ───────────────────────────────────────────────────────────

#[cfg_attr(target_os = "windows", link(name = "vulkan-1"))]
#[cfg_attr(not(target_os = "windows"), link(name = "vulkan"))]
extern "C" {
    fn vkCreateInstance(
        create_info: *const VkInstanceCreateInfo,
        allocator: *const std::ffi::c_void,
        instance: *mut VkInstance,
    ) -> VkResult;
    fn vkDestroyInstance(instance: VkInstance, allocator: *const std::ffi::c_void);
    fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        count: *mut u32,
        devices: *mut VkPhysicalDevice,
    ) -> VkResult;
    fn vkGetPhysicalDeviceProperties(
        device: VkPhysicalDevice,
        properties: *mut VkPhysicalDeviceProperties,
    );
    fn vkGetPhysicalDeviceMemoryProperties(
        device: VkPhysicalDevice,
        properties: *mut VkPhysicalDeviceMemoryProperties,
    );
    fn vkGetPhysicalDeviceQueueFamilyProperties(
        device: VkPhysicalDevice,
        count: *mut u32,
        properties: *mut VkQueueFamilyProperties,
    );
    fn vkCreateDevice(
        physical_device: VkPhysicalDevice,
        create_info: *const VkDeviceCreateInfo,
        allocator: *const std::ffi::c_void,
        device: *mut VkDevice,
    ) -> VkResult;
    fn vkDestroyDevice(device: VkDevice, allocator: *const std::ffi::c_void);
    fn vkGetDeviceQueue(
        device: VkDevice,
        queue_family_index: u32,
        queue_index: u32,
        queue: *mut VkQueue,
    );
    fn vkAllocateMemory(
        device: VkDevice,
        allocate_info: *const VkMemoryAllocateInfo,
        allocator: *const std::ffi::c_void,
        memory: *mut VkDeviceMemory,
    ) -> VkResult;
    fn vkFreeMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        allocator: *const std::ffi::c_void,
    );
    fn vkMapMemory(
        device: VkDevice,
        memory: VkDeviceMemory,
        offset: u64,
        size: u64,
        flags: u32,
        pp_data: *mut *mut u8,
    ) -> VkResult;
    fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);

    // ── Command Buffer / Staging FFI ─────────────────────────────────
    fn vkCreateBuffer(
        device: VkDevice,
        create_info: *const VkBufferCreateInfo,
        allocator: *const std::ffi::c_void,
        buffer: *mut VkBuffer,
    ) -> VkResult;
    fn vkDestroyBuffer(
        device: VkDevice,
        buffer: VkBuffer,
        allocator: *const std::ffi::c_void,
    );
    fn vkGetBufferMemoryRequirements(
        device: VkDevice,
        buffer: VkBuffer,
        requirements: *mut VkMemoryRequirements,
    );
    fn vkBindBufferMemory(
        device: VkDevice,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        offset: u64,
    ) -> VkResult;
    fn vkCreateCommandPool(
        device: VkDevice,
        create_info: *const VkCommandPoolCreateInfo,
        allocator: *const std::ffi::c_void,
        pool: *mut VkCommandPool,
    ) -> VkResult;
    fn vkDestroyCommandPool(
        device: VkDevice,
        pool: VkCommandPool,
        allocator: *const std::ffi::c_void,
    );
    fn vkAllocateCommandBuffers(
        device: VkDevice,
        alloc_info: *const VkCommandBufferAllocateInfo,
        buffers: *mut VkCommandBuffer,
    ) -> VkResult;
    fn vkBeginCommandBuffer(
        buffer: VkCommandBuffer,
        begin_info: *const VkCommandBufferBeginInfo,
    ) -> VkResult;
    fn vkEndCommandBuffer(buffer: VkCommandBuffer) -> VkResult;
    fn vkCmdCopyBuffer(
        command_buffer: VkCommandBuffer,
        src_buffer: VkBuffer,
        dst_buffer: VkBuffer,
        region_count: u32,
        p_regions: *const VkBufferCopy,
    );
    fn vkCreateFence(
        device: VkDevice,
        create_info: *const VkFenceCreateInfo,
        allocator: *const std::ffi::c_void,
        fence: *mut VkFence,
    ) -> VkResult;
    fn vkDestroyFence(
        device: VkDevice,
        fence: VkFence,
        allocator: *const std::ffi::c_void,
    );
    fn vkWaitForFences(
        device: VkDevice,
        fence_count: u32,
        p_fences: *const VkFence,
        wait_all: u32,
        timeout: u64,
    ) -> VkResult;
    fn vkResetFences(
        device: VkDevice,
        fence_count: u32,
        p_fences: *const VkFence,
    ) -> VkResult;
    fn vkQueueSubmit(
        queue: VkQueue,
        submit_count: u32,
        p_submits: *const VkSubmitInfo,
        fence: VkFence,
    ) -> VkResult;
    fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn vk_check(result: VkResult) -> Result<(), HalError> {
    match result {
        VK_SUCCESS => Ok(()),
        VK_ERROR_OUT_OF_DEVICE_MEMORY | VK_ERROR_OUT_OF_HOST_MEMORY => {
            Err(HalError::OutOfMemory {
                requested: 0,
                available: 0,
            })
        }
        code => Err(HalError::DriverError {
            code,
            message: format!("Vulkan error (VkResult = {code})"),
        }),
    }
}

// ── Internal Types ───────────────────────────────────────────────────────

/// A tracked Vulkan device memory allocation.
struct VulkanAllocation {
    /// The `VkDeviceMemory` handle.
    memory: VkDeviceMemory,
    /// Allocation size in bytes.
    size: usize,
}

/// Interior-mutable state for `VulkanHal`.
struct VulkanHalInner {
    /// Active allocations keyed by `VkDeviceMemory` address (as u64).
    allocations: HashMap<u64, VulkanAllocation>,
}

// ── VulkanHal ────────────────────────────────────────────────────────────

/// Vulkan GPU backend implementing `GpuHal`.
///
/// Creates a full Vulkan instance → physical device → logical device
/// pipeline during construction. Supports two memory strategies:
///
/// 1. **Host-visible+coherent** (default path): Direct `vkMapMemory`
///    for data transfer — works on all GPUs, fast on iGPU/UMA.
///
/// 2. **Staged transfer** (discrete GPU path): Uses a host-visible
///    staging buffer + `vkCmdCopyBuffer` in a one-shot command buffer
///    to upload data to device-local VRAM — optimal for discrete GPUs
///    where PCIe BAR size is limited.
pub struct VulkanHal {
    instance: VkInstance,
    device: VkDevice,
    /// Queue used for transfer/compute submission.
    queue: VkQueue,
    /// Queue family index for command pool creation.
    queue_family_index: u32,
    /// Selected memory type index (host-visible + coherent).
    memory_type_index: u32,
    /// Device-local-only memory type index (for discrete GPU staging).
    /// `None` on iGPU/UMA where device-local is also host-visible.
    device_local_type_index: Option<u32>,
    /// Total heap size for the selected memory type.
    total_vram: usize,
    /// Device name.
    device_name: String,
    /// Vulkan API version reported by the physical device.
    api_version: u32,
    /// Interior-mutable allocation tracking.
    inner: Mutex<VulkanHalInner>,
}

// SAFETY: VkDevice and VkInstance are thread-safe when externally synchronised.
// Our Mutex provides that synchronisation for the allocation map.
unsafe impl Send for VulkanHal {}
unsafe impl Sync for VulkanHal {}

impl VulkanHal {
    /// Create a new Vulkan HAL, selecting the given physical device index.
    ///
    /// Performs the full Vulkan initialisation sequence:
    /// 1. Create `VkInstance`
    /// 2. Enumerate physical devices
    /// 3. Enumerate queue families → select one with transfer capability
    /// 4. Create `VkDevice` with a single queue
    /// 5. Query memory types → select host-visible+coherent type
    pub fn new(device_index: u32) -> Result<Self, HalError> {
        // ── 1. Create Instance ──────────────────────────────────────────
        let app_name = b"STRIX\0";
        let engine_name = b"Air.rs\0";
        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr() as *const i8,
            application_version: 1,
            p_engine_name: engine_name.as_ptr() as *const i8,
            engine_version: 1,
            api_version: (1 << 22) | (2 << 12), // Vulkan 1.2
        };
        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
        };
        let mut instance = ptr::null_mut();
        unsafe { vk_check(vkCreateInstance(&create_info, ptr::null(), &mut instance))? };

        // ── 2. Enumerate Physical Devices ────────────────────────────────
        let mut count = 0u32;
        unsafe {
            vk_check(vkEnumeratePhysicalDevices(instance, &mut count, ptr::null_mut()))?
        };
        if count == 0 {
            unsafe { vkDestroyInstance(instance, ptr::null()) };
            return Err(HalError::Unsupported("no Vulkan devices found".into()));
        }
        let mut devices = vec![ptr::null_mut(); count as usize];
        unsafe {
            vk_check(vkEnumeratePhysicalDevices(
                instance,
                &mut count,
                devices.as_mut_ptr(),
            ))?
        };

        let idx = device_index.min(count - 1) as usize;
        let physical_device = devices[idx];

        // ── 3. Get Device Properties ─────────────────────────────────────
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceProperties(physical_device, &mut props) };
        let name_len = props.device_name.iter().position(|&b| b == 0).unwrap_or(256);
        let device_name = String::from_utf8_lossy(&props.device_name[..name_len]).into_owned();
        let api_version = props.api_version;

        // ── 4. Find Queue Family with Transfer Support ───────────────────
        let mut qf_count = 0u32;
        unsafe {
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qf_count,
                ptr::null_mut(),
            )
        };
        let mut qf_props = vec![
            VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [0; 3],
            };
            qf_count as usize
        ];
        unsafe {
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut qf_count,
                qf_props.as_mut_ptr(),
            )
        };

        // Prefer a queue with transfer+compute; fall back to any queue.
        let mut selected_qf = 0u32;
        for (i, qf) in qf_props.iter().enumerate() {
            if qf.queue_count > 0
                && (qf.queue_flags & VK_QUEUE_TRANSFER_BIT != 0
                    || qf.queue_flags & VK_QUEUE_COMPUTE_BIT != 0
                    || qf.queue_flags & VK_QUEUE_GRAPHICS_BIT != 0)
            {
                selected_qf = i as u32;
                // Prefer compute/transfer over graphics-only.
                if qf.queue_flags & VK_QUEUE_COMPUTE_BIT != 0 {
                    break;
                }
            }
        }

        // ── 5. Create Logical Device ─────────────────────────────────────
        let queue_priority: f32 = 1.0;
        let queue_ci = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index: selected_qf,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
        };
        let device_ci = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            p_enabled_features: ptr::null(),
        };
        let mut device = ptr::null_mut();
        let result = unsafe {
            vkCreateDevice(physical_device, &device_ci, ptr::null(), &mut device)
        };
        if result != VK_SUCCESS {
            unsafe { vkDestroyInstance(instance, ptr::null()) };
            vk_check(result)?;
        }

        // Get queue handle.
        let mut queue = ptr::null_mut();
        unsafe { vkGetDeviceQueue(device, selected_qf, 0, &mut queue) };

        // ── 6. Select Memory Type ────────────────────────────────────────
        let mut mem_props: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_props) };

        // Strategy: find host-visible+coherent memory (mappable). This works
        // on all GPUs (integrated and discrete). On discrete GPUs the heap
        // may be smaller (PCIe BAR), but memory copies via map/unmap are
        // always correct.
        let mut memory_type_index = u32::MAX;
        let mut total_vram = 0usize;

        // First pass: look for device-local + host-visible + coherent (ideal for iGPU/UMA).
        for i in 0..mem_props.memory_type_count as usize {
            let mt = &mem_props.memory_types[i];
            let flags = mt.property_flags;
            if flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0
                && flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0
                && flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0
            {
                memory_type_index = i as u32;
                total_vram = mem_props.memory_heaps[mt.heap_index as usize].size as usize;
                break;
            }
        }

        // Second pass: fall back to any host-visible + coherent.
        if memory_type_index == u32::MAX {
            for i in 0..mem_props.memory_type_count as usize {
                let mt = &mem_props.memory_types[i];
                let flags = mt.property_flags;
                if flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0
                    && flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0
                {
                    memory_type_index = i as u32;
                    total_vram = mem_props.memory_heaps[mt.heap_index as usize].size as usize;
                    break;
                }
            }
        }

        if memory_type_index == u32::MAX {
            unsafe {
                vkDestroyDevice(device, ptr::null());
                vkDestroyInstance(instance, ptr::null());
            }
            return Err(HalError::Unsupported(
                "no host-visible+coherent memory type found".into(),
            ));
        }

        // Also find device-local-only type for staged transfers on discrete GPUs.
        let mut device_local_type_index = None;
        for i in 0..mem_props.memory_type_count as usize {
            let mt = &mem_props.memory_types[i];
            let flags = mt.property_flags;
            // Device-local but NOT host-visible = discrete VRAM
            if flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0
                && flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0
            {
                device_local_type_index = Some(i as u32);
                // Use the largest heap for total_vram
                let heap_size = mem_props.memory_heaps[mt.heap_index as usize].size as usize;
                if heap_size > total_vram {
                    total_vram = heap_size;
                }
                break;
            }
        }

        Ok(Self {
            instance,
            device,
            queue,
            queue_family_index: selected_qf,
            memory_type_index,
            device_local_type_index,
            total_vram,
            device_name,
            api_version,
            inner: Mutex::new(VulkanHalInner {
                allocations: HashMap::new(),
            }),
        })
    }

    /// Total discoverable VRAM for the selected heap.
    pub fn total_vram(&self) -> usize {
        self.total_vram
    }

    /// Whether the device has separate device-local VRAM (discrete GPU).
    pub fn has_device_local_vram(&self) -> bool {
        self.device_local_type_index.is_some()
    }

    /// Perform a staged copy to device-local VRAM.
    ///
    /// This is the optimal transfer path for discrete GPUs:
    /// 1. Allocate host-visible staging buffer
    /// 2. Map → memcpy data → unmap
    /// 3. Create command buffer with vkCmdCopyBuffer
    /// 4. Submit and fence-wait
    /// 5. Free staging resources
    ///
    /// Returns the device-local `VkDeviceMemory` handle as a `GpuPtr`.
    /// Falls back to host-visible allocation if no device-local type.
    pub fn staged_copy_to_device_local(
        &self,
        data: &[u8],
    ) -> Result<GpuPtr, HalError> {
        let size = data.len();
        let dl_type_idx = match self.device_local_type_index {
            Some(idx) => idx,
            None => {
                // No discrete VRAM — fall back to host-visible copy
                let ptr = self.allocate_vram(size, 256)?;
                self.copy_to_vram(ptr, data.as_ptr(), size, 0)?;
                return Ok(ptr);
            }
        };

        unsafe {
            // ── 1. Create destination buffer (device-local) ──────────
            let dst_buffer_ci = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                size: size as u64,
                usage: VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
            };
            let mut dst_buffer: VkBuffer = ptr::null_mut();
            vk_check(vkCreateBuffer(self.device, &dst_buffer_ci, ptr::null(), &mut dst_buffer))?;

            let mut dst_reqs: VkMemoryRequirements = std::mem::zeroed();
            vkGetBufferMemoryRequirements(self.device, dst_buffer, &mut dst_reqs);

            let dst_alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: dst_reqs.size,
                memory_type_index: dl_type_idx,
            };
            let mut dst_memory: VkDeviceMemory = ptr::null_mut();
            let result = vkAllocateMemory(self.device, &dst_alloc_info, ptr::null(), &mut dst_memory);
            if result != VK_SUCCESS {
                vkDestroyBuffer(self.device, dst_buffer, ptr::null());
                vk_check(result)?;
            }
            vk_check(vkBindBufferMemory(self.device, dst_buffer, dst_memory, 0))?;

            // ── 2. Create staging buffer (host-visible) ─────────────
            let src_buffer_ci = VkBufferCreateInfo {
                s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                size: size as u64,
                usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
            };
            let mut src_buffer: VkBuffer = ptr::null_mut();
            vk_check(vkCreateBuffer(self.device, &src_buffer_ci, ptr::null(), &mut src_buffer))?;

            let mut src_reqs: VkMemoryRequirements = std::mem::zeroed();
            vkGetBufferMemoryRequirements(self.device, src_buffer, &mut src_reqs);

            let src_alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: src_reqs.size,
                memory_type_index: self.memory_type_index,
            };
            let mut src_memory: VkDeviceMemory = ptr::null_mut();
            vk_check(vkAllocateMemory(self.device, &src_alloc_info, ptr::null(), &mut src_memory))?;
            vk_check(vkBindBufferMemory(self.device, src_buffer, src_memory, 0))?;

            // ── 3. Map staging → memcpy → unmap ─────────────────────
            let mut mapped: *mut u8 = ptr::null_mut();
            vk_check(vkMapMemory(self.device, src_memory, 0, size as u64, 0, &mut mapped))?;
            ptr::copy_nonoverlapping(data.as_ptr(), mapped, size);
            vkUnmapMemory(self.device, src_memory);

            // ── 4. Record command buffer ─────────────────────────────
            let pool_ci = VkCommandPoolCreateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                queue_family_index: self.queue_family_index,
            };
            let mut cmd_pool: VkCommandPool = ptr::null_mut();
            vk_check(vkCreateCommandPool(self.device, &pool_ci, ptr::null(), &mut cmd_pool))?;

            let cb_alloc_info = VkCommandBufferAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool: cmd_pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                command_buffer_count: 1,
            };
            let mut cmd_buf: VkCommandBuffer = ptr::null_mut();
            vk_check(vkAllocateCommandBuffers(self.device, &cb_alloc_info, &mut cmd_buf))?;

            let begin_info = VkCommandBufferBeginInfo {
                s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            vk_check(vkBeginCommandBuffer(cmd_buf, &begin_info))?;

            let copy_region = VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: size as u64,
            };
            vkCmdCopyBuffer(cmd_buf, src_buffer, dst_buffer, 1, &copy_region);
            vk_check(vkEndCommandBuffer(cmd_buf))?;

            // ── 5. Submit + fence-wait ───────────────────────────────
            let fence_ci = VkFenceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
            };
            let mut fence: VkFence = ptr::null_mut();
            vk_check(vkCreateFence(self.device, &fence_ci, ptr::null(), &mut fence))?;

            let submit_info = VkSubmitInfo {
                s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };
            vk_check(vkQueueSubmit(self.queue, 1, &submit_info, fence))?;

            // Wait with 5-second timeout
            let result = vkWaitForFences(self.device, 1, &fence, 1, 5_000_000_000);
            if result == VK_TIMEOUT {
                // Still clean up on timeout
                vkDestroyFence(self.device, fence, ptr::null());
                vkDestroyCommandPool(self.device, cmd_pool, ptr::null());
                vkDestroyBuffer(self.device, src_buffer, ptr::null());
                vkFreeMemory(self.device, src_memory, ptr::null());
                vkDestroyBuffer(self.device, dst_buffer, ptr::null());
                vkFreeMemory(self.device, dst_memory, ptr::null());
                return Err(HalError::Timeout);
            }
            vk_check(result)?;

            // ── 6. Cleanup staging resources ─────────────────────────
            vkDestroyFence(self.device, fence, ptr::null());
            vkDestroyCommandPool(self.device, cmd_pool, ptr::null());
            vkDestroyBuffer(self.device, src_buffer, ptr::null());
            vkFreeMemory(self.device, src_memory, ptr::null());
            // Keep dst_buffer alive — caller owns it via dst_memory
            vkDestroyBuffer(self.device, dst_buffer, ptr::null());

            // Track the device-local allocation
            let key = dst_memory as u64;
            let mut inner = self.inner.lock().unwrap();
            inner.allocations.insert(
                key,
                VulkanAllocation { memory: dst_memory, size },
            );

            Ok(GpuPtr(key))
        }
    }
}

impl Drop for VulkanHal {
    fn drop(&mut self) {
        // Free any remaining allocations.
        if let Ok(mut inner) = self.inner.lock() {
            let keys: Vec<u64> = inner.allocations.keys().copied().collect();
            for key in keys {
                if let Some(alloc) = inner.allocations.remove(&key) {
                    if !alloc.memory.is_null() {
                        unsafe { vkFreeMemory(self.device, alloc.memory, ptr::null()) };
                    }
                }
            }
        }
        // Destroy device BEFORE instance (Vulkan spec requirement).
        if !self.device.is_null() {
            unsafe { vkDestroyDevice(self.device, ptr::null()) };
        }
        if !self.instance.is_null() {
            unsafe { vkDestroyInstance(self.instance, ptr::null()) };
        }
    }
}

impl GpuHal for VulkanHal {
    fn info(&self) -> Result<GpuInfo, HalError> {
        let inner = self.inner.lock().unwrap();
        let used: usize = inner.allocations.values().map(|a| a.size).sum();
        Ok(GpuInfo {
            name: self.device_name.clone(),
            vram_total: self.total_vram,
            vram_free: self.total_vram.saturating_sub(used),
            // Encode Vulkan API version as a comparable integer.
            compute_capability: self.api_version >> 12,
            bus_bandwidth: 16_000_000_000, // conservative PCIe 3.0 x16
        })
    }

    fn allocate_vram(&self, size: usize, _alignment: usize) -> Result<GpuPtr, HalError> {
        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: size as u64,
            memory_type_index: self.memory_type_index,
        };
        let mut memory: VkDeviceMemory = ptr::null_mut();
        let result = unsafe {
            vkAllocateMemory(self.device, &alloc_info, ptr::null(), &mut memory)
        };
        vk_check(result)?;

        let key = memory as u64;
        let mut inner = self.inner.lock().unwrap();
        inner.allocations.insert(
            key,
            VulkanAllocation { memory, size },
        );
        Ok(GpuPtr(key))
    }

    fn free_vram(&self, ptr: GpuPtr) -> Result<(), HalError> {
        let mut inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .remove(&ptr.0)
            .ok_or_else(|| HalError::DriverError {
                code: -3,
                message: format!("double-free or unknown GpuPtr({:#x})", ptr.0),
            })?;
        unsafe { vkFreeMemory(self.device, alloc.memory, ptr::null()) };
        Ok(())
    }

    fn copy_to_vram(
        &self,
        dst: GpuPtr,
        src: *const u8,
        size: usize,
        _stream: u32,
    ) -> Result<(), HalError> {
        let inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get(&dst.0)
            .ok_or_else(|| HalError::DriverError {
                code: -1,
                message: format!("unknown GpuPtr({:#x})", dst.0),
            })?;
        let memory = alloc.memory;
        let alloc_size = alloc.size;
        drop(inner); // Release lock before blocking FFI call.

        if size > alloc_size {
            return Err(HalError::DriverError {
                code: -2,
                message: format!("write size {} exceeds alloc size {}", size, alloc_size),
            });
        }

        let mut mapped: *mut u8 = ptr::null_mut();
        let result = unsafe {
            vkMapMemory(self.device, memory, 0, size as u64, 0, &mut mapped)
        };
        vk_check(result)?;
        unsafe { ptr::copy_nonoverlapping(src, mapped, size) };
        unsafe { vkUnmapMemory(self.device, memory) };
        Ok(())
    }

    fn copy_from_vram(
        &self,
        dst: *mut u8,
        src: GpuPtr,
        size: usize,
        _stream: u32,
    ) -> Result<(), HalError> {
        let inner = self.inner.lock().unwrap();
        let alloc = inner
            .allocations
            .get(&src.0)
            .ok_or_else(|| HalError::DriverError {
                code: -1,
                message: format!("unknown GpuPtr({:#x})", src.0),
            })?;
        let memory = alloc.memory;
        let alloc_size = alloc.size;
        drop(inner);

        if size > alloc_size {
            return Err(HalError::DriverError {
                code: -2,
                message: format!("read size {} exceeds alloc size {}", size, alloc_size),
            });
        }

        let mut mapped: *mut u8 = ptr::null_mut();
        let result = unsafe {
            vkMapMemory(self.device, memory, 0, size as u64, 0, &mut mapped)
        };
        vk_check(result)?;
        unsafe { ptr::copy_nonoverlapping(mapped as *const u8, dst, size) };
        unsafe { vkUnmapMemory(self.device, memory) };
        Ok(())
    }

    fn sync_stream(&self, _stream: u32) -> Result<(), HalError> {
        // Memory copies via vkMapMemory/vkUnmapMemory on host-coherent
        // memory are synchronous — no fence needed.
        Ok(())
    }

    fn vram_used(&self) -> Result<usize, HalError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.allocations.values().map(|a| a.size).sum())
    }
}
