module storage

import vsl.vulkan

// VulkanStorageParams provides optional parameters for creating VulkanStorage.
@[params]
pub struct VulkanStorageParams {
	device &vulkan.Device = unsafe { nil }
}

// VulkanStorage wraps a Vulkan GPU buffer as a VTL storage backend.
// Mirrors the VclStorage pattern from storage/vcl_d_vcl.v.
@[heap]
pub struct VulkanStorage[T] {
pub mut:
	data   &vulkan.GpuBuffer
	nelems int
}

// vulkan converts CPU storage to a Vulkan GPU buffer.
// Returns a new VulkanStorage with the data uploaded to the GPU.
pub fn (cpu &CpuStorage[T]) vulkan(params VulkanStorageParams) !&VulkanStorage[T] {
	mut device := params.device

	// Use shared default device if none specified
	if isnil(device) {
		device = get_vulkan_device()!
	}

	n := cpu.data.len
	esz := usize(sizeof(T))
	size := vulkan.DeviceSize(usize(n) * esz)

	mut buf := device.buffer(size)!
	mut bytes := []u8{len: int(size)}
	unsafe { C.memcpy(bytes.data, cpu.data.data, usize(size)) }
	buf.load(bytes)!

	return &VulkanStorage[T]{
		data:   buf
		nelems: n
	}
}

// cpu reads data from GPU back to CPU storage.
pub fn (s &VulkanStorage[T]) cpu() !&CpuStorage[T] {
	arr := s.to_array()!
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array reads data from GPU and returns as a []T.
@[inline]
pub fn (s &VulkanStorage[T]) to_array[T]() ![]T {
	mut bytes := []u8{len: int(s.data.size)}
	bytes = s.data.store(mut bytes) or { return error('VulkanStorage.to_array: store failed: ${err}') }
	mut result := []T{len: s.nelems}
	unsafe { C.memcpy(result.data, bytes.data, bytes.len) }
	return result
}

// release frees the GPU buffer.
@[inline]
pub fn (s &VulkanStorage[T]) release() ! {
	if !isnil(s.data) {
		s.data.release()
	}
}

// --------------------------------------------------------------------------
// Shared Vulkan device (lazy, module-level)
// --------------------------------------------------------------------------
__global g_vk_dev = &vulkan.Device(unsafe { nil })

// get_vulkan_device returns a shared Vulkan device, creating it on first call.
fn get_vulkan_device() !&vulkan.Device {
	d := unsafe { g_vk_dev }
	if isnil(d) {
		mut new_d := vulkan.new_device()!
		unsafe {
			g_vk_dev = new_d
		}
		return new_d
	}
	return d
}
